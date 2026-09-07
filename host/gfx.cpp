// BR2b-1: D3D12 device + swapchain + clear/present. The device is created as
// ID3D12Device5 (DXR-capable) so BR2b-2 can add ray tracing on the same device.
#include "gfx.h"
#include "bridge.h"
#include "dlss.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#define SAFE_RELEASE(p) do { if (p) { (p)->Release(); (p) = nullptr; } } while (0)
static const UINT kFrames = 2;
static const UINT LIGHT_SRV_SLOT = 10;
static const UINT SHARC_HASH_UAV_SLOT = 12;
static const UINT SHARC_ACCUM_UAV_SLOT = 13;
static const UINT SHARC_RESOLVED_UAV_SLOT = 14;
static const UINT RESTIR_RES0_UAV_SLOT = 15;
static const UINT RESTIR_RES1_UAV_SLOT = 16;
static const UINT RESTIR_SURF0_UAV_SLOT = 17;
static const UINT RESTIR_SURF1_UAV_SLOT = 18;
static const UINT FOG_UAV_SLOT = 19;
static const UINT SHARC_CAPACITY = 1u << 20; // 40 MiB total at the default 8/16/16-byte layouts
static const UINT TEX_SRV_BASE   = 21;       // slot 19 particles, slot 20 live screen sky

static ID3D12Device5*        g_dev = nullptr;
static ID3D12CommandQueue*   g_queue = nullptr;
static IDXGISwapChain3*      g_swap = nullptr;
static ID3D12DescriptorHeap* g_rtvHeap = nullptr;
static UINT                  g_rtvSize = 0;
static ID3D12Resource*       g_back[kFrames] = {};
static ID3D12CommandAllocator* g_alloc[kFrames] = {};
static ID3D12GraphicsCommandList4* g_cl = nullptr;
static ID3D12Fence*          g_fence = nullptr;
static UINT64                g_fenceVal = 0;
static UINT64                g_frameFenceVal[kFrames] = {};
static HANDLE                g_fenceEvent = nullptr;
static UINT                  g_frameIndex = 0;
static int                   g_w = 0, g_h = 0;     // output (window) resolution
static int                   g_rw = 0, g_rh = 0;   // ray-traced render resolution (<= output when upscaling)
static int                   g_dlssQuality = 1;     // 0=DLAA 1=Quality 2=Balanced 3=Performance 4=UltraPerf
static int                   g_jitterPhases = 8;    // Halton sequence length (scales with upscale ratio^2)
static float g_sunDir[3] = { 0.62f, 0.30f, 0.72f };  // cached real sun (falls back until the light pass is seen)
static float g_sunCol[3] = { 3.36f, 3.14f, 2.82f };
static bool  g_sunSeen   = false;
static bool  g_lightsSeen = false;
static bool  g_skySeen    = false;
struct CachedLight { BridgeLight light; uint32_t lastSeen; };
static std::vector<CachedLight> g_lightCache;
struct EmitterAnchor { float p[3]; uint32_t lastSeen; };
static std::vector<EmitterAnchor> g_emitterAnchors;

static void wait_gpu()
{
    const UINT64 v = ++g_fenceVal;
    g_queue->Signal(g_fence, v);
    if (g_fence->GetCompletedValue() < v) {
        g_fence->SetEventOnCompletion(v, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
}

static ID3D12InfoQueue* g_infoQ = nullptr;
void gfx_drain_messages()
{
    if (!g_infoQ) return;
    UINT64 n = g_infoQ->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T len = 0; g_infoQ->GetMessage(i, nullptr, &len);
        if (!len) continue;
        D3D12_MESSAGE* m = (D3D12_MESSAGE*)malloc(len);
        if (m && SUCCEEDED(g_infoQ->GetMessage(i, m, &len)))
            printf("[d3d12] sev=%d id=%d: %s\n", m->Severity, m->ID, m->pDescription);
        free(m);
    }
    g_infoQ->ClearStoredMessages();
}

bool gfx_init(HWND hwnd, int w, int h)
{
    g_w = w; g_h = h;

    // Debug layer (validation) -- surfaces binding/state errors via gfx_drain_messages.
    if (getenv("WOLF_DBG")) {
        ID3D12Debug* dbg = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) { dbg->EnableDebugLayer(); dbg->Release(); printf("[gfx] debug layer ON\n"); }
    }

    // DRED: capture the GPU page-fault address + last op if the device hangs.
    ID3D12DeviceRemovedExtendedDataSettings* dred = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
        dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        dred->Release();
    }

    IDXGIFactory6* factory = nullptr;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) { printf("[gfx] factory fail\n"); return false; }

    // First hardware adapter that gives a DXR-capable device.
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(
             i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 ad; adapter->GetDesc1(&ad);
        if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { SAFE_RELEASE(adapter); continue; }
        if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_dev)))) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
            g_dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5));
            printf("[gfx] device on %ls (RaytracingTier=%d)\n", ad.Description, (int)o5.RaytracingTier);
            SAFE_RELEASE(adapter);
            break;
        }
        SAFE_RELEASE(adapter);
    }
    if (!g_dev) { printf("[gfx] no D3D12 device\n"); return false; }
    g_dev->QueryInterface(IID_PPV_ARGS(&g_infoQ));

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    g_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue));

    DXGI_SWAP_CHAIN_DESC1 sc = {};
    sc.BufferCount = kFrames;
    sc.Width = w; sc.Height = h;
    sc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc.SampleDesc.Count = 1;
    IDXGISwapChain1* sc1 = nullptr;
    if (FAILED(factory->CreateSwapChainForHwnd(g_queue, hwnd, &sc, nullptr, nullptr, &sc1))) {
        printf("[gfx] swapchain fail\n"); return false;
    }
    sc1->QueryInterface(IID_PPV_ARGS(&g_swap));
    SAFE_RELEASE(sc1);
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    SAFE_RELEASE(factory);
    g_frameIndex = g_swap->GetCurrentBackBufferIndex();

    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.NumDescriptors = kFrames;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_rtvHeap));
    g_rtvSize = g_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = g_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kFrames; ++i) {
        g_swap->GetBuffer(i, IID_PPV_ARGS(&g_back[i]));
        g_dev->CreateRenderTargetView(g_back[i], nullptr, rtv);
        rtv.ptr += g_rtvSize;
        g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_alloc[i]));
    }
    g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc[0], nullptr, IID_PPV_ARGS(&g_cl));
    g_cl->Close();
    g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence));
    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    printf("[gfx] swapchain %dx%d ready\n", w, h);
    return true;
}

void gfx_clear_present(float r, float g, float b)
{
    if (!g_dev) return;
    UINT i = g_frameIndex;
    g_alloc[i]->Reset();
    g_cl->Reset(g_alloc[i], nullptr);

    D3D12_RESOURCE_BARRIER br = {};
    br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br.Transition.pResource = g_back[i];
    br.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    br.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cl->ResourceBarrier(1, &br);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = g_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += (SIZE_T)i * g_rtvSize;
    const float c[4] = { r, g, b, 1.0f };
    g_cl->ClearRenderTargetView(rtv, c, 0, nullptr);

    br.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    br.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_cl->ResourceBarrier(1, &br);
    g_cl->Close();

    ID3D12CommandList* lists[] = { g_cl };
    g_queue->ExecuteCommandLists(1, lists);
    g_swap->Present(1, 0);

    // Fully wait for THIS frame's GPU work before continuing -- prevents any
    // submission backlog that could hang the GPU/whole system.
    g_frameFenceVal[i] = ++g_fenceVal;
    g_queue->Signal(g_fence, g_fenceVal);
    if (g_fence->GetCompletedValue() < g_fenceVal) {
        g_fence->SetEventOnCompletion(g_fenceVal, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
    g_frameIndex = g_swap->GetCurrentBackBufferIndex();
}

// ============================ BR2b-2 : DXR path tracer ======================

static ID3D12RootSignature*  g_rtRS = nullptr;
static ID3D12RootSignature*  g_localRS = nullptr;      // per-instance geometry (hit group)
static ID3D12StateObject*    g_rtSO = nullptr;
static ID3D12Resource*       g_skyCB = nullptr;        // b2: multi-layer sky (cloud layer indices + affines)
static ID3D12Resource*       g_sharcCB = nullptr;      // b3: SHARC grid/resolve parameters
static ID3D12Resource*       g_sharcHash = nullptr;
static ID3D12Resource*       g_sharcAccum = nullptr;
static ID3D12Resource*       g_sharcResolved = nullptr;
static ID3D12PipelineState*  g_sharcResolvePSO = nullptr;
static bool                  g_sharcClear = true;
static float                 g_sharcPrevEye[3] = {0,0,0};
static ID3D12Resource*       g_sbtHit = nullptr;       // per-frame hit records
static uint8_t               g_hitId[32];
static uint8_t               g_partHitId[32];          // particle any-hit group shader id
static const UINT            HIT_REC = 128;            // 32 id + 8 ib VA + 8 vb VA + 80 bytes of constants
static const uint32_t        MAX_INST = 65536;         // TLAS instance capacity (world grows large)
static std::unordered_map<uint32_t, ID3D12Resource*> g_tex;      // by bridge tex id
static std::vector<ID3D12Resource*>                  g_frameUploads;  // texture staging
static ID3D12Resource*                               g_screenSkyTex = nullptr; // heap slot TEX_SRV_BASE-1
static uint32_t                                      g_screenSkySeq = 0;
// Per-frame expanded particle geometry (world-space quads from the game's particle VS).
static ID3D12Resource*                               g_partSeqIB = nullptr;    // shared sequential index buffer 0..N-1
static uint32_t                                      g_partSeqIBCount = 0;
static uint32_t                                      g_partSeq = 0;            // last consumed particle sequence
static const UINT            SKYENV_UAV_SLOT = 11;   // persistent lat-long real-sky env map
static ID3D12Resource*       g_skyEnvTex = nullptr;
static ID3D12DescriptorHeap* g_clearHeap = nullptr; // non-shader-visible UAV (for clears)
static bool                  g_skyEnvClear = true;   // clear the env map (init + level change)
static ID3D12DescriptorHeap* g_srvHeap = nullptr;   // shader-visible descriptor heap (see slot map below)
static ID3D12Resource*       g_outTex = nullptr;    // [8] R8G8B8A8, final tonemapped -> backbuffer
// Path tracer G-buffer for DLSS Ray Reconstruction (all render-res, UAV state).
static ID3D12Resource*       g_colorTex = nullptr;  // [0] RGBA16F linear HDR noisy color
static ID3D12Resource*       g_depthTex = nullptr;  // [2] R32F   linear view depth
static ID3D12Resource*       g_mvTex    = nullptr;  // [3] RG16F  screen motion vectors (px)
static ID3D12Resource*       g_normTex  = nullptr;  // [4] RGBA16F world normals
static ID3D12Resource*       g_albTex   = nullptr;  // [5] RGBA8  diffuse albedo
static ID3D12Resource*       g_specTex  = nullptr;  // [6] RGBA8  specular albedo
static ID3D12Resource*       g_roughTex = nullptr;  // [7] R8     roughness
static ID3D12Resource*       g_ppTex    = nullptr;  // [9] RGBA16F RR output (linear HDR)
static ID3D12RootSignature*  g_tmRS  = nullptr;     // tonemap compute root sig
static ID3D12PipelineState*  g_tmPSO = nullptr;     // tonemap compute PSO
static ID3D12RootSignature*  g_fogRS = nullptr;     // quarter-res ellipsoid volumetrics
static ID3D12PipelineState*  g_fogPSO = nullptr;
static ID3D12Resource*       g_fogTex = nullptr;    // rgb in-scattering, a transmittance
static ID3D12Resource*       g_fogVolumeBuf = nullptr;
// Dense effects can submit hundreds of overlapping sprites. Bound the work per
// frame and prefer the nearest content; distant layers are visually redundant.
static const uint32_t        MAX_FOG_VOLUMES = 32;
static const uint32_t        MAX_PARTICLE_FIRE_VERTS = 24576;
static float  g_prevRelWC[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };  // last frame rot-only world->clip (eye-relative)
static float  g_prevEye[3] = {0,0,0};
static uint32_t g_jitterPhase = 0;
static bool   g_dlssReset = true;                   // force RR history reset next eval
static ID3D12Resource*       g_sbt = nullptr;       // shader table (raygen/miss/hit)
static ID3D12Resource*       g_blasScratch = nullptr;
static ID3D12Resource*       g_tlasScratch = nullptr;
static ID3D12Resource*       g_tlasResult = nullptr;
static ID3D12Resource*       g_instBuf = nullptr;   // instance descs (upload)
static ID3D12Resource*       g_lightBuf = nullptr;  // current frame's BridgeLight array (upload)
static ID3D12Resource*       g_restirCB = nullptr;  // b4: ReSTIR DI ping-pong slots + controls
static ID3D12Resource*       g_restirReservoir[2] = {};
static ID3D12Resource*       g_restirSurface[2] = {};
static bool                  g_restirClear = true;
static UINT64                g_tlasScratchSize = 0, g_tlasResultSize = 0, g_instBufSize = 0;
static UINT                  g_srvInc = 0;

struct GpuMesh { ID3D12Resource* res; UINT size; UINT stride; };
static std::unordered_map<uint32_t, GpuMesh> g_vb, g_ib;   // by bridge id
struct Blas { ID3D12Resource* result; };
static std::unordered_map<uint64_t, Blas> g_blas;          // by surface key
static int  g_blasBudget = 0;   // new BLAS allowed to build this frame (spread load)
static bool g_lost = false;     // device removed

// Matches FogVolume in fog.hlsl (six float4s). Smoke quads are converted to
// soft ellipsoids and never enter the TLAS.
struct FogVolume {
    float centerDensity[4];
    float axisU[4], axisV[4], axisW[4];
    float uvRect[4];
    float colorTex[4];
};

// Persistent world: surfaces accumulate here (world space) and never leave until a
// level change, so off-screen geometry still exists for shadows / GI / reflections.
struct PInst {
    ID3D12Resource*           blas;
    float                     xform[12];   // local -> world (3x4 row-major)
    uint32_t                  instanceID;
    D3D12_GPU_VIRTUAL_ADDRESS ibVA, vbVA;
    uint32_t stride, istride, startIndex, baseVertex, texId, uvOff, posOff;  // texId resolved at build time
    uint32_t flags;            // DRAW_FLAG_* (sky = emissive)
    uint32_t emissiveTexId;    // additive material pass merged onto this surface
    uint32_t specTexId, normalTexId, depthTexId; // material texture ids
    float    specParams[4];    // $vSpecularPowerIntensityBiasScale
    float    parallaxParams[4];// $vParallaxScaleBias
    uint32_t lastFrame;
    bool     confirmed;        // seen twice at the same transform -> static -> keep forever
};
static std::unordered_map<uint64_t, PInst>    g_world;   // confirmed-static, permanent
static std::unordered_map<uint64_t, uint32_t> g_seen;    // fullKey -> frame first seen (confirmation)
static float    g_viewToWorld[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };   // last-known camera basis
static bool     g_v2wValid  = false;
static uint32_t g_lastVbCount = 0;
static uint32_t g_frameNo   = 0;

// Perf instrumentation.
static LARGE_INTEGER g_qpcFreq = {0};
static double g_cpuMs = 0, g_frameMs = 0, g_tlasMs = 0; static int g_perfN = 0;
static double qpc_ms(LARGE_INTEGER a, LARGE_INTEGER b) { return (b.QuadPart - a.QuadPart) * 1000.0 / g_qpcFreq.QuadPart; }

// out(3x4 affine) = a * b, both with implicit last row (0,0,0,1).
static void affine_mul(const float* a, const float* b, float* o)
{
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            o[r*4+c] = a[r*4+0]*b[0*4+c] + a[r*4+1]*b[1*4+c] + a[r*4+2]*b[2*4+c];
        o[r*4+3] = a[r*4+0]*b[3] + a[r*4+1]*b[7] + a[r*4+2]*b[11] + a[r*4+3];
    }
}

static bool is_baked_world_transform(const float* m)
{
    // Baked map surfaces resolve to ~identity after viewToWorld * modelView (verts
    // already in world space). Props/doors/lifts carry a real model->world transform
    // (translation = the object's world position, i.e. large) and must never be made
    // permanent. Rotation stays tight; translation gets a generous tolerance because
    // float precision at large map coordinates jitters it by up to ~1 unit.
    static const float rot[9] = { 1,0,0, 0,1,0, 0,0,1 };
    const int ri[9] = { 0,1,2, 4,5,6, 8,9,10 };
    for (int i = 0; i < 9; ++i) if (fabsf(m[ri[i]] - rot[i]) > 0.05f) return false;
    if (fabsf(m[3]) > 2.0f || fabsf(m[7]) > 2.0f || fabsf(m[11]) > 2.0f) return false;
    return true;
}

// Row-major 4x4 multiply (row-vector convention: v' = v * M, so worldToClip = W2V * V2C).
static void mat4_mul(const float* a, const float* b, float* o)
{
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        float s = 0; for (int k = 0; k < 4; ++k) s += a[r*4+k] * b[k*4+c];
        o[r*4+c] = s;
    }
}
// worldToView (row-major, row-vector): top-left 3x3 = viewToWorld rotation block, last row = -eye*V.
static void build_world_to_view(const float* v2w, const float* eye, float* W)
{
    for (int i = 0; i < 3; ++i) { W[i*4+0]=v2w[i*4+0]; W[i*4+1]=v2w[i*4+1]; W[i*4+2]=v2w[i*4+2]; W[i*4+3]=0; }
    W[12] = -(eye[0]*v2w[0] + eye[1]*v2w[4] + eye[2]*v2w[8]);
    W[13] = -(eye[0]*v2w[1] + eye[1]*v2w[5] + eye[2]*v2w[9]);
    W[14] = -(eye[0]*v2w[2] + eye[1]*v2w[6] + eye[2]*v2w[10]);
    W[15] = 1;
}
// Perspective viewToClip (RH, looking -Z, depth [0,1]) from the game's projection scales.
static void build_view_to_clip(float sx, float sy, float* P)
{
    const float n = 1.0f, f = 100000.0f;
    memset(P, 0, 16*sizeof(float));
    P[0] = sx; P[5] = sy; P[10] = f/(n-f); P[11] = -1.0f; P[14] = n*f/(n-f);
}
static float halton(uint32_t i, uint32_t b)
{
    float r = 0, inv = 1.0f/b; i += 1;
    while (i) { r += (i % b) * inv; i /= b; inv /= b; }
    return r;
}

static ID3D12Resource* create_buffer(UINT64 size, D3D12_HEAP_TYPE heap,
                                     D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size ? size : 1; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = flags;
    ID3D12Resource* r = nullptr;
    g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r));
    return r;
}
static void upload_bytes(ID3D12Resource* res, const void* data, UINT size)
{
    if (!res) return;
    void* p = nullptr; D3D12_RANGE none = {0, 0};
    if (SUCCEEDED(res->Map(0, &none, &p))) { memcpy(p, data, size); res->Unmap(0, nullptr); }
}
static void uav_barrier() { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; g_cl->ResourceBarrier(1, &b); }

bool gfx_rt_init()
{
    // Load compiled DXIL library (built next to the exe).
    char path[MAX_PATH]; GetModuleFileNameA(nullptr, path, MAX_PATH);
    char* slash = strrchr(path, '\\'); if (slash) strcpy_s(slash + 1, 64, "shader.cso");
    FILE* f = nullptr; fopen_s(&f, path, "rb");
    if (!f) { printf("[rt] shader.cso not found (%s)\n", path); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> dxil(n); fread(dxil.data(), 1, n, f); fclose(f);

    // Global root signature: table(u0 out, t0 TLAS) + root constants (b0 camera/lighting).
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[0].NumDescriptors = 1; ranges[0].BaseShaderRegister = 0; ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[1].NumDescriptors = 1; ranges[1].BaseShaderRegister = 0; ranges[1].OffsetInDescriptorsFromTableStart = 1;
    // Root signature is exactly at the 64-DWORD limit: descriptor tables/root CBVs
    // carry sky, SHARC and ReSTIR state while the camera remains root constants.
    D3D12_DESCRIPTOR_RANGE sharcRange = {};
    sharcRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; sharcRange.NumDescriptors = 3;
    sharcRange.BaseShaderRegister = 1; sharcRange.OffsetInDescriptorsFromTableStart = 0;
    D3D12_ROOT_PARAMETER rp[6] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[0].DescriptorTable.NumDescriptorRanges = 2; rp[0].DescriptorTable.pDescriptorRanges = ranges;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[1].Constants.Num32BitValues = 56; rp[1].Constants.ShaderRegister = 0;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[2].Descriptor.ShaderRegister = 2;   // b2 sky layers
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[3].DescriptorTable.NumDescriptorRanges = 1; rp[3].DescriptorTable.pDescriptorRanges = &sharcRange;
    rp[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[4].Descriptor.ShaderRegister = 3;   // b3 SHARC
    rp[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[5].Descriptor.ShaderRegister = 4;   // b4 ReSTIR DI
    D3D12_STATIC_SAMPLER_DESC samp = {};
    samp.Filter = D3D12_FILTER_ANISOTROPIC; samp.MaxAnisotropy = 16;   // sky uses SampleGrad -> anisotropic AA of the pole spokes
    samp.AddressU = samp.AddressV = samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samp.MaxLOD = D3D12_FLOAT32_MAX; samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 6; rsd.pParameters = rp;
    rsd.NumStaticSamplers = 1; rsd.pStaticSamplers = &samp;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;  // ResourceDescriptorHeap[]
    ID3DBlob* sig = nullptr; ID3DBlob* err = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) {
        printf("[rt] serialize RS failed: %.*s\n", err ? (int)err->GetBufferSize() : 0, err ? (char*)err->GetBufferPointer() : ""); return false;
    }
    if (FAILED(g_dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&g_rtRS)))) { printf("[rt] CreateRootSignature failed\n"); return false; }
    sig->Release();

    // Local root signature (per hit-group instance): IB(t1), VB(t2), Geo consts(b1).
    D3D12_ROOT_PARAMETER lp[3] = {};
    lp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; lp[0].Descriptor.ShaderRegister = 1;
    lp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; lp[1].Descriptor.ShaderRegister = 2;
    lp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; lp[2].Constants.ShaderRegister = 1; lp[2].Constants.Num32BitValues = 20;
    D3D12_ROOT_SIGNATURE_DESC lrsd = {}; lrsd.NumParameters = 3; lrsd.pParameters = lp; lrsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
    ID3DBlob* lsig = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&lrsd, D3D_ROOT_SIGNATURE_VERSION_1, &lsig, &err))) { printf("[rt] serialize local RS failed\n"); return false; }
    if (FAILED(g_dev->CreateRootSignature(0, lsig->GetBufferPointer(), lsig->GetBufferSize(), IID_PPV_ARGS(&g_localRS)))) { printf("[rt] CreateRootSignature(local) failed\n"); return false; }
    lsig->Release();

    // RT pipeline state object.
    D3D12_STATE_SUBOBJECT so[8] = {};
    D3D12_DXIL_LIBRARY_DESC lib = {}; lib.DXILLibrary.pShaderBytecode = dxil.data(); lib.DXILLibrary.BytecodeLength = n;
    so[0].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY; so[0].pDesc = &lib;
    D3D12_HIT_GROUP_DESC hg = {}; hg.HitGroupExport = L"HitGroup"; hg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES; hg.ClosestHitShaderImport = L"CHit";
    so[1].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP; so[1].pDesc = &hg;
    D3D12_RAYTRACING_SHADER_CONFIG sc = {}; sc.MaxPayloadSizeInBytes = 104; sc.MaxAttributeSizeInBytes = 8;  // material + particle emission + volumetric transmittance
    so[2].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG; so[2].pDesc = &sc;
    D3D12_GLOBAL_ROOT_SIGNATURE grs = {}; grs.pGlobalRootSignature = g_rtRS;
    so[3].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE; so[3].pDesc = &grs;
    D3D12_RAYTRACING_PIPELINE_CONFIG pc = {}; pc.MaxTraceRecursionDepth = 2;   // primary + shadow
    so[4].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG; so[4].pDesc = &pc;
    D3D12_LOCAL_ROOT_SIGNATURE lrs = {}; lrs.pLocalRootSignature = g_localRS;
    so[5].Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE; so[5].pDesc = &lrs;
    // Transparent particle billboards use an any-hit (accumulate + IgnoreHit) with no closest hit.
    D3D12_HIT_GROUP_DESC phg = {}; phg.HitGroupExport = L"PartHitGroup"; phg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES; phg.AnyHitShaderImport = L"PartAnyHit";
    so[6].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP; so[6].pDesc = &phg;
    const wchar_t* hgNames[2] = { L"HitGroup", L"PartHitGroup" };
    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION assoc = {}; assoc.pSubobjectToAssociate = &so[5]; assoc.NumExports = 2; assoc.pExports = hgNames;
    so[7].Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION; so[7].pDesc = &assoc;
    D3D12_STATE_OBJECT_DESC sod = {}; sod.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE; sod.NumSubobjects = 8; sod.pSubobjects = so;
    if (FAILED(g_dev->CreateStateObject(&sod, IID_PPV_ARGS(&g_rtSO)))) { printf("[rt] CreateStateObject failed\n"); return false; }

    // DLSS Ray Reconstruction: init + pick the ray-traced render resolution (upscaling)
    // BEFORE sizing the G-buffer. Default Quality tier; WOLF_DLSS env overrides.
    g_rw = g_w; g_rh = g_h;
    if (dlss_init(g_dev)) {
        const char* q = getenv("WOLF_DLSS"); g_dlssQuality = 1;
        if (q) { if (!strcmp(q,"dlaa")) g_dlssQuality=0; else if (!strcmp(q,"balanced")) g_dlssQuality=2;
                 else if (!strcmp(q,"performance")) g_dlssQuality=3; else if (!strcmp(q,"ultraperf")) g_dlssQuality=4;
                 else g_dlssQuality=1; }
        int rw, rh;
        if (g_dlssQuality != 0 && dlss_query_render_res(g_w, g_h, g_dlssQuality, &rw, &rh)) { g_rw = rw; g_rh = rh; }
        g_jitterPhases = (int)(8.0f * ((float)g_h / g_rh) * ((float)g_h / g_rh) + 0.5f);
        if (g_jitterPhases < 8) g_jitterPhases = 8; if (g_jitterPhases > 64) g_jitterPhases = 64;
        printf("[rt] DLSS RR: ray-trace %dx%d -> upscale %dx%d (q=%d, jitterPhases=%d)\n", g_rw, g_rh, g_w, g_h, g_dlssQuality, g_jitterPhases);
    } else {
        printf("[rt] DLSS unavailable -- native res, noisy fallback\n");
    }

    // G-buffer + output textures. Ray-traced buffers are RENDER res (g_rw x g_rh); the RR
    // output + tonemap target are OUTPUT res (g_w x g_h). Heap slot map:
    //   [0] colorHDR (u0)  [1] TLAS SRV  [2] depth  [3] mv  [4] normal  [5] albedo
    //   [6] spec  [7] rough  [8] outTex (tonemap u0)  [9] ppOut (RR out / tonemap u1)
    //   [10] local lights [11] sky env [12..14] SHARC hash/accum/resolved
    //   [15..18] ReSTIR reservoir/surface ping-pong [19] live sky [20+] materials
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    auto make_tex = [&](DXGI_FORMAT fmt, int tw, int th, ID3D12Resource** out) {
        D3D12_RESOURCE_DESC t = {}; t.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        t.Width = tw; t.Height = th; t.DepthOrArraySize = 1; t.MipLevels = 1;
        t.Format = fmt; t.SampleDesc.Count = 1; t.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(out));
    };
    make_tex(DXGI_FORMAT_R16G16B16A16_FLOAT, g_rw, g_rh, &g_colorTex);
    make_tex(DXGI_FORMAT_R32_FLOAT,          g_rw, g_rh, &g_depthTex);
    make_tex(DXGI_FORMAT_R32G32_FLOAT,       g_rw, g_rh, &g_mvTex);   // full float: exact sub-pixel MVs
    make_tex(DXGI_FORMAT_R16G16B16A16_FLOAT, g_rw, g_rh, &g_normTex);
    make_tex(DXGI_FORMAT_R8G8B8A8_UNORM,     g_rw, g_rh, &g_albTex);
    make_tex(DXGI_FORMAT_R8G8B8A8_UNORM,     g_rw, g_rh, &g_specTex);
    make_tex(DXGI_FORMAT_R8_UNORM,           g_rw, g_rh, &g_roughTex);
    make_tex(DXGI_FORMAT_R8G8B8A8_UNORM,     g_w,  g_h,  &g_outTex);
    make_tex(DXGI_FORMAT_R16G16B16A16_FLOAT, g_w,  g_h,  &g_ppTex);
    make_tex(DXGI_FORMAT_R16G16B16A16_FLOAT, (g_w+3)/4, (g_h+3)/4, &g_fogTex);

    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.NumDescriptors = TEX_SRV_BASE + BRIDGE_MAX_TEX;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_srvHeap));
    g_srvInc = g_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto make_uav = [&](ID3D12Resource* res, DXGI_FORMAT fmt, UINT slot) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {}; u.Format = fmt; u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE h = g_srvHeap->GetCPUDescriptorHandleForHeapStart(); h.ptr += (SIZE_T)slot * g_srvInc;
        g_dev->CreateUnorderedAccessView(res, nullptr, &u, h);
    };
    make_uav(g_colorTex, DXGI_FORMAT_R16G16B16A16_FLOAT, 0);
    make_uav(g_depthTex, DXGI_FORMAT_R32_FLOAT,          2);
    make_uav(g_mvTex,    DXGI_FORMAT_R32G32_FLOAT,       3);
    make_uav(g_normTex,  DXGI_FORMAT_R16G16B16A16_FLOAT, 4);
    make_uav(g_albTex,   DXGI_FORMAT_R8G8B8A8_UNORM,     5);
    make_uav(g_specTex,  DXGI_FORMAT_R8G8B8A8_UNORM,     6);
    make_uav(g_roughTex, DXGI_FORMAT_R8_UNORM,           7);
    make_uav(g_outTex,   DXGI_FORMAT_R8G8B8A8_UNORM,     8);
    make_uav(g_ppTex,    DXGI_FORMAT_R16G16B16A16_FLOAT, 9);
    make_uav(g_fogTex,   DXGI_FORMAT_R16G16B16A16_FLOAT, FOG_UAV_SLOT);

    // Persistent lat-long "real sky" env map: primary sky rays paint the live
    // screen sky into it by direction; reflections/GI read it back (all directions).
    { D3D12_RESOURCE_DESC t = {}; t.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      t.Width = 2048; t.Height = 1024; t.DepthOrArraySize = 1; t.MipLevels = 1;
      t.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; t.SampleDesc.Count = 1; t.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_skyEnvTex)); }
    make_uav(g_skyEnvTex, DXGI_FORMAT_R16G16B16A16_FLOAT, SKYENV_UAV_SLOT);
    { D3D12_DESCRIPTOR_HEAP_DESC ch = {}; ch.NumDescriptors = 8; ch.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
      g_dev->CreateDescriptorHeap(&ch, IID_PPV_ARGS(&g_clearHeap));
      D3D12_UNORDERED_ACCESS_VIEW_DESC u = {}; u.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      g_dev->CreateUnorderedAccessView(g_skyEnvTex, nullptr, &u, g_clearHeap->GetCPUDescriptorHandleForHeapStart()); }

    // SHARC's default layouts are 8-byte hash keys and two 16-byte radiance
    // records.  Keep these as UAV-state buffers for the lifetime of the renderer.
    g_sharcHash = create_buffer((UINT64)SHARC_CAPACITY * 8u, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    g_sharcAccum = create_buffer((UINT64)SHARC_CAPACITY * 16u, D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    g_sharcResolved = create_buffer((UINT64)SHARC_CAPACITY * 16u, D3D12_HEAP_TYPE_DEFAULT,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!g_sharcHash || !g_sharcAccum || !g_sharcResolved) { printf("[rt] SHARC buffer creation failed\n"); return false; }
    auto make_buffer_uav = [&](ID3D12Resource* res, UINT slot, UINT elements, UINT stride, UINT clearSlot) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {}; u.Format = DXGI_FORMAT_UNKNOWN; u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = elements; u.Buffer.StructureByteStride = stride;
        D3D12_CPU_DESCRIPTOR_HANDLE visible = g_srvHeap->GetCPUDescriptorHandleForHeapStart(); visible.ptr += (SIZE_T)slot * g_srvInc;
        g_dev->CreateUnorderedAccessView(res, nullptr, &u, visible);
        D3D12_CPU_DESCRIPTOR_HANDLE clear = g_clearHeap->GetCPUDescriptorHandleForHeapStart(); clear.ptr += (SIZE_T)clearSlot * g_srvInc;
        g_dev->CreateUnorderedAccessView(res, nullptr, &u, clear);
    };
    make_buffer_uav(g_sharcHash, SHARC_HASH_UAV_SLOT, SHARC_CAPACITY, 8, 1);
    make_buffer_uav(g_sharcAccum, SHARC_ACCUM_UAV_SLOT, SHARC_CAPACITY, 16, 2);
    make_buffer_uav(g_sharcResolved, SHARC_RESOLVED_UAV_SLOT, SHARC_CAPACITY, 16, 3);

    // ReSTIR DI history. Each reservoir and compact world-space surface record is
    // 16 bytes per render pixel; ping-pong avoids same-dispatch read/write hazards.
    const UINT restirPixels = (UINT)g_rw * (UINT)g_rh;
    for (UINT ri = 0; ri < 2; ++ri) {
        g_restirReservoir[ri] = create_buffer((UINT64)restirPixels * 16u, D3D12_HEAP_TYPE_DEFAULT,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        g_restirSurface[ri] = create_buffer((UINT64)restirPixels * 16u, D3D12_HEAP_TYPE_DEFAULT,
                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
    if (!g_restirReservoir[0] || !g_restirReservoir[1] || !g_restirSurface[0] || !g_restirSurface[1]) {
        printf("[rt] ReSTIR DI buffer creation failed\n"); return false;
    }
    make_buffer_uav(g_restirReservoir[0], RESTIR_RES0_UAV_SLOT, restirPixels, 16, 4);
    make_buffer_uav(g_restirReservoir[1], RESTIR_RES1_UAV_SLOT, restirPixels, 16, 5);
    make_buffer_uav(g_restirSurface[0],   RESTIR_SURF0_UAV_SLOT, restirPixels, 16, 6);
    make_buffer_uav(g_restirSurface[1],   RESTIR_SURF1_UAV_SLOT, restirPixels, 16, 7);

    g_lightBuf = create_buffer((UINT64)BRIDGE_MAX_LIGHTS * sizeof(BridgeLight), D3D12_HEAP_TYPE_UPLOAD,
                               D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    if (!g_lightBuf) { printf("[rt] local-light buffer create failed\n"); return false; }
    D3D12_SHADER_RESOURCE_VIEW_DESC lightSrv = {};
    lightSrv.Format = DXGI_FORMAT_UNKNOWN;
    lightSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    lightSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    lightSrv.Buffer.NumElements = BRIDGE_MAX_LIGHTS;
    lightSrv.Buffer.StructureByteStride = sizeof(BridgeLight);
    D3D12_CPU_DESCRIPTOR_HANDLE lightHandle = g_srvHeap->GetCPUDescriptorHandleForHeapStart();
    lightHandle.ptr += (SIZE_T)LIGHT_SRV_SLOT * g_srvInc;
    g_dev->CreateShaderResourceView(g_lightBuf, &lightSrv, lightHandle);

    g_fogVolumeBuf = create_buffer((UINT64)MAX_FOG_VOLUMES * sizeof(FogVolume), D3D12_HEAP_TYPE_UPLOAD,
                                   D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    if (!g_fogVolumeBuf || !g_fogTex) { printf("[rt] fog resources create failed\n"); return false; }

    // Tonemap/Veil compute pipeline: table(u0 outTex, u1 ppOut) + b0 params.
    {
        D3D12_DESCRIPTOR_RANGE tr[2] = {};
        tr[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; tr[0].NumDescriptors = 1; tr[0].BaseShaderRegister = 0; tr[0].OffsetInDescriptorsFromTableStart = 0;
        tr[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; tr[1].NumDescriptors = 1; tr[1].BaseShaderRegister = 1; tr[1].OffsetInDescriptorsFromTableStart = 1;
        D3D12_ROOT_PARAMETER trp[2] = {};
        trp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        trp[0].DescriptorTable.NumDescriptorRanges = 2; trp[0].DescriptorTable.pDescriptorRanges = tr;
        trp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        trp[1].Constants.ShaderRegister = 0; trp[1].Constants.Num32BitValues = 24;
        D3D12_ROOT_SIGNATURE_DESC trsd = {}; trsd.NumParameters = 2; trsd.pParameters = trp;
        D3D12_STATIC_SAMPLER_DESC tsamp[2] = {};
        for(int si=0;si<2;++si){tsamp[si].Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;tsamp[si].AddressU=tsamp[si].AddressV=tsamp[si].AddressW=si?D3D12_TEXTURE_ADDRESS_MODE_CLAMP:D3D12_TEXTURE_ADDRESS_MODE_WRAP;tsamp[si].MaxLOD=D3D12_FLOAT32_MAX;tsamp[si].ShaderRegister=si;tsamp[si].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;}
        trsd.NumStaticSamplers=2; trsd.pStaticSamplers=tsamp;
        trsd.Flags=D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
        ID3DBlob* tsig = nullptr; ID3DBlob* terr = nullptr;
        if (FAILED(D3D12SerializeRootSignature(&trsd, D3D_ROOT_SIGNATURE_VERSION_1, &tsig, &terr))) { printf("[rt] tonemap RS serialize failed\n"); return false; }
        if (FAILED(g_dev->CreateRootSignature(0, tsig->GetBufferPointer(), tsig->GetBufferSize(), IID_PPV_ARGS(&g_tmRS)))) { printf("[rt] tonemap RS create failed\n"); return false; }
        tsig->Release();
        char tmp[MAX_PATH]; GetModuleFileNameA(nullptr, tmp, MAX_PATH);
        char* sl = strrchr(tmp, '\\'); if (sl) strcpy_s(sl + 1, 64, "tonemap.cso");
        FILE* tf = nullptr; fopen_s(&tf, tmp, "rb");
        if (!tf) { printf("[rt] tonemap.cso not found\n"); return false; }
        fseek(tf, 0, SEEK_END); long tn = ftell(tf); fseek(tf, 0, SEEK_SET);
        std::vector<char> tcs(tn); fread(tcs.data(), 1, tn, tf); fclose(tf);
        D3D12_COMPUTE_PIPELINE_STATE_DESC cpd = {}; cpd.pRootSignature = g_tmRS;
        cpd.CS.pShaderBytecode = tcs.data(); cpd.CS.BytecodeLength = tn;
        if (FAILED(g_dev->CreateComputePipelineState(&cpd, IID_PPV_ARGS(&g_tmPSO)))) { printf("[rt] tonemap PSO create failed\n"); return false; }
    }

    // Quarter-resolution analytic volume integration: u0 fog target, root t0
    // volume buffer, and b0 camera/lighting constants.
    {
        D3D12_DESCRIPTOR_RANGE fr = {};
        fr.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; fr.NumDescriptors = 1;
        fr.BaseShaderRegister = 0; fr.OffsetInDescriptorsFromTableStart = 0;
        D3D12_ROOT_PARAMETER fp[3] = {};
        fp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        fp[0].DescriptorTable.NumDescriptorRanges = 1; fp[0].DescriptorTable.pDescriptorRanges = &fr;
        fp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; fp[1].Descriptor.ShaderRegister = 0;
        fp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        fp[2].Constants.ShaderRegister = 0; fp[2].Constants.Num32BitValues = 28;
        D3D12_STATIC_SAMPLER_DESC fs = {};
        fs.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        fs.AddressU = fs.AddressV = fs.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        fs.MaxLOD = D3D12_FLOAT32_MAX; fs.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC fd = {};
        fd.NumParameters = 3; fd.pParameters = fp; fd.NumStaticSamplers = 1; fd.pStaticSamplers = &fs;
        fd.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
        ID3DBlob* sig = nullptr; ID3DBlob* ferr = nullptr;
        if (FAILED(D3D12SerializeRootSignature(&fd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &ferr))) {
            printf("[rt] fog RS serialize failed: %.*s\n", ferr?(int)ferr->GetBufferSize():0, ferr?(char*)ferr->GetBufferPointer():""); return false;
        }
        if (FAILED(g_dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&g_fogRS)))) { sig->Release(); return false; }
        sig->Release();
        char path[MAX_PATH]; GetModuleFileNameA(nullptr,path,MAX_PATH);
        char* sl=strrchr(path,'\\'); if(sl) strcpy_s(sl+1,64,"fog.cso");
        FILE* ff=nullptr; fopen_s(&ff,path,"rb"); if(!ff){printf("[rt] fog.cso not found\n");return false;}
        fseek(ff,0,SEEK_END); long fn=ftell(ff); fseek(ff,0,SEEK_SET);
        std::vector<char> code(fn); fread(code.data(),1,fn,ff); fclose(ff);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd={}; pd.pRootSignature=g_fogRS; pd.CS.pShaderBytecode=code.data(); pd.CS.BytecodeLength=fn;
        if(FAILED(g_dev->CreateComputePipelineState(&pd,IID_PPV_ARGS(&g_fogPSO)))){printf("[rt] fog PSO create failed\n");return false;}
    }

    // SHARC Resolve compute shader uses the same global root signature/bindings.
    {
        char tmp[MAX_PATH]; GetModuleFileNameA(nullptr, tmp, MAX_PATH);
        char* sl = strrchr(tmp, '\\'); if (sl) strcpy_s(sl + 1, 64, "sharc_resolve.cso");
        FILE* sf = nullptr; fopen_s(&sf, tmp, "rb");
        if (!sf) { printf("[rt] sharc_resolve.cso not found\n"); return false; }
        fseek(sf, 0, SEEK_END); long sn = ftell(sf); fseek(sf, 0, SEEK_SET);
        std::vector<char> scs(sn); fread(scs.data(), 1, sn, sf); fclose(sf);
        D3D12_COMPUTE_PIPELINE_STATE_DESC cpd = {}; cpd.pRootSignature = g_rtRS;
        cpd.CS.pShaderBytecode = scs.data(); cpd.CS.BytecodeLength = sn;
        if (FAILED(g_dev->CreateComputePipelineState(&cpd, IID_PPV_ARGS(&g_sharcResolvePSO)))) { printf("[rt] SHARC resolve PSO create failed\n"); return false; }
    }

    // Shader table: render raygen@0, SHARC update raygen@64, miss table@128.
    // Hit records are per-instance and live in a separate per-frame buffer.
    ID3D12StateObjectProperties* props = nullptr; g_rtSO->QueryInterface(IID_PPV_ARGS(&props));
    g_sbt = create_buffer(256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    uint8_t* map = nullptr; D3D12_RANGE none = {0, 0}; g_sbt->Map(0, &none, (void**)&map);
    memcpy(map + 0,  props->GetShaderIdentifier(L"RayGen"),     32);
    memcpy(map + 64, props->GetShaderIdentifier(L"SharcUpdateRayGen"), 32);
    memcpy(map + 128, props->GetShaderIdentifier(L"Miss"),       32);   // miss index 0
    memcpy(map + 160, props->GetShaderIdentifier(L"ShadowMiss"), 32);   // miss index 1 (stride 32)
    g_sbt->Unmap(0, nullptr);
    memcpy(g_hitId, props->GetShaderIdentifier(L"HitGroup"), 32);
    memcpy(g_partHitId, props->GetShaderIdentifier(L"PartHitGroup"), 32);
    props->Release();
    g_sbtHit = create_buffer((UINT64)MAX_INST * HIT_REC, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);

    g_blasScratch = create_buffer(64u << 20, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // TLAS buffers sized ONCE for the max instance count and never recreated
    // (recreating them mid-flight caused a use-after-free -> GPU page fault).
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tin = {};
    tin.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tin.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; tin.NumDescs = MAX_INST;
    tin.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tinfo = {};
    g_dev->GetRaytracingAccelerationStructurePrebuildInfo(&tin, &tinfo);
    g_tlasScratch = create_buffer(tinfo.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    g_tlasResult  = create_buffer(tinfo.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    g_instBuf = create_buffer((UINT64)MAX_INST * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    g_skyCB   = create_buffer(256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);  // 20 values, 256-byte CBV
    g_sharcCB = create_buffer(256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    g_restirCB = create_buffer(256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    if (!g_skyCB || !g_sharcCB || !g_restirCB) { printf("[rt] frame constant buffer creation failed\n"); return false; }

    printf("[rt] SHARC ready: %u entries, 40 MiB, 5x5 sparse update (set WOLF_NOSHARC=1 to disable)\n", SHARC_CAPACITY);
    printf("[rt] ReSTIR DI ready: %u pixels, temporal + 4-neighbor spatial reuse\n", restirPixels);
    printf("[rt] pipeline ready (%dx%d)\n", g_w, g_h);
    return true;
}

static DXGI_FORMAT btex_dxgi(uint32_t f)
{
    switch (f) {
        case BTEX_BC1: return DXGI_FORMAT_BC1_UNORM_SRGB;
        case BTEX_BC2: return DXGI_FORMAT_BC2_UNORM_SRGB;
        case BTEX_BC3: return DXGI_FORMAT_BC3_UNORM_SRGB;
        case BTEX_RGBA16F: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default:       return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    }
}

static UINT host_row_bytes(uint32_t fmt, uint32_t w)
{
    if (fmt == BTEX_BC1) return ((w + 3) / 4) * 8;
    if (fmt == BTEX_BC2 || fmt == BTEX_BC3) return ((w + 3) / 4) * 16;
    if (fmt == BTEX_RGBA16F) return w * 8;
    return w * 4;
}
static UINT host_tex_rows(uint32_t fmt, uint32_t h) { return (fmt == BTEX_BGRA8 || fmt == BTEX_RGBA16F) ? h : (h + 3) / 4; }

// Create a GPU texture (with its mip chain, if captured), copy the pixels in, SRV at heapSlot.
// Source layout from the client: face-major, mip-minor (face0[mip0,mip1,...], face1[...]).
static ID3D12Resource* upload_texture(const BridgeTex& t, const uint8_t* src, uint32_t heapSlot)
{
    DXGI_FORMAT fmt = btex_dxgi(t.fmt);
    UINT faces = t.faces == 6 ? 6u : 1u;
    UINT mips  = t.mipCount < 1 ? 1u : t.mipCount;
    UINT subs  = faces * mips;
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {}; td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = t.width; td.Height = t.height; td.DepthOrArraySize = (UINT16)faces; td.MipLevels = (UINT16)mips;
    td.Format = fmt; td.SampleDesc.Count = 1;
    ID3D12Resource* res = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&res)))) return nullptr;

    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(subs);
    std::vector<UINT> rows(subs);
    std::vector<UINT64> rowBytes(subs);
    UINT64 total = 0;
    g_dev->GetCopyableFootprints(&td, 0, subs, 0, fp.data(), rows.data(), rowBytes.data(), &total);
    ID3D12Resource* up = create_buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    if (!up) { res->Release(); return nullptr; }

    // Per-face byte size = sum of all mip sizes (all faces identical).
    UINT64 faceSize = 0;
    for (UINT m = 0; m < mips; ++m) {
        UINT mw = t.width >> m; if (!mw) mw = 1; UINT mh = t.height >> m; if (!mh) mh = 1;
        faceSize += (UINT64)host_row_bytes(t.fmt, mw) * host_tex_rows(t.fmt, mh);
    }
    uint8_t* map = nullptr; D3D12_RANGE none = {0, 0};
    up->Map(0, &none, (void**)&map);
    for (UINT face = 0; face < faces; ++face) {
        UINT64 mipOff = 0;
        for (UINT m = 0; m < mips; ++m) {
            UINT sub = face * mips + m;
            UINT mw = t.width >> m; if (!mw) mw = 1; UINT mh = t.height >> m; if (!mh) mh = 1;
            UINT srcPitch = host_row_bytes(t.fmt, mw), srcRows = host_tex_rows(t.fmt, mh);
            const uint8_t* mipSrc = src + (UINT64)face * faceSize + mipOff;
            UINT copyBytes = fp[sub].Footprint.RowPitch < srcPitch ? fp[sub].Footprint.RowPitch : srcPitch;
            for (UINT r = 0; r < rows[sub] && r < srcRows; ++r)
                memcpy(map + fp[sub].Offset + (UINT64)r * fp[sub].Footprint.RowPitch,
                       mipSrc + (UINT64)r * srcPitch, copyBytes);
            mipOff += (UINT64)srcPitch * srcRows;
        }
    }
    up->Unmap(0, nullptr);

    for (UINT sub = 0; sub < subs; ++sub) {
        D3D12_TEXTURE_COPY_LOCATION dst = {}; dst.pResource = res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = sub;
        D3D12_TEXTURE_COPY_LOCATION srcL = {}; srcL.pResource = up; srcL.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; srcL.PlacedFootprint = fp[sub];
        g_cl->CopyTextureRegion(&dst, 0, 0, 0, &srcL, nullptr);
    }
    D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = res;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cl->ResourceBarrier(1, &b);

    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {}; sd.Format = fmt;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (faces == 6) { sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE; sd.TextureCube.MipLevels = mips; }
    else            { sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;   sd.Texture2D.MipLevels = mips; }
    D3D12_CPU_DESCRIPTOR_HANDLE hnd = g_srvHeap->GetCPUDescriptorHandleForHeapStart(); hnd.ptr += (SIZE_T)heapSlot * g_srvInc;
    g_dev->CreateShaderResourceView(res, &sd, hnd);

    g_frameUploads.push_back(up);   // released once this frame's GPU work completes
    return res;
}

static void sync_screen_sky(const BridgeHeader* h)
{
    const BridgeScreenSky& s = h->screenSky;
    uint32_t seq0 = s.sequence;
    if (getenv("WOLF_SKYDBG")) { static int n=0; if ((n++ % 60)==0) printf("[skydbg] valid=%u seq=%u %ux%u pitch=%u size=%u\n", s.valid, s.sequence, s.width, s.height, s.rowPitch, s.size); }
    if (!s.valid || (seq0 & 1u) || seq0 == g_screenSkySeq || !s.width || !s.height ||
        s.rowPitch < s.width * 4u || s.size != (uint64_t)s.rowPitch * s.height ||
        s.size > BRIDGE_SCREEN_SKY_BYTES) return;
    uint32_t w=s.width, ht=s.height, pitch=s.rowPitch, size=s.size;
    std::vector<uint8_t> snapshot(size);
    MemoryBarrier();
    memcpy(snapshot.data(), bridge_screen_sky((void*)h), size);
    MemoryBarrier();
    if (h->screenSky.sequence != seq0 || (seq0 & 1u)) return;

    BridgeTex t = {}; t.size=size; t.width=w; t.height=ht; t.fmt=BTEX_BGRA8;
    t.rowPitch=pitch; t.rows=ht; t.faces=1;
    ID3D12Resource* fresh = upload_texture(t, snapshot.data(), TEX_SRV_BASE - 1u);
    if (fresh) {
        SAFE_RELEASE(g_screenSkyTex);
        g_screenSkyTex = fresh; g_screenSkySeq = seq0;
    }
}

static int g_texBudget = 0;

static void sync_geometry(const BridgeHeader* h)
{
    uint32_t vbCount = h->vbCount, ibCount = h->ibCount;
    if (vbCount > BRIDGE_MAX_MESHES) vbCount = BRIDGE_MAX_MESHES;
    if (ibCount > BRIDGE_MAX_MESHES) ibCount = BRIDGE_MAX_MESHES;
    for (uint32_t id = (uint32_t)g_vb.size(); id < vbCount; ++id) {
        const BridgeVB& v = bridge_vbs((void*)h)[id];
        if ((uint64_t)v.arenaOff + v.size > BRIDGE_ARENA_SIZE) return;   // not ready / bad
        ID3D12Resource* r = create_buffer(v.size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
        if (!r) return;                                                 // retry next frame
        upload_bytes(r, bridge_arena((void*)h) + v.arenaOff, v.size);
        g_vb[id] = { r, v.size, v.stride };
    }
    for (uint32_t id = (uint32_t)g_ib.size(); id < ibCount; ++id) {
        const BridgeIB& b = bridge_ibs((void*)h)[id];
        if ((uint64_t)b.arenaOff + b.size > BRIDGE_ARENA_SIZE) return;
        ID3D12Resource* r = create_buffer(b.size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
        if (!r) return;
        upload_bytes(r, bridge_arena((void*)h) + b.arenaOff, b.size);
        g_ib[id] = { r, b.size, (UINT)b.indexStride };
    }
    // Upload new textures (budgeted per frame to avoid a big copy stall).
    uint32_t texCount = h->texCount; if (texCount > BRIDGE_MAX_TEX) texCount = BRIDGE_MAX_TEX;
    for (uint32_t id = (uint32_t)g_tex.size(); id < texCount && g_texBudget > 0; ++id) {
        const BridgeTex& t = bridge_texs((void*)h)[id];
        ID3D12Resource* r = nullptr;
        if ((uint64_t)t.arenaOff + t.size <= BRIDGE_ARENA_SIZE && t.width && t.height && t.size)
            r = upload_texture(t, bridge_arena((void*)h) + t.arenaOff, TEX_SRV_BASE + id);
        g_tex[id] = r;
        --g_texBudget;
    }
}

static bool draw_indices_valid(const BridgeHeader* h, const BridgeDraw& d,
                               const GpuMesh& vb, const GpuMesh& ib)
{
    if (!vb.stride || (ib.stride != 2 && ib.stride != 4) ||
        d.posOffset + 12u > vb.stride || d.primCount == 0) return false;
    int64_t vertexBase = (int64_t)d.vbOffset + (int64_t)(int32_t)d.baseVertex * vb.stride;
    if (vertexBase < 0 || (uint64_t)vertexBase + d.posOffset + 12u > vb.size) return false;
    uint64_t indexEnd = (uint64_t)d.startIndex + (uint64_t)d.primCount * 3u;
    if (indexEnd > ib.size / ib.stride) return false;

    const BridgeIB& src = bridge_ibs((void*)h)[d.ibId];
    if ((uint64_t)src.arenaOff + src.size > BRIDGE_ARENA_SIZE || src.size < ib.size) return false;
    const uint8_t* indices = bridge_arena((void*)h) + src.arenaOff;
    uint64_t declaredEnd = (uint64_t)d.minIndex + d.numVertices;
    for (uint64_t i = d.startIndex; i < indexEnd; ++i) {
        uint32_t idx;
        if (ib.stride == 2) { uint16_t q; memcpy(&q, indices + i * 2u, 2); idx = q; }
        else memcpy(&idx, indices + i * 4u, 4);
        // These bounds come from DrawIndexedPrimitive itself. A violation means
        // the bridge snapshot is stale; never feed it to a DXR AS build.
        if (idx < d.minIndex || (uint64_t)idx >= declaredEnd) return false;
        uint64_t at = (uint64_t)vertexBase + (uint64_t)idx * vb.stride + d.posOffset;
        if (at + 12u > vb.size) return false;
    }
    return true;
}

// Estimate the world-space center of emissive fixture geometry directly from
// the bridge's CPU snapshot. Sampling an AABB is sufficient for associating a
// deferred light volume with nearby lamp glass/flame/sign geometry.
static bool emissive_draw_center(const BridgeHeader* h, const BridgeDraw& d, const float* world, float* out)
{
    if (d.vbId >= h->vbCount || d.ibId >= h->ibCount || !d.primCount) return false;
    const BridgeVB& vb = bridge_vbs((void*)h)[d.vbId];
    const BridgeIB& ib = bridge_ibs((void*)h)[d.ibId];
    if (!vb.stride || (ib.indexStride != 2 && ib.indexStride != 4) || d.posOffset + 12u > vb.stride) return false;
    if ((uint64_t)vb.arenaOff + vb.size > BRIDGE_ARENA_SIZE || (uint64_t)ib.arenaOff + ib.size > BRIDGE_ARENA_SIZE) return false;
    int64_t vertexBase = (int64_t)d.vbOffset + (int64_t)(int32_t)d.baseVertex * vb.stride;
    if (vertexBase < 0) return false;
    uint64_t first = d.startIndex, count = (uint64_t)d.primCount * 3u;
    if (first + count > ib.size / ib.indexStride) return false;
    const uint8_t* arena = bridge_arena((void*)h);
    const uint8_t* indices = arena + ib.arenaOff;
    const uint8_t* vertices = arena + vb.arenaOff;
    float lo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    uint64_t step = count > 192u ? count / 192u : 1u;
    uint32_t samples = 0;
    for (uint64_t e = 0; e < count; e += step) {
        uint64_t atIndex = first + e; uint32_t idx = 0;
        if (ib.indexStride == 2) { uint16_t q; memcpy(&q, indices + atIndex*2u, 2); idx = q; }
        else memcpy(&idx, indices + atIndex*4u, 4);
        uint64_t at = (uint64_t)vertexBase + (uint64_t)idx * vb.stride + d.posOffset;
        if (at + 12u > vb.size) continue;
        float p[3]; memcpy(p, vertices + at, 12);
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
        float wp[3] = {
            world[0]*p[0] + world[1]*p[1] + world[2]*p[2] + world[3],
            world[4]*p[0] + world[5]*p[1] + world[6]*p[2] + world[7],
            world[8]*p[0] + world[9]*p[1] + world[10]*p[2] + world[11]
        };
        for (int a=0;a<3;++a) { lo[a]=fminf(lo[a],wp[a]); hi[a]=fmaxf(hi[a],wp[a]); }
        ++samples;
    }
    if (!samples) return false;
    // Whole walls/large additive overlays are not physical light fixtures and
    // must not authorize every deferred volume in the surrounding room.
    float ex = hi[0]-lo[0], ey = hi[1]-lo[1], ez = hi[2]-lo[2];
    if (fmaxf(ex, fmaxf(ey, ez)) > 160.0f) return false;
    for (int a=0;a<3;++a) out[a] = 0.5f * (lo[a] + hi[a]);
    return true;
}

static uint64_t surface_key(const BridgeDraw& d)
{
    uint64_t h = 1469598103934665603ULL;
    const uint32_t fields[] = { d.vbId, d.ibId, d.startIndex, d.primCount,
        d.baseVertex, d.minIndex, d.numVertices, d.vbOffset, d.posOffset };
    for (uint32_t v : fields) { h ^= v; h *= 1099511628211ULL; }
    return h;
}

static ID3D12Resource* build_blas(const BridgeHeader* h, const BridgeDraw& d)
{
    auto vb = g_vb.find(d.vbId); auto ib = g_ib.find(d.ibId);
    if (vb == g_vb.end() || ib == g_ib.end()) return nullptr;
    if (!draw_indices_valid(h, d, vb->second, ib->second)) return nullptr;
    uint64_t key = surface_key(d);
    auto it = g_blas.find(key);
    if (it != g_blas.end()) return it->second.result;
    if (g_blasBudget <= 0) return nullptr;      // build later; avoids one-frame TDR

    // Guard against malformed ranges (bad baseVertex/index range would otherwise
    // underflow VertexCount or make the AS build read out of bounds -> crash).
    int64_t signedVertexBase = (int64_t)d.vbOffset + (int64_t)(int32_t)d.baseVertex * vb->second.stride;
    if (signedVertexBase < 0) return nullptr;
    UINT64 vertexBase = (UINT64)signedVertexBase;
    if (!vb->second.stride || d.posOffset + 12u > vb->second.stride ||
        vertexBase + d.posOffset + 12u > vb->second.size) return nullptr;
    UINT vtxAvail = 1u + (UINT)((vb->second.size - vertexBase - d.posOffset - 12u) / vb->second.stride);
    UINT idxAvail = ib->second.stride ? ib->second.size / ib->second.stride : 0;
    if (d.primCount == 0) return nullptr;
    if (d.numVertices && ((UINT64)d.minIndex + d.numVertices > vtxAvail)) return nullptr;
    if ((uint64_t)d.startIndex + (uint64_t)d.primCount * 3 > idxAvail) return nullptr;
    if (d.primCount > 200000u) return nullptr;   // skip pathologically large meshes

    D3D12_RAYTRACING_GEOMETRY_DESC gd = {};
    gd.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES; gd.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    gd.Triangles.VertexBuffer.StartAddress = vb->second.res->GetGPUVirtualAddress() + vertexBase + d.posOffset;
    gd.Triangles.VertexBuffer.StrideInBytes = vb->second.stride;
    gd.Triangles.VertexCount = vtxAvail;
    gd.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    gd.Triangles.IndexBuffer = ib->second.res->GetGPUVirtualAddress() + (UINT64)d.startIndex * ib->second.stride;
    gd.Triangles.IndexCount = d.primCount * 3;
    gd.Triangles.IndexFormat = ib->second.stride == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
    in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; in.NumDescs = 1; in.pGeometryDescs = &gd;
    in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    g_dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
    if (info.ScratchDataSizeInBytes > (64u << 20)) return nullptr;   // too big for shared scratch
    if (info.ResultDataMaxSizeInBytes == 0 || info.ResultDataMaxSizeInBytes > (256u << 20)) return nullptr;

    ID3D12Resource* result = create_buffer(info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                           D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!result) return nullptr;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
    bd.Inputs = in; bd.ScratchAccelerationStructureData = g_blasScratch->GetGPUVirtualAddress();
    bd.DestAccelerationStructureData = result->GetGPUVirtualAddress();
    g_cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
    uav_barrier();                       // serialize shared scratch reuse
    --g_blasBudget;
    g_blas[key] = { result };
    return result;
}

// Particle texture/class identity is stored on every compacted vertex. This lets
// the whole frame use one geometry and one shader record, avoiding both the old
// one-BLAS-per-draw watchdog failures and fragile GeometryIndex/SBT mapping.
struct GpuPartVert {
    float pos[3];
    float uv[2];
    uint32_t color;
    uint32_t texHeap;
    uint32_t cls;
};
static_assert(sizeof(GpuPartVert) == 32, "GpuPartVert shader layout mismatch");

static ID3D12Resource* build_particle_blas(D3D12_GPU_VIRTUAL_ADDRESS vbVA, D3D12_GPU_VIRTUAL_ADDRESS ibVA,
                                           uint32_t totalVerts)
{
    if (totalVerts < 3) return nullptr;
    D3D12_RAYTRACING_GEOMETRY_DESC gd = {};
    gd.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    gd.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    gd.Triangles.VertexBuffer.StartAddress = vbVA;
    gd.Triangles.VertexBuffer.StrideInBytes = sizeof(GpuPartVert);
    gd.Triangles.VertexCount = totalVerts;
    gd.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    gd.Triangles.IndexBuffer = ibVA;
    gd.Triangles.IndexCount = totalVerts;
    gd.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
    in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    in.NumDescs = 1;
    in.pGeometryDescs = &gd;
    in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    g_dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
    if (info.ScratchDataSizeInBytes > (64u<<20) || info.ResultDataMaxSizeInBytes == 0 ||
        info.ResultDataMaxSizeInBytes > (64u<<20)) return nullptr;
    ID3D12Resource* result = create_buffer(info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!result) return nullptr;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
    bd.Inputs = in; bd.ScratchAccelerationStructureData = g_blasScratch->GetGPUVirtualAddress();
    bd.DestAccelerationStructureData = result->GetGPUVirtualAddress();
    g_cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
    uav_barrier();
    return result;
}

static bool make_fog_volume(const BridgePartVert* v, uint32_t texHeap, FogVolume& out)
{
    float e1[3], e2[3];
    for (int k=0;k<3;++k) { e1[k]=v[1].pos[k]-v[0].pos[k]; e2[k]=v[2].pos[k]-v[0].pos[k]; }
    float du1=v[1].uv[0]-v[0].uv[0], dv1=v[1].uv[1]-v[0].uv[1];
    float du2=v[2].uv[0]-v[0].uv[0], dv2=v[2].uv[1]-v[0].uv[1];
    float det=du1*dv2-dv1*du2;
    if (fabsf(det)<1e-8f) return false;
    float inv=1.0f/det, dPdu[3], dPdv[3];
    for(int k=0;k<3;++k){ dPdu[k]=(e1[k]*dv2-e2[k]*dv1)*inv; dPdv[k]=(e2[k]*du1-e1[k]*du2)*inv; }

    float u0=FLT_MAX,v0=FLT_MAX,u1=-FLT_MAX,v1=-FLT_MAX;
    float tint[4]={0,0,0,0};
    for(int i=0;i<6;++i){
        u0=fminf(u0,v[i].uv[0]); v0=fminf(v0,v[i].uv[1]);
        u1=fmaxf(u1,v[i].uv[0]); v1=fmaxf(v1,v[i].uv[1]);
        uint32_t c=v[i].color;
        tint[0]+=(c&255u)/255.0f; tint[1]+=((c>>8)&255u)/255.0f;
        tint[2]+=((c>>16)&255u)/255.0f; tint[3]+=(c>>24)/255.0f;
    }
    for(float& x:tint) x/=6.0f;
    float uc=(u0+u1)*0.5f, vc=(v0+v1)*0.5f;
    for(int k=0;k<3;++k) out.centerDensity[k]=v[0].pos[k]+dPdu[k]*(uc-v[0].uv[0])+dPdv[k]*(vc-v[0].uv[1]);

    auto len3=[](const float* a){return sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);};
    float lu=len3(dPdu); if(!(lu>1e-5f) || !std::isfinite(lu)) return false;
    float au[3]={dPdu[0]/lu,dPdu[1]/lu,dPdu[2]/lu};
    float dotuv=dPdv[0]*au[0]+dPdv[1]*au[1]+dPdv[2]*au[2];
    float vv[3]={dPdv[0]-au[0]*dotuv,dPdv[1]-au[1]*dotuv,dPdv[2]-au[2]*dotuv};
    float lv=len3(vv); if(!(lv>1e-5f) || !std::isfinite(lv)) return false;
    float av[3]={vv[0]/lv,vv[1]/lv,vv[2]/lv};
    float aw[3]={au[1]*av[2]-au[2]*av[1],au[2]*av[0]-au[0]*av[2],au[0]*av[1]-au[1]*av[0]};
    float halfU=fabsf(u1-u0)*lu*0.5f, halfV=fabsf(v1-v0)*lv*0.5f;
    if(halfU<0.25f || halfV<0.25f || halfU>2000.0f || halfV>2000.0f) return false;
    float thick=fmaxf(3.0f,fminf(96.0f,fminf(halfU,halfV)*0.32f));
    for(int k=0;k<3;++k){out.axisU[k]=au[k];out.axisV[k]=av[k];out.axisW[k]=aw[k];}
    out.axisU[3]=halfU; out.axisV[3]=halfV; out.axisW[3]=thick;
    out.centerDensity[3]=fmaxf(0.05f,tint[3]);
    out.uvRect[0]=u0; out.uvRect[1]=v0; out.uvRect[2]=u1; out.uvRect[3]=v1;
    out.colorTex[0]=tint[0]; out.colorTex[1]=tint[1]; out.colorTex[2]=tint[2];
    memcpy(&out.colorTex[3],&texHeap,4);
    for(int k=0;k<3;++k) if(!std::isfinite(out.centerDensity[k])) return false;
    return true;
}

void gfx_render_scene(const BridgeHeader* h)
{
    if (g_lost) { Sleep(16); return; }                 // device dead: stop touching D3D12
    static bool s_green = (getenv("WOLF_GREEN") != nullptr);
    if (s_green) { gfx_clear_present(0.0f, 0.5f, 0.0f); return; }   // present-path sanity test
    if (!h->camera.valid) { gfx_clear_present(0.05f, 0.06f, 0.09f); return; }
    if (!g_qpcFreq.QuadPart) QueryPerformanceCounter(&g_qpcFreq), QueryPerformanceFrequency(&g_qpcFreq);
    LARGE_INTEGER tFrame0; QueryPerformanceCounter(&tFrame0);

    UINT i = g_frameIndex;
    g_alloc[i]->Reset();
    g_cl->Reset(g_alloc[i], nullptr);

    g_blasBudget = 96;                                 // cap new BLAS builds per frame
    g_texBudget = 32;                                  // cap new texture uploads per frame
    sync_geometry(h);
    sync_screen_sky(h);

    // Level change: the client resets its counts on a new session/level -> flush.
    if (h->vbCount < g_lastVbCount) {
        g_world.clear();
        g_seen.clear();
        for (auto& b : g_blas) if (b.second.result) b.second.result->Release();
        g_blas.clear();
        for (auto& v : g_vb) if (v.second.res) v.second.res->Release();
        for (auto& b : g_ib) if (b.second.res) b.second.res->Release();
        g_vb.clear(); g_ib.clear();
        for (auto& t : g_tex) if (t.second) t.second->Release();
        g_tex.clear();
        g_lightCache.clear();
        g_emitterAnchors.clear();
        g_skyEnvClear = true;                          // new level: reset accumulated sky
        g_sharcClear = true;                           // SHARC is world-space and cannot cross levels
        g_restirClear = true;                          // direct-light reservoirs reference the old scene
        g_v2wValid = false;
        g_dlssReset = true;                            // new level: drop RR history
    }
    g_lastVbCount = h->vbCount;

    // Build the RR feature once (records weight upload into this frame's list).
    bool dlssJustCreated = false;
    if (dlss_available() && !dlss_ready()) { dlss_create(g_cl, g_rw, g_rh, g_w, g_h, g_dlssQuality); dlssJustCreated = true; }

    uint32_t rl = h->readyList & 1;
    const BridgeDraw* draws = bridge_draws((void*)h, rl);
    uint32_t nd = h->drawCount[rl];
    if (nd > BRIDGE_MAX_DRAWS) nd = BRIDGE_MAX_DRAWS;
    uint32_t vbCount = h->vbCount, ibCount = h->ibCount;
    const float* eye = h->camera.viewOrigin;

    // Pass 1: recover the camera basis from a static-world draw (its modelView IS
    // the view matrix, so translation == -R*eye). viewToWorld = [R^T | eye].
    bool v2wFound = false;
    for (uint32_t k = 0; k < nd; ++k) {
        const BridgeDraw& d = draws[k];
        if (d.vbId >= vbCount || d.ibId >= ibCount || d.primCount == 0) continue;
        const float* m = d.transform;
        bool fin = true; for (int t = 0; t < 12; ++t) { float f = m[t]; if (f != f || f > 1e30f || f < -1e30f) { fin = false; break; } }
        if (!fin) continue;
        float e0 = -(m[0]*eye[0]+m[1]*eye[1]+m[2]*eye[2]);
        float e1 = -(m[4]*eye[0]+m[5]*eye[1]+m[6]*eye[2]);
        float e2 = -(m[8]*eye[0]+m[9]*eye[1]+m[10]*eye[2]);
        if (fabsf(e0-m[3])+fabsf(e1-m[7])+fabsf(e2-m[11]) < 3.0f) {
            for (int r = 0; r < 3; ++r) {
                g_viewToWorld[r*4+0] = m[0*4+r]; g_viewToWorld[r*4+1] = m[1*4+r];
                g_viewToWorld[r*4+2] = m[2*4+r]; g_viewToWorld[r*4+3] = eye[r];
            }
            g_v2wValid = true; v2wFound = true;
            break;
        }
    }

    // Pass 2: render this frame's draws LIVE (so the current view is never fragmented
    // and dynamics show at their current spot with no trail), and promote a placement
    // into the permanent world only once it's been seen static across frames.
    ++g_frameNo;
    struct LiveDraw { ID3D12Resource* blas; float xform[12]; D3D12_GPU_VIRTUAL_ADDRESS ibVA, vbVA;
                      uint32_t instanceID, stride, istride, startIndex, baseVertex, texId, uvOff, posOff, flags;
                      uint32_t emissiveTexId, specTexId, normalTexId, depthTexId;
                      float specParams[4], parallaxParams[4]; };
    std::vector<LiveDraw> live;
    std::unordered_set<uint64_t> liveKeys;
    std::unordered_map<uint64_t, size_t> liveIndex;
    for (auto it = g_seen.begin(); it != g_seen.end(); ) {   // prune stale sighting records
        if (g_frameNo - it->second > 2) it = g_seen.erase(it); else ++it;
    }
    if (g_v2wValid) {
        live.reserve(nd);
        for (uint32_t k = 0; k < nd; ++k) {
            const BridgeDraw& d = draws[k];
            if (d.vbId >= vbCount || d.ibId >= ibCount) continue;
            if (d.primCount == 0 || d.primCount > 4000000u) continue;
            bool fin = true; for (int t = 0; t < 12; ++t) { float f = d.transform[t]; if (f != f || f > 1e30f || f < -1e30f) { fin = false; break; } }
            if (!fin) continue;
            float world[12]; affine_mul(g_viewToWorld, d.transform, world);
            if ((d.flags & DRAW_FLAG_EMISSIVE) && !(d.flags & DRAW_FLAG_SKY)) {
                float center[3];
                if (emissive_draw_center(h, d, world, center)) {
                    EmitterAnchor* match = nullptr;
                    for (EmitterAnchor& a : g_emitterAnchors) {
                        float dx=a.p[0]-center[0], dy=a.p[1]-center[1], dz=a.p[2]-center[2];
                        if (dx*dx+dy*dy+dz*dz < 48.0f*48.0f) { match=&a; break; }
                    }
                    if (match) { memcpy(match->p, center, sizeof(center)); match->lastSeen = g_frameNo; }
                    else if (g_emitterAnchors.size() < 4096) {
                        EmitterAnchor a = {{center[0],center[1],center[2]},g_frameNo}; g_emitterAnchors.push_back(a);
                    }
                }
            }
            // Baked map surfaces resolve to ~identity (verts already in world space).
            // Key them by STABLE surface identity only -- NO world transform -- so float
            // precision at large map coords can't change the key frame-to-frame (which
            // was making the live copy and the frozen g_world copy fail to match ->
            // two overlapping meshes). Props (non-identity) keep a per-instance key so
            // multiple instances of the same mesh in one frame aren't deduped away.
            bool baked = is_baked_world_transform(world);
            uint64_t key = surface_key(d);
            if (!baked) {
                uint64_t xh = 1469598103934665603ULL;
                for (int t = 0; t < 12; ++t) { int32_t q = (int32_t)lroundf(world[t] * 4.0f); xh = (xh ^ (uint32_t)q) * 1099511628211ULL; }
                key = (key * 1099511628211ULL) ^ xh;
            }
            // Live draws are AUTHORITATIVE for on-screen geometry (rendered at their true
            // current position); g_world only fills in meshes NOT drawn this frame (off-screen).
            auto duplicate = liveIndex.find(key);
            if (duplicate != liveIndex.end()) {
                // Wolf commonly submits the same triangles twice: first the opaque/base
                // material, then an additive ColorMap. Keep one BLAS instance and merge
                // the second pass into a distinct emission channel (no z-fighting).
                LiveDraw& old = live[duplicate->second];
                uint32_t passTex = (d.texId != 0xFFFFFFFFu && d.uvOffset != 0xFFFFFFFFu) ? d.texId : 0xFFFFFFFFu;
                if (d.flags & DRAW_FLAG_EMISSIVE) {
                    old.emissiveTexId = passTex;
                    old.flags |= d.flags & (DRAW_FLAG_EMISSIVE | DRAW_FLAG_EMISSIVE_ALPHA);
                    auto wi = g_world.find(key);
                    if (wi != g_world.end()) { wi->second.emissiveTexId = passTex; wi->second.flags |= old.flags; }
                } else if (old.texId == 0xFFFFFFFFu) {
                    old.texId = passTex; old.specTexId = d.specTexId; old.normalTexId = d.normalTexId;
                    old.depthTexId = d.depthTexId;
                    memcpy(old.specParams, d.specParams, sizeof(old.specParams));
                    memcpy(old.parallaxParams, d.parallaxParams, sizeof(old.parallaxParams));
                }
                continue;
            }
            ID3D12Resource* blas = build_blas(h, d);
            if (!blas) continue;
            auto vbit = g_vb.find(d.vbId), ibit = g_ib.find(d.ibId);
            if (vbit == g_vb.end() || ibit == g_ib.end()) continue;
            uint32_t texId = (d.texId != 0xFFFFFFFFu && d.uvOffset != 0xFFFFFFFFu) ? d.texId : 0xFFFFFFFFu;
            uint32_t uvOff = (d.uvOffset != 0xFFFFFFFFu) ? d.uvOffset : 0;
            uint32_t instID = (d.vbId*2654435761u + d.startIndex*40503u) & 0xFFFFFF;
            LiveDraw ld; ld.blas = blas; memcpy(ld.xform, world, sizeof(world)); ld.instanceID = instID;
            int64_t signedVertexBase = (int64_t)d.vbOffset + (int64_t)(int32_t)d.baseVertex * vbit->second.stride;
            if (signedVertexBase < 0) continue;
            UINT64 vertexBase = (UINT64)signedVertexBase;
            if (vertexBase >= vbit->second.size) continue;
            ld.ibVA = ibit->second.res->GetGPUVirtualAddress();
            ld.vbVA = vbit->second.res->GetGPUVirtualAddress() + vertexBase;
            ld.stride = vbit->second.stride; ld.istride = ibit->second.stride; ld.startIndex = d.startIndex; ld.baseVertex = 0;
            ld.texId = (d.flags & DRAW_FLAG_EMISSIVE) ? 0xFFFFFFFFu : texId;
            ld.emissiveTexId = (d.flags & DRAW_FLAG_EMISSIVE) ? texId : 0xFFFFFFFFu;
            ld.uvOff = uvOff; ld.posOff = d.posOffset; ld.flags = d.flags;
            ld.specTexId = d.specTexId; ld.normalTexId = d.normalTexId; ld.depthTexId = d.depthTexId;
            memcpy(ld.specParams, d.specParams, sizeof(ld.specParams));
            memcpy(ld.parallaxParams, d.parallaxParams, sizeof(ld.parallaxParams));
            liveIndex[key] = live.size();
            liveKeys.insert(key);
            live.push_back(ld);
            // Promote to the permanent world once this exact placement was also seen on
            // an earlier frame (static). Moving objects never repeat a placement -> never promoted.
            if (baked) {
                auto s = g_seen.find(key);
                if (s != g_seen.end() && s->second != g_frameNo) {   // static: seen on an earlier frame -> permanent
                    PInst pi; pi.blas = blas; memcpy(pi.xform, world, sizeof(world)); pi.instanceID = instID;
                    pi.ibVA = ld.ibVA; pi.vbVA = ld.vbVA; pi.stride = ld.stride; pi.istride = ld.istride;
                    pi.startIndex = ld.startIndex; pi.baseVertex = ld.baseVertex; pi.texId = ld.texId; pi.uvOff = uvOff; pi.posOff = ld.posOff;
                    pi.flags = ld.flags; pi.emissiveTexId = ld.emissiveTexId;
                    pi.specTexId = ld.specTexId; pi.normalTexId = ld.normalTexId; pi.depthTexId = ld.depthTexId;
                    memcpy(pi.specParams, ld.specParams, sizeof(pi.specParams));
                    memcpy(pi.parallaxParams, ld.parallaxParams, sizeof(pi.parallaxParams));
                    pi.lastFrame = g_frameNo; pi.confirmed = true;
                    g_world[key] = pi;
                } else {
                    g_seen[key] = g_frameNo;
                }
            }
        }
    }

    // Build instance + hit arrays: permanent world (on+off screen) + this frame's live draws.
    LARGE_INTEGER tCpu0; QueryPerformanceCounter(&tCpu0);
    struct HitRec { D3D12_GPU_VIRTUAL_ADDRESS ibVA, vbVA; uint32_t stride, istride, startIndex, baseVertex, texHeap, uvOff, posOff;
                    uint32_t specHeap, normalHeap, emissiveHeap, depthHeap;
                    float specParams[4], parallaxParams[4]; uint32_t isParticle; };
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> inst;
    std::vector<HitRec> hits;
    std::vector<FogVolume> fogVolumes;
    fogVolumes.reserve(256);
    inst.reserve(g_world.size() + live.size()); hits.reserve(g_world.size() + live.size());
    auto heapOf = [&](uint32_t texId) -> uint32_t {
        if (texId != 0xFFFFFFFFu) { auto tit = g_tex.find(texId); if (tit != g_tex.end() && tit->second) return TEX_SRV_BASE + texId; }
        return 0;
    };
    auto pushInst = [&](ID3D12Resource* blas, const float* xf, uint32_t instID, D3D12_GPU_VIRTUAL_ADDRESS ibVA,
                        D3D12_GPU_VIRTUAL_ADDRESS vbVA, uint32_t stride, uint32_t istride, uint32_t startIndex,
                        uint32_t baseVertex, uint32_t texId, uint32_t uvOff, uint32_t posOff, uint32_t flags,
                        uint32_t emissiveTexId, uint32_t specTexId, uint32_t normalTexId, uint32_t depthTexId,
                        const float* specParams, const float* parallaxParams) {
        if (inst.size() >= MAX_INST) return;
        D3D12_RAYTRACING_INSTANCE_DESC id = {};
        memcpy(id.Transform, xf, 12 * sizeof(float));
        id.InstanceID = instID;
        // A merged base+emission surface remains ordinary shadow-casting world geometry.
        // A standalone additive card is visible to camera/GI but not to shadow rays.
        id.InstanceMask = (flags & DRAW_FLAG_SKY) ? 0x02 :
                          ((flags & DRAW_FLAG_EMISSIVE) && texId == 0xFFFFFFFFu) ? 0x04 : 0x01;
        id.InstanceContributionToHitGroupIndex = (UINT)inst.size();
        id.AccelerationStructure = blas->GetGPUVirtualAddress();
        uint32_t texHeap = heapOf(texId);
        if (flags & DRAW_FLAG_SKY) texHeap |= 0x80000000u;        // emissive sky-dome marker
        uint32_t emissiveHeap = heapOf(emissiveTexId);
        if ((flags & DRAW_FLAG_EMISSIVE_ALPHA) && emissiveHeap) emissiveHeap |= 0x80000000u;
        if ((flags & DRAW_FLAG_PARTICLE) && emissiveHeap) emissiveHeap |= 0x40000000u;   // pure emitter: zero albedo
        uint32_t normalHeap = heapOf(normalTexId);
        if (normalHeap && normalTexId < h->texCount && bridge_texs((void*)h)[normalTexId].fmt == BTEX_BC3)
            normalHeap |= 0x80000000u; // BC3 idTech normal: X in alpha, Y in green
        HitRec hr = { ibVA, vbVA, stride, istride, startIndex, baseVertex, texHeap, uvOff, posOff,
                      heapOf(specTexId), normalHeap, emissiveHeap, heapOf(depthTexId),
                      { specParams[0], specParams[1], specParams[2], specParams[3] },
                      { parallaxParams[0], parallaxParams[1], parallaxParams[2], parallaxParams[3] },
                      (flags & DRAW_FLAG_PARTICLE) ? 1u : 0u };
        inst.push_back(id); hits.push_back(hr);
    };
    static bool s_noWorld = (getenv("WOLF_NOWORLD") != nullptr);   // diagnostic: live draws only
    if (!s_noWorld)
    for (auto& kv : g_world) {
        if (liveKeys.count(kv.first)) continue;   // drawn live this frame -> live copy wins (no frozen duplicate)
        const PInst& p = kv.second; pushInst(p.blas, p.xform, p.instanceID, p.ibVA, p.vbVA, p.stride, p.istride, p.startIndex, p.baseVertex, p.texId, p.uvOff, p.posOff, p.flags, p.emissiveTexId, p.specTexId, p.normalTexId, p.depthTexId, p.specParams, p.parallaxParams);
    }
    for (auto& ld : live)     { pushInst(ld.blas, ld.xform, ld.instanceID, ld.ibVA, ld.vbVA, ld.stride, ld.istride, ld.startIndex, ld.baseVertex, ld.texId, ld.uvOff, ld.posOff, ld.flags, ld.emissiveTexId, ld.specTexId, ld.normalTexId, ld.depthTexId, ld.specParams, ld.parallaxParams); }

    // Expanded particle billboards: per-frame BLAS per range -> emissive instances.
    {
        static bool s_noParticles = (getenv("WOLF_NOPARTICLES") != nullptr);
        static bool s_noFog = (getenv("WOLF_NOFOG") != nullptr);
        const BridgeParticles& pp = h->particles;
        uint32_t seq = pp.sequence;
        if (pp.valid && !(seq & 1u) && pp.rangeCount && pp.rangeCount <= BRIDGE_MAX_PART_RANGES &&
            pp.vertCount && (uint64_t)pp.vertCount * sizeof(BridgePartVert) <= BRIDGE_PARTICLE_BYTES) {
            static std::vector<BridgePartRange> pr; static std::vector<BridgePartVert> pvv;
            pr.resize(pp.rangeCount); pvv.resize(pp.vertCount);
            MemoryBarrier();
            memcpy(pr.data(),  bridge_part_ranges((void*)h), (size_t)pp.rangeCount * sizeof(BridgePartRange));
            memcpy(pvv.data(), bridge_part_verts((void*)h),  (size_t)pp.vertCount  * sizeof(BridgePartVert));
            MemoryBarrier();
            if (h->particles.sequence == seq) {   // not torn
                    if (g_partSeqIBCount < pp.vertCount) {          // grow shared sequential index buffer
                        uint32_t want = pp.vertCount + 8192u;
                        SAFE_RELEASE(g_partSeqIB);
                        g_partSeqIB = create_buffer((UINT64)want * 4u, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
                        g_partSeqIBCount = 0;
                        if (g_partSeqIB) {
                            uint32_t* ip = nullptr; D3D12_RANGE none = {0,0};
                            if (SUCCEEDED(g_partSeqIB->Map(0, &none, (void**)&ip))) {
                                for (uint32_t k = 0; k < want; ++k) ip[k] = k;
                                g_partSeqIB->Unmap(0, nullptr); g_partSeqIBCount = want;
                            } else SAFE_RELEASE(g_partSeqIB);
                        }
                    }
                    if (g_partSeqIB) {
                        D3D12_GPU_VIRTUAL_ADDRESS ibVA = g_partSeqIB->GetGPUVirtualAddress();
                        const float ident[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
                        int added = 0, skipTex = 0, skipBlas = 0, skipCls = 0;
                        uint32_t classSeen[3] = {};
                        uint32_t firstTex = 0xFFFFFFFFu, firstHeap = 0;
                        // Blend state is not a material identity: Wolf also draws
                        // some smoke frames premultiplied. Any texture observed in
                        // a true volumetric range is smoke for all of its ranges.
                        std::unordered_set<uint32_t> fogTexIds;
                        for(uint32_t ri=0;ri<pp.rangeCount;++ri) if(pr[ri].cls==PART_VOLUMETRIC) fogTexIds.insert(pr[ri].texId);
                        std::vector<uint32_t> selected;
                        selected.reserve(min(pp.rangeCount, 128u));
                        for (uint32_t ri = 0; ri < pp.rangeCount && selected.size() < 128; ++ri) {
                            const BridgePartRange& r = pr[ri];
                            if (ri == 0) { firstTex = r.texId; firstHeap = heapOf(r.texId); }
                            if ((r.cls != PART_EMISSIVE && r.cls != PART_VOLUMETRIC && r.cls != PART_EMISSIVE_ALPHA) || r.vertCount < 3) { skipCls++; continue; }
                            ++classSeen[r.cls];
                            // Wolf's smoke sprites are deliberately enormous soft
                            // quads. Intersecting those cards directly makes their
                            // edges clip through the camera and creates pathological
                            // any-hit overdraw. Keep them in the bridge for a later
                            // density-volume conversion, but do not put the cards in
                            // the particle BLAS. Both fire families remain active.
                            bool isFog = r.cls == PART_VOLUMETRIC || fogTexIds.count(r.texId) != 0;
                            if (isFog) {
                                uint32_t fogTexHeap=heapOf(r.texId);
                                if (fogTexHeap && !s_noFog) {
                                    // The expanded stream is a triangle list; Wolf's
                                    // billboards are two triangles / six vertices.
                                    for(uint32_t vo=0;vo+5<r.vertCount && fogVolumes.size()<512;vo+=6){
                                        FogVolume fv={};
                                        if(make_fog_volume(&pvv[r.firstVert+vo],fogTexHeap,fv)) fogVolumes.push_back(fv);
                                    }
                                }
                                skipCls++; continue;
                            }
                            if ((uint64_t)r.firstVert + r.vertCount > pp.vertCount) continue;
                            if (!heapOf(r.texId)) { skipTex++; continue; }   // texture not uploaded yet
                            if (s_noParticles) { skipCls++; continue; }
                            selected.push_back(ri);
                        }
                        std::vector<GpuPartVert> fireVerts;
                        std::stable_sort(selected.begin(), selected.end(), [&](uint32_t a, uint32_t b) {
                            const BridgePartVert& va = pvv[pr[a].firstVert];
                            const BridgePartVert& vb = pvv[pr[b].firstVert];
                            auto d2 = [&](const BridgePartVert& v) {
                                float x=v.pos[0]-eye[0], y=v.pos[1]-eye[1], z=v.pos[2]-eye[2];
                                return x*x+y*y+z*z;
                            };
                            return d2(va) < d2(vb);
                        });
                        fireVerts.reserve(std::min<uint32_t>(MAX_PARTICLE_FIRE_VERTS, pp.vertCount));
                        for (uint32_t ri : selected) {
                            const BridgePartRange& r = pr[ri];
                            uint32_t texHeap = heapOf(r.texId);
                            uint32_t room = MAX_PARTICLE_FIRE_VERTS - (uint32_t)fireVerts.size();
                            uint32_t take = min(r.vertCount, room);
                            take -= take % 3u;
                            for (uint32_t vi = 0; vi < take; ++vi) {
                                const BridgePartVert& src = pvv[r.firstVert + vi];
                                GpuPartVert dst = {};
                                memcpy(dst.pos, src.pos, sizeof(dst.pos));
                                memcpy(dst.uv, src.uv, sizeof(dst.uv));
                                dst.color = src.color;
                                dst.texHeap = texHeap;
                                dst.cls = r.cls;
                                fireVerts.push_back(dst);
                            }
                            if (fireVerts.size() >= MAX_PARTICLE_FIRE_VERTS) break;
                        }
                        if (!fireVerts.empty() && inst.size() < MAX_INST && hits.size() < MAX_INST) {
                            UINT64 vbBytes = (UINT64)fireVerts.size() * sizeof(GpuPartVert);
                            ID3D12Resource* partVB = create_buffer(vbBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
                            if (!partVB) { skipBlas = (int)selected.size(); }
                            ID3D12Resource* blas = nullptr;
                            D3D12_GPU_VIRTUAL_ADDRESS vbVA = 0;
                            if (partVB) {
                                upload_bytes(partVB, fireVerts.data(), (UINT)vbBytes);
                                g_frameUploads.push_back(partVB);
                                vbVA = partVB->GetGPUVirtualAddress();
                                blas = build_particle_blas(vbVA, ibVA, (uint32_t)fireVerts.size());
                            }
                            if (!blas) {
                                skipBlas = (int)selected.size();
                            } else {
                                g_frameUploads.push_back(blas);

                                D3D12_RAYTRACING_INSTANCE_DESC id = {};
                                memcpy(id.Transform, ident, sizeof(ident));
                                id.InstanceID = 0xE00000u;
                                id.InstanceMask = 0x04;
                                id.InstanceContributionToHitGroupIndex = (UINT)hits.size();
                                id.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
                                id.AccelerationStructure = blas->GetGPUVirtualAddress();
                                inst.push_back(id);

                                HitRec hr = {};
                                hr.ibVA = ibVA; hr.vbVA = vbVA;
                                hr.stride = (uint32_t)sizeof(GpuPartVert); hr.istride = 4u;
                                hr.startIndex = 0u; hr.baseVertex = 0u;
                                hr.texHeap = 0u; hr.uvOff = 12u; hr.posOff = 0u;
                                hr.emissiveHeap = 0x40000000u;
                                hr.isParticle = 1u;
                                hits.push_back(hr);
                                added = (int)selected.size();
                            }
                        }
                        static int s_pdbg = 0;
                        if ((s_pdbg++ % 30) == 0) printf("[rt] particles: ranges=%u verts=%u class(E/V/A)=%u/%u/%u added=%d (skipTex=%d skipBlas=%d skipCls=%d firstTex=%u firstHeap=%u)\n",
                            pp.rangeCount, pp.vertCount, classSeen[0], classSeen[1], classSeen[2], added,
                            skipTex, skipBlas, skipCls, firstTex, firstHeap);
                    }
            }
        }
    }

    // Center-distance order gives stable front-to-back compositing for the
    // bounded volume list. Prefer the nearest smoke when an extreme effect emits
    // more than the fixed budget.
    std::sort(fogVolumes.begin(),fogVolumes.end(),[&](const FogVolume& a,const FogVolume& b){
        auto d2=[&](const FogVolume& v){float x=v.centerDensity[0]-eye[0],y=v.centerDensity[1]-eye[1],z=v.centerDensity[2]-eye[2];return x*x+y*y+z*z;};
        return d2(a)<d2(b);
    });
    if(fogVolumes.size()>MAX_FOG_VOLUMES) fogVolumes.resize(MAX_FOG_VOLUMES);
    if(!fogVolumes.empty()) upload_bytes(g_fogVolumeBuf,fogVolumes.data(),(UINT)(fogVolumes.size()*sizeof(FogVolume)));

    // Fill the per-instance hit shader table (id + IB VA + VB VA + geo constants).
    if (!hits.empty()) {
        uint8_t* hm = nullptr; D3D12_RANGE none = {0, 0};
        if (SUCCEEDED(g_sbtHit->Map(0, &none, (void**)&hm))) {
            for (size_t r = 0; r < hits.size(); ++r) {
                uint8_t* rec = hm + r * HIT_REC;
                memcpy(rec + 0,  hits[r].isParticle ? g_partHitId : g_hitId, 32);
                memcpy(rec + 32, &hits[r].ibVA, 8);
                memcpy(rec + 40, &hits[r].vbVA, 8);
                uint32_t c[20] = {};
                c[0] = hits[r].stride; c[1] = hits[r].istride; c[2] = hits[r].startIndex; c[3] = hits[r].baseVertex;
                c[4] = hits[r].texHeap; c[5] = hits[r].uvOff; c[6] = hits[r].posOff;
                c[7] = hits[r].specHeap; c[8] = hits[r].normalHeap;
                c[9] = hits[r].emissiveHeap;
                c[10] = hits[r].depthHeap;
                memcpy(&c[12], hits[r].specParams, 16);   // float4 begins on a 16-byte cbuffer register
                memcpy(&c[16], hits[r].parallaxParams, 16);
                memcpy(rec + 48, c, 80);
            }
            g_sbtHit->Unmap(0, nullptr);
        }
    }
    LARGE_INTEGER tCpu1; QueryPerformanceCounter(&tCpu1);
    g_cpuMs += qpc_ms(tCpu0, tCpu1);

    static int dbg = 0;
    if ((dbg++ % 30) == 0)
        printf("[rt] draws=%u live=%zu world=%zu vbCount=%u v2wFound=%d eye=(%.0f %.0f %.0f)\n",
               nd, live.size(), g_world.size(), h->vbCount, v2wFound ? 1 : 0, eye[0], eye[1], eye[2]);

    if (inst.size() > MAX_INST) inst.resize(MAX_INST);
    if (!inst.empty()) {
        upload_bytes(g_instBuf, inst.data(), (UINT)(inst.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC)));

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; in.NumDescs = (UINT)inst.size();
        in.InstanceDescs = g_instBuf->GetGPUVirtualAddress();
        in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
        bd.Inputs = in; bd.ScratchAccelerationStructureData = g_tlasScratch->GetGPUVirtualAddress();
        bd.DestAccelerationStructureData = g_tlasResult->GetGPUVirtualAddress();
        g_cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
        uav_barrier();

        // (Re)create the TLAS SRV at heap slot 1.
        D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sv.RaytracingAccelerationStructure.Location = g_tlasResult->GetGPUVirtualAddress();
        D3D12_CPU_DESCRIPTOR_HANDLE sh = g_srvHeap->GetCPUDescriptorHandleForHeapStart(); sh.ptr += g_srvInc;
        g_dev->CreateShaderResourceView(nullptr, &sv, sh);

        // Dispatch primary rays into g_outTex.
        ID3D12DescriptorHeap* heaps[] = { g_srvHeap };
        g_cl->SetDescriptorHeaps(1, heaps);
        g_cl->SetComputeRootSignature(g_rtRS);
        g_cl->SetComputeRootDescriptorTable(0, g_srvHeap->GetGPUDescriptorHandleForHeapStart());
        D3D12_GPU_DESCRIPTOR_HANDLE sharcTable = g_srvHeap->GetGPUDescriptorHandleForHeapStart();
        sharcTable.ptr += (UINT64)SHARC_HASH_UAV_SLOT * g_srvInc;
        g_cl->SetComputeRootDescriptorTable(3, sharcTable);

        // Clear the persistent sky env map on init / level change (alpha 0 = uncovered).
        if (g_skyEnvClear && g_skyEnvTex && g_clearHeap) {
            D3D12_GPU_DESCRIPTOR_HANDLE gh = g_srvHeap->GetGPUDescriptorHandleForHeapStart();
            gh.ptr += (UINT64)SKYENV_UAV_SLOT * g_srvInc;
            const float zero[4] = { 0, 0, 0, 0 };
            g_cl->ClearUnorderedAccessViewFloat(gh, g_clearHeap->GetCPUDescriptorHandleForHeapStart(), g_skyEnvTex, zero, 0, nullptr);
            uav_barrier();
            g_skyEnvClear = false;
        }

        // All SHARC resources must begin at zero.  Also reset them on a level
        // transition because their keys encode absolute world-space locations.
        if (g_sharcClear && g_clearHeap) {
            const UINT zero[4] = {0,0,0,0};
            ID3D12Resource* resources[3] = { g_sharcHash, g_sharcAccum, g_sharcResolved };
            const UINT slots[3] = { SHARC_HASH_UAV_SLOT, SHARC_ACCUM_UAV_SLOT, SHARC_RESOLVED_UAV_SLOT };
            for (UINT si = 0; si < 3; ++si) {
                D3D12_GPU_DESCRIPTOR_HANDLE gh = g_srvHeap->GetGPUDescriptorHandleForHeapStart(); gh.ptr += (UINT64)slots[si] * g_srvInc;
                D3D12_CPU_DESCRIPTOR_HANDLE ch = g_clearHeap->GetCPUDescriptorHandleForHeapStart(); ch.ptr += (SIZE_T)(si + 1) * g_srvInc;
                g_cl->ClearUnorderedAccessViewUint(gh, ch, resources[si], zero, 0, nullptr);
            }
            uav_barrier();
            g_sharcClear = false;
        }

        if (g_restirClear && g_clearHeap) {
            const UINT zero[4] = {0,0,0,0};
            ID3D12Resource* resources[4] = { g_restirReservoir[0], g_restirReservoir[1],
                                             g_restirSurface[0], g_restirSurface[1] };
            const UINT slots[4] = { RESTIR_RES0_UAV_SLOT, RESTIR_RES1_UAV_SLOT,
                                    RESTIR_SURF0_UAV_SLOT, RESTIR_SURF1_UAV_SLOT };
            for (UINT ri = 0; ri < 4; ++ri) {
                D3D12_GPU_DESCRIPTOR_HANDLE gh = g_srvHeap->GetGPUDescriptorHandleForHeapStart();
                gh.ptr += (UINT64)slots[ri] * g_srvInc;
                D3D12_CPU_DESCRIPTOR_HANDLE ch = g_clearHeap->GetCPUDescriptorHandleForHeapStart();
                ch.ptr += (SIZE_T)(ri + 4) * g_srvInc;
                g_cl->ClearUnorderedAccessViewUint(gh, ch, resources[ri], zero, 0, nullptr);
            }
            uav_barrier();
            g_restirClear = false;
        }

        // Real sun from the game's global-light pass (cached; falls back until seen).
        if (h->lighting.sunValid) {
            g_sunDir[0] = h->lighting.sunDir[0]; g_sunDir[1] = h->lighting.sunDir[1]; g_sunDir[2] = h->lighting.sunDir[2];
            float I = h->lighting.sunColor[3] > 0.0f ? h->lighting.sunColor[3] : 1.0f;
            g_sunCol[0] = h->lighting.sunColor[0]*I; g_sunCol[1] = h->lighting.sunColor[1]*I; g_sunCol[2] = h->lighting.sunColor[2]*I;
            if (!g_sunSeen) { printf("[rt] real sun: dir=(%.3f %.3f %.3f) color=(%.2f %.2f %.2f)\n", g_sunDir[0],g_sunDir[1],g_sunDir[2], g_sunCol[0],g_sunCol[1],g_sunCol[2]); g_sunSeen = true; }
        }
        const float* worldSun = g_sunDir;
        for (auto it = g_emitterAnchors.begin(); it != g_emitterAnchors.end(); ) {
            if (g_frameNo - it->lastSeen > 600) it = g_emitterAnchors.erase(it); else ++it;
        }
        uint32_t observedCount = h->lighting.lightCount;
        if (observedCount > BRIDGE_MAX_LIGHTS) observedCount = BRIDGE_MAX_LIGHTS;
        std::vector<BridgeLight> observed(observedCount);
        if (observedCount) {
            memcpy(observed.data(), bridge_lights((void*)h), observedCount * sizeof(BridgeLight));
            // Wolf's deferred-light shaders provide $vViewLightPosition. Convert
            // those positions into the world space used by the DXR scene.
            for (BridgeLight& l : observed) if (l.space == BLIGHT_VIEW_SPACE) {
                float v[3] = { l.origin[0], l.origin[1], l.origin[2] };
                l.origin[0] = eye[0] + g_viewToWorld[0]*v[0] + g_viewToWorld[1]*v[1] + g_viewToWorld[2]*v[2];
                l.origin[1] = eye[1] + g_viewToWorld[4]*v[0] + g_viewToWorld[5]*v[1] + g_viewToWorld[6]*v[2];
                l.origin[2] = eye[2] + g_viewToWorld[8]*v[0] + g_viewToWorld[9]*v[1] + g_viewToWorld[10]*v[2];
                l.space = BLIGHT_WORLD_SPACE;
            }
            // A localized deferred volume is still not necessarily a fixture:
            // Wolf uses many invisible fill lights. Keep only volumes spatially
            // attached to emissive geometry captured from lamps, flames, etc.
            static bool s_anyVolume = (getenv("WOLF_LIGHT_ANY_VOLUME") != nullptr);
            if (!s_anyVolume) {
                std::vector<BridgeLight> fixtureLights; fixtureLights.reserve(observed.size());
                for (const BridgeLight& l : observed) {
                    float radius = l.invRadius > 1e-7f ? 1.0f/l.invRadius : 0.0f;
                    // Light origins authored for lamps/torches sit very close to
                    // their glowing mesh.  Room-fill lights merely sharing the
                    // vicinity no longer qualify.
                    float attach = fmaxf(36.0f, fminf(radius * 0.20f, 80.0f));
                    bool nearEmitter = false;
                    for (const EmitterAnchor& a : g_emitterAnchors) {
                        float dx=a.p[0]-l.origin[0], dy=a.p[1]-l.origin[1], dz=a.p[2]-l.origin[2];
                        if (dx*dx+dy*dy+dz*dz <= attach*attach) { nearEmitter=true; break; }
                    }
                    if (nearEmitter) fixtureLights.push_back(l);
                }
                observed.swap(fixtureLights);
                observedCount = (uint32_t)observed.size();
            }
            // Light-volume passes can be culled briefly. Merge into a short
            // world-space cache instead of allowing direct illumination to pop.
            for (const BridgeLight& l : observed) {
                CachedLight* match = nullptr;
                float radius = l.invRadius > 1e-7f ? 1.0f / l.invRadius : 100.0f;
                float tolerance = radius * 0.02f;
                if (tolerance < 4.0f) tolerance = 4.0f;
                for (CachedLight& c : g_lightCache) {
                    float dx=c.light.origin[0]-l.origin[0], dy=c.light.origin[1]-l.origin[1], dz=c.light.origin[2]-l.origin[2];
                    if (dx*dx + dy*dy + dz*dz <= tolerance*tolerance) { match = &c; break; }
                }
                if (match) { match->light = l; match->lastSeen = g_frameNo; }
                else if (g_lightCache.size() < BRIDGE_MAX_LIGHTS) g_lightCache.push_back({l, g_frameNo});
            }
        }
        for (auto it = g_lightCache.begin(); it != g_lightCache.end(); ) {
            // Light volumes can be briefly culled, but a 180-frame hold made old
            // lights accumulate across rooms.  Eight frames hides submission gaps
            // while keeping animated/off lights responsive.
            if (g_frameNo - it->lastSeen > 8) it = g_lightCache.erase(it); else ++it;
        }
        uint32_t lightCount = (uint32_t)g_lightCache.size();
        std::vector<BridgeLight> frameLights; frameLights.reserve(lightCount);
        for (const CachedLight& c : g_lightCache) {
            BridgeLight l = c.light;
            uint32_t age = g_frameNo - c.lastSeen;
            if (age > 2) l.color[3] *= fmaxf(0.0f, 1.0f - (float)(age - 2) / 6.0f);
            frameLights.push_back(l);
        }
        static bool s_noLights = (getenv("WOLF_NOLIGHTS") != nullptr);
        if (s_noLights) {
            lightCount = 0;
            static bool s_reported = false;
            if (!s_reported) { printf("[rt] WOLF_NOLIGHTS: converted local lights disabled (sun/emissives remain)\n"); s_reported = true; }
        }
        if (lightCount) upload_bytes(g_lightBuf, frameLights.data(), lightCount * sizeof(BridgeLight));
        { static int s_lc = 0; if ((s_lc++ % 60) == 0) printf("[rt] light cache size=%u (observed this frame=%u)\n", lightCount, observedCount); }
        if (lightCount && !g_lightsSeen) {
            const BridgeLight& l = frameLights[0];
            printf("[rt] local lights active: count=%u first=(%.1f %.1f %.1f) invRadius=%.6f\n",
                   lightCount, l.origin[0], l.origin[1], l.origin[2], l.invRadius);
            g_lightsSeen = true;
        }

        // Sky source: prefer the game's LIVE sky (screen-space render target). Primary
        // sky rays paint it into a persistent lat-long env map by direction, so
        // reflections/GI/looking-around read the real sky (see sky_radiance). The
        // captured $EnvironmentMap cube is only a last resort (it reads back black).
        // Sky: analytic base (shader, driven by the real sun) everywhere, with the
        // live framebuffer sky overlaid when available (mode 2 -> accumulated env map).
        // The $EnvironmentMap cube is unused (it reads back black).
        uint32_t skyHeap = 0, skyMode = 0;
        float skyScale[4] = { 1, 1, 1, 1 };
        if (g_screenSkyTex) {
            skyHeap = TEX_SRV_BASE - 1u;
            skyMode = 2u; // live screen-space sky -> accumulated into g_skyEnvTex
            if (!g_skySeen) { printf("[rt] live sky active (analytic base + real overlay)\n"); g_skySeen = true; }
        }
        float sx = h->camera.proj[0], sy = h->camera.proj[5];

        // Camera matrices for RR (row-major, row-vector).
        float W2V[16], V2C[16];
        build_world_to_view(g_viewToWorld, eye, W2V);
        build_view_to_clip(sx, sy, V2C);
        // Rotation-only world->clip (translation zeroed) for eye-relative MV reprojection.
        float R4[16]; memcpy(R4, W2V, sizeof(R4)); R4[3]=R4[7]=R4[11]=0; R4[12]=R4[13]=R4[14]=0; R4[15]=1;
        float relWC[16]; mat4_mul(R4, V2C, relWC);
        float camDelta[3] = { eye[0]-g_prevEye[0], eye[1]-g_prevEye[1], eye[2]-g_prevEye[2] };

        // Deterministic Halton jitter, in pixels, range ~[-0.5, 0.5].
        static bool s_noJitter = (getenv("WOLF_NOJITTER") != nullptr);
        float jitterX = halton(g_jitterPhase, 2) - 0.5f;
        float jitterY = halton(g_jitterPhase, 3) - 0.5f;
        if (s_noJitter) { jitterX = 0.0f; jitterY = 0.0f; }   // diagnostic: is the residual the jitter?
        g_jitterPhase = (g_jitterPhase + 1) % g_jitterPhases;   // phase count scales with upscale ratio

        // A large single-frame jump (teleport) invalidates motion vectors -> reset RR.
        float eyeJump = fabsf(eye[0]-g_prevEye[0]) + fabsf(eye[1]-g_prevEye[1]) + fabsf(eye[2]-g_prevEye[2]);
        if (eyeJump > 500.0f) g_dlssReset = true;

        // Diagnostic: is the RECOVERED camera stable frame-to-frame when the player holds still?
        if (getenv("WOLF_CAMLOG")) {
            static float pv[12] = {0}; float vd = 0;
            for (int c = 0; c < 12; ++c) { vd += fabsf(g_viewToWorld[c] - pv[c]); pv[c] = g_viewToWorld[c]; }
            printf("[cam] eyeD=%.5f v2wD=%.6f eye=(%.3f %.3f %.3f)\n", eyeJump, vd, eye[0], eye[1], eye[2]);
        }

        static uint32_t g_accFrame = 0; g_accFrame++;
        static float s_mv0 = (getenv("WOLF_MV0") ? 1.0f : 0.0f);   // debug: force zero motion vectors
        float cam[56] = {
            g_viewToWorld[0], g_viewToWorld[1], g_viewToWorld[2],  sx,
            g_viewToWorld[4], g_viewToWorld[5], g_viewToWorld[6],  sy,
            g_viewToWorld[8], g_viewToWorld[9], g_viewToWorld[10], V2C[10],   // gVW2.w = proj A (f/(n-f))
            eye[0], eye[1], eye[2], (float)g_rw,      // ray-gen grid width = RENDER res
            worldSun[0], worldSun[1], worldSun[2], (float)g_rh,   // render res height
            (float)g_accFrame, s_mv0, jitterX, jitterY,
            g_prevRelWC[0],  g_prevRelWC[1],  g_prevRelWC[2],  g_prevRelWC[3],
            g_prevRelWC[4],  g_prevRelWC[5],  g_prevRelWC[6],  g_prevRelWC[7],
            g_prevRelWC[8],  g_prevRelWC[9],  g_prevRelWC[10], g_prevRelWC[11],
            g_prevRelWC[12], g_prevRelWC[13], g_prevRelWC[14], g_prevRelWC[15],
            camDelta[0], camDelta[1], camDelta[2], V2C[14],                  // gCamDelta.w = proj C (n*f/(n-f))
            g_sunCol[0], g_sunCol[1], g_sunCol[2], 0.0f,                     // gSunColor = real sun radiance
            skyScale[0], skyScale[1], skyScale[2], skyScale[3],              // captured sky tint/exposure
            0, 0, 0, 0,                                                       // raw uint lighting descriptor data below
        };
        uint32_t envSlot = g_skyEnvTex ? SKYENV_UAV_SLOT : 0u;   // gSkyColor.w = env-map UAV index
        memcpy(cam + 51, &envSlot, 4);
        uint32_t lightingInfo[4] = { LIGHT_SRV_SLOT, lightCount, skyHeap, skyMode };
        memcpy(cam + 52, lightingInfo, sizeof(lightingInfo));
        g_cl->SetComputeRoot32BitConstants(1, 56, cam, 0);

        // Multi-layer sky (b2 CBV): 16 floats of cloud-layer texture affines + a uint4
        // of their heap slots. Layout matches cbuffer SkyLayers: gSkyL1x0/x1/gSkyL2x0/x1, gSkyLayers.
        // The client republishes lighting every frame and resets its layer count in
        // begin_frame, so (like the light cache) a bare read often catches 0 mid-swap.
        // Cache the last non-empty layer set and reuse it, so the sky doesn't drop out.
        static BridgeLighting s_skyCache = {}; static bool s_skyCacheValid = false;
        if (h->lighting.skyLayerCount > 0) { s_skyCache = h->lighting; s_skyCacheValid = true; }
        {
            float skyCB[48] = {0};
            const BridgeLighting& LT = s_skyCacheValid ? s_skyCache : h->lighting;
            for (uint32_t li = 0; li < LT.skyLayerCount && li < 2; ++li) {
                const BridgeSkyLayer& sl = LT.skyLayers[li];
                ((uint32_t*)skyCB)[16 + li*2 + 0] = (sl.colorTexId != 0xFFFFFFFFu && g_tex.count(sl.colorTexId)) ? TEX_SRV_BASE + sl.colorTexId : 0u;
                ((uint32_t*)skyCB)[16 + li*2 + 1] = (sl.alphaTexId != 0xFFFFFFFFu && g_tex.count(sl.alphaTexId)) ? TEX_SRV_BASE + sl.alphaTexId : 0u;
                skyCB[li*8+0]=sl.xform[0]; skyCB[li*8+1]=sl.xform[1]; skyCB[li*8+2]=sl.xform[2]; skyCB[li*8+3]=0.0f;
                skyCB[li*8+4]=sl.xform[3]; skyCB[li*8+5]=sl.xform[4]; skyCB[li*8+6]=sl.xform[5]; skyCB[li*8+7]=0.0f;
            }
            // Quadratic dome dir->UV coeffs (static per map; read direct, no race).
            // gSkyUu[3] = u[0..9] (2 pad), gSkyVv[3] = v[0..9] (2 pad); gSkyMisc.x = valid.
            const float* M = h->lighting.skyUVMat;
            for (int i = 0; i < 10; ++i) { skyCB[20+i] = M[i]; skyCB[32+i] = M[10+i]; }
            ((uint32_t*)skyCB)[44] = h->lighting.skyUVValid;
            upload_bytes(g_skyCB, skyCB, sizeof(skyCB));
            g_cl->SetComputeRootConstantBufferView(2, g_skyCB->GetGPUVirtualAddress());
        }
        static bool s_noSharc = (getenv("WOLF_NOSHARC") != nullptr);
        const uint32_t sharcEnabled = s_noSharc ? 0u : 1u;
        // b3 layout: camera+scale, capacity/enabled/frame/block, previous camera,
        // resolve history/responsive/stale.  5x5 sparse updates = ~4% paths.
        float sharcCB[16] = { eye[0], eye[1], eye[2], 16.0f };
        uint32_t* sharcU = (uint32_t*)sharcCB;
        sharcU[4] = SHARC_CAPACITY; sharcU[5] = sharcEnabled; sharcU[6] = g_accFrame; sharcU[7] = 5u;
        sharcCB[8] = g_sharcPrevEye[0]; sharcCB[9] = g_sharcPrevEye[1]; sharcCB[10] = g_sharcPrevEye[2];
        sharcU[12] = 12u; sharcU[13] = 4u; sharcU[14] = 64u; // faster response to animated emitters/lights
        upload_bytes(g_sharcCB, sharcCB, sizeof(sharcCB));
        g_cl->SetComputeRootConstantBufferView(4, g_sharcCB->GetGPUVirtualAddress());

        // ReSTIR DI history ping-pongs once per rendered frame. Animated lights
        // remain responsive because reservoirs store an index/weight, not radiance.
        static bool s_noRestir = (getenv("WOLF_NORESTIR") != nullptr);
        uint32_t restirCB[8] = {};
        UINT curRi = g_accFrame & 1u, prevRi = curRi ^ 1u;
        restirCB[0] = curRi ? RESTIR_RES1_UAV_SLOT : RESTIR_RES0_UAV_SLOT;
        restirCB[1] = prevRi ? RESTIR_RES1_UAV_SLOT : RESTIR_RES0_UAV_SLOT;
        restirCB[2] = curRi ? RESTIR_SURF1_UAV_SLOT : RESTIR_SURF0_UAV_SLOT;
        restirCB[3] = prevRi ? RESTIR_SURF1_UAV_SLOT : RESTIR_SURF0_UAV_SLOT;
        restirCB[4] = (uint32_t)g_rw; restirCB[5] = (uint32_t)g_rh;
        restirCB[6] = 20u; restirCB[7] = s_noRestir ? 0u : 8u;
        upload_bytes(g_restirCB, restirCB, sizeof(restirCB));
        g_cl->SetComputeRootConstantBufferView(5, g_restirCB->GetGPUVirtualAddress());
        g_cl->SetPipelineState1(g_rtSO);
        memcpy(g_prevRelWC, relWC, sizeof(relWC));       // this frame -> "prev" next frame
        g_prevEye[0]=eye[0]; g_prevEye[1]=eye[1]; g_prevEye[2]=eye[2];

        D3D12_DISPATCH_RAYS_DESC dr = {};
        D3D12_GPU_VIRTUAL_ADDRESS sbt = g_sbt->GetGPUVirtualAddress();
        dr.MissShaderTable.StartAddress = sbt + 128; dr.MissShaderTable.SizeInBytes = 64; dr.MissShaderTable.StrideInBytes = 32;
        dr.HitGroupTable.StartAddress = g_sbtHit->GetGPUVirtualAddress();
        dr.HitGroupTable.SizeInBytes = (UINT64)hits.size() * HIT_REC; dr.HitGroupTable.StrideInBytes = HIT_REC;

        // 1) Sparse SHARC update; 2) resolve into temporal cache; 3) normal
        // render, which queries SHARC only at secondary diffuse vertices.
        if (sharcEnabled) {
            dr.RayGenerationShaderRecord.StartAddress = sbt + 64; dr.RayGenerationShaderRecord.SizeInBytes = 32;
            dr.Width = (g_rw + 4) / 5; dr.Height = (g_rh + 4) / 5; dr.Depth = 1;
            g_cl->DispatchRays(&dr);
            uav_barrier();
            g_cl->SetPipelineState(g_sharcResolvePSO);
            g_cl->Dispatch((SHARC_CAPACITY + 255u) / 256u, 1, 1);
            uav_barrier();
            g_cl->SetPipelineState1(g_rtSO);
        }
        dr.RayGenerationShaderRecord.StartAddress = sbt + 0; dr.RayGenerationShaderRecord.SizeInBytes = 32;
        dr.Width = g_rw; dr.Height = g_rh; dr.Depth = 1;   // ray-trace at render res
        g_cl->DispatchRays(&dr);
        uav_barrier();
        g_sharcPrevEye[0]=eye[0]; g_sharcPrevEye[1]=eye[1]; g_sharcPrevEye[2]=eye[2];

        // --- Denoise: DLSS Ray Reconstruction (color + G-buffer -> g_ppTex) ---
        static bool s_noRR = (getenv("WOLF_NORR") != nullptr);
        bool didRR = false;
        if (dlss_ready() && !dlssJustCreated && !s_noRR) {
            DlssFrame f = {};
            f.color = g_colorTex; f.output = g_ppTex; f.depth = g_depthTex; f.motion = g_mvTex;
            f.normal = g_normTex; f.albedo = g_albTex; f.specAlbedo = g_specTex; f.roughness = g_roughTex;
            f.jitterX = jitterX; f.jitterY = jitterY; f.reset = g_dlssReset ? 1 : 0;
            f.worldToView = W2V; f.viewToClip = V2C;
            didRR = dlss_evaluate(g_cl, f);
            g_dlssReset = false;
        }
        if (!didRR && g_rw == g_w && g_rh == g_h) {
            // Fallback (RR unavailable, native res): copy raw noisy color into g_ppTex.
            // (When upscaling, this only happens on the 1-frame RR create; skip -> 1 black frame.)
            D3D12_RESOURCE_BARRIER cb[2] = {};
            cb[0].Transition.pResource = g_colorTex; cb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; cb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            cb[1].Transition.pResource = g_ppTex;    cb[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; cb[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            for (auto& x : cb) { x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; }
            g_cl->ResourceBarrier(2, cb);
            g_cl->CopyResource(g_ppTex, g_colorTex);
            cb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE; cb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            cb[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;   cb[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            g_cl->ResourceBarrier(2, cb);
        }

        // --- Quarter-res textured ellipsoid smoke volumes -------------------
        // This is deliberately separate from DXR: no billboard intersections,
        // no any-hit overdraw, and no particle geometry in the TLAS.
        {
            ID3D12DescriptorHeap* fh[]={g_srvHeap}; g_cl->SetDescriptorHeaps(1,fh);
            g_cl->SetComputeRootSignature(g_fogRS);
            D3D12_GPU_DESCRIPTOR_HANDLE fogHandle=g_srvHeap->GetGPUDescriptorHandleForHeapStart();
            fogHandle.ptr+=(UINT64)FOG_UAV_SLOT*g_srvInc;
            g_cl->SetComputeRootDescriptorTable(0,fogHandle);
            g_cl->SetComputeRootShaderResourceView(1,g_fogVolumeBuf->GetGPUVirtualAddress());
            float fp[28]={
                g_viewToWorld[0],g_viewToWorld[1],g_viewToWorld[2],0,
                g_viewToWorld[4],g_viewToWorld[5],g_viewToWorld[6],0,
                g_viewToWorld[8],g_viewToWorld[9],g_viewToWorld[10],0,
                eye[0],eye[1],eye[2],0,
                sx,sy,V2C[10],V2C[14],
                g_sunCol[0],g_sunCol[1],g_sunCol[2],0,
                1.35f,0.10f,0.045f,0
            };
            uint32_t fogCount=(uint32_t)fogVolumes.size(); memcpy(&fp[15],&fogCount,4);
            g_cl->SetComputeRoot32BitConstants(2,28,fp,0);
            g_cl->SetPipelineState(g_fogPSO);
            g_cl->Dispatch(((g_w+3)/4+7)/8,((g_h+3)/4+7)/8,1);
            uav_barrier();
        }

        // --- Tonemap g_ppTex -> g_outTex (RR may have unbound our heap/state) ---
        ID3D12DescriptorHeap* th[] = { g_srvHeap };
        g_cl->SetDescriptorHeaps(1, th);
        g_cl->SetComputeRootSignature(g_tmRS);
        D3D12_GPU_DESCRIPTOR_HANDLE tmBase = g_srvHeap->GetGPUDescriptorHandleForHeapStart(); tmBase.ptr += (UINT64)8 * g_srvInc;
        g_cl->SetComputeRootDescriptorTable(0, tmBase);
        static bool veilWasActive = false;
        bool veilActive = h->veil.active != 0;
        if (veilActive != veilWasActive) {
            printf("[rt] Veil %s (game compositor detected)\n", veilActive ? "enter" : "exit");
            veilWasActive = veilActive;
        }
        uint32_t veilData[24] = {};
        auto veilHeap = [&](uint32_t id)->uint32_t { return (id!=0xFFFFFFFFu && g_tex.count(id)) ? TEX_SRV_BASE+id : 0u; };
        veilData[0]=veilHeap(h->veil.normal0TexId); veilData[1]=veilHeap(h->veil.normal1TexId);
        veilData[2]=veilHeap(h->veil.maskTexId); veilData[3]=veilActive?1u:0u;
        memcpy(veilData+4,h->veil.distanceControl,16); memcpy(veilData+8,h->veil.rampControl,16);
        memcpy(veilData+12,h->veil.normalXform0,16); memcpy(veilData+16,h->veil.normalXform1,16);
        memcpy(veilData+20,&h->veil.aspect,4);
        g_cl->SetComputeRoot32BitConstants(1,24,veilData,0);
        g_cl->SetPipelineState(g_tmPSO);
        g_cl->Dispatch((g_w + 7) / 8, (g_h + 7) / 8, 1);
        uav_barrier();
    }

    // Copy g_outTex -> current backbuffer, present.
    D3D12_RESOURCE_BARRIER b[2] = {};
    b[0].Transition.pResource = g_outTex;   b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b[1].Transition.pResource = g_back[i];  b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;          b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    for (auto& x : b) { x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; }
    g_cl->ResourceBarrier(2, b);
    g_cl->CopyResource(g_back[i], g_outTex);
    b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE; b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;   b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_cl->ResourceBarrier(2, b);

    g_cl->Close();
    ID3D12CommandList* lists[] = { g_cl };
    g_queue->ExecuteCommandLists(1, lists);
    gfx_drain_messages();
    static UINT s_sync = (getenv("WOLF_VSYNC") ? 1u : 0u);   // uncapped by default; WOLF_VSYNC re-enables vsync
    HRESULT pr = g_swap->Present(s_sync, 0);
    if (FAILED(pr) || FAILED(g_dev->GetDeviceRemovedReason())) {
        if (!g_lost) {
            g_lost = true;
            printf("[rt] DEVICE REMOVED present=0x%08lx reason=0x%08lx (frame draws=%u inst=%zu)\n",
                   pr, g_dev->GetDeviceRemovedReason(), nd, inst.size());
            ID3D12DeviceRemovedExtendedData* d = nullptr;
            if (SUCCEEDED(g_dev->QueryInterface(IID_PPV_ARGS(&d)))) {
                D3D12_DRED_PAGE_FAULT_OUTPUT pf = {};
                if (SUCCEEDED(d->GetPageFaultAllocationOutput(&pf)))
                    printf("[dred] PageFaultVA=0x%llx\n", (unsigned long long)pf.PageFaultVA);
                D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc = {};
                if (SUCCEEDED(d->GetAutoBreadcrumbsOutput(&bc))) {
                    for (const D3D12_AUTO_BREADCRUMB_NODE* n = bc.pHeadAutoBreadcrumbNode; n; n = n->pNext) {
                        UINT done = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0;
                        printf("[dred] cmdlist ops=%u lastCompleted=%u", n->BreadcrumbCount, done);
                        if (done < n->BreadcrumbCount) printf("  -> HUNG on op[%u]=%d", done, n->pCommandHistory[done]);
                        printf("\n");
                    }
                }
                d->Release();
            }
        }
        return;
    }

    // Fully wait for THIS frame's GPU work before continuing -- prevents any
    // submission backlog that could hang the GPU/whole system.
    g_frameFenceVal[i] = ++g_fenceVal;
    g_queue->Signal(g_fence, g_fenceVal);
    if (g_fence->GetCompletedValue() < g_fenceVal) {
        g_fence->SetEventOnCompletion(g_fenceVal, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
    g_frameIndex = g_swap->GetCurrentBackBufferIndex();

    // This frame's GPU work is done: free the texture staging buffers.
    for (ID3D12Resource* u : g_frameUploads) if (u) u->Release();
    g_frameUploads.clear();

    LARGE_INTEGER tFrame1; QueryPerformanceCounter(&tFrame1);
    g_frameMs += qpc_ms(tFrame0, tFrame1);
    if (++g_perfN >= 60) {
        static bool s_perf = (getenv("WOLF_PERF") != nullptr);
        double fMs = g_frameMs / g_perfN, cMs = g_cpuMs / g_perfN;
        if (s_perf)
            printf("[perf] %.1f fps  frame=%.2fms  cpu(inst+sbt)=%.2fms  gpu~=%.2fms  world=%zu inst=%zu\n",
                   1000.0 / fMs, fMs, cMs, fMs - cMs, g_world.size(), inst.size());
        g_frameMs = g_cpuMs = 0; g_perfN = 0;
    }
}

void gfx_shutdown()
{
    if (g_dev) wait_gpu();
    dlss_shutdown();
    SAFE_RELEASE(g_sharcResolvePSO);
    SAFE_RELEASE(g_sharcCB); SAFE_RELEASE(g_sharcHash); SAFE_RELEASE(g_sharcAccum); SAFE_RELEASE(g_sharcResolved);
    SAFE_RELEASE(g_restirCB);
    for (UINT ri = 0; ri < 2; ++ri) { SAFE_RELEASE(g_restirReservoir[ri]); SAFE_RELEASE(g_restirSurface[ri]); }
    SAFE_RELEASE(g_fogPSO); SAFE_RELEASE(g_fogRS); SAFE_RELEASE(g_fogVolumeBuf);
    SAFE_RELEASE(g_tmPSO); SAFE_RELEASE(g_tmRS);
    SAFE_RELEASE(g_lightBuf);
    SAFE_RELEASE(g_screenSkyTex);
    SAFE_RELEASE(g_partSeqIB);
    SAFE_RELEASE(g_colorTex); SAFE_RELEASE(g_depthTex); SAFE_RELEASE(g_mvTex); SAFE_RELEASE(g_normTex);
    SAFE_RELEASE(g_albTex); SAFE_RELEASE(g_specTex); SAFE_RELEASE(g_roughTex); SAFE_RELEASE(g_fogTex); SAFE_RELEASE(g_ppTex); SAFE_RELEASE(g_outTex);
    SAFE_RELEASE(g_skyEnvTex); SAFE_RELEASE(g_clearHeap); SAFE_RELEASE(g_srvHeap);
    SAFE_RELEASE(g_skyCB); SAFE_RELEASE(g_sbt); SAFE_RELEASE(g_sbtHit);
    SAFE_RELEASE(g_blasScratch); SAFE_RELEASE(g_tlasScratch); SAFE_RELEASE(g_tlasResult); SAFE_RELEASE(g_instBuf);
    SAFE_RELEASE(g_rtSO); SAFE_RELEASE(g_localRS); SAFE_RELEASE(g_rtRS);
    if (g_fenceEvent) CloseHandle(g_fenceEvent);
    SAFE_RELEASE(g_fence);
    SAFE_RELEASE(g_cl);
    for (UINT i = 0; i < kFrames; ++i) { SAFE_RELEASE(g_alloc[i]); SAFE_RELEASE(g_back[i]); }
    SAFE_RELEASE(g_rtvHeap);
    SAFE_RELEASE(g_swap);
    SAFE_RELEASE(g_queue);
    SAFE_RELEASE(g_dev);
}
