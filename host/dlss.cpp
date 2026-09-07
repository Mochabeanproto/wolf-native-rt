// DLSS Ray Reconstruction wrapper. See dlss.h.
#include "dlss.h"
#include <cstdio>
#include <cstring>
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_dlssd.h"

static NVSDK_NGX_Parameter* g_params = nullptr;
static NVSDK_NGX_Handle*    g_feat   = nullptr;
static bool g_init  = false;
static bool g_avail = false;
static bool g_ready = false;
static int  g_rw = 0, g_rh = 0;
static ID3D12Device* g_dev = nullptr;

bool dlss_available() { return g_avail; }
bool dlss_ready()     { return g_ready; }

bool dlss_init(ID3D12Device* dev)
{
    // Arbitrary dev project id; engine "custom". Log/data dir = the exe's cwd (".").
    static const char* kProj = "b7c1e2d4-8a35-4f10-9c62-1f4a7e6d3b90";
    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(
        kProj, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", L".", dev);
    if (NVSDK_NGX_FAILED(r)) { printf("[dlss] NGX Init failed (0x%08x)\n", r); return false; }
    g_init = true; g_dev = dev;

    r = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_params);
    if (NVSDK_NGX_FAILED(r) || !g_params) { printf("[dlss] GetCapabilityParameters failed (0x%08x)\n", r); return false; }

    int avail = 0;
    NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &avail);
    if (!avail) {
        int initResult = 0;
        NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &initResult);
        printf("[dlss] Ray Reconstruction NOT available on this GPU/driver (initResult=0x%08x)\n", initResult);
        return false;
    }
    g_avail = true;
    printf("[dlss] Ray Reconstruction available\n");
    return true;
}

static NVSDK_NGX_PerfQuality_Value quality_value(int q)
{
    switch (q) {
        case 1:  return NVSDK_NGX_PerfQuality_Value_MaxQuality;       // Quality
        case 2:  return NVSDK_NGX_PerfQuality_Value_Balanced;         // Balanced
        case 3:  return NVSDK_NGX_PerfQuality_Value_MaxPerf;          // Performance
        case 4:  return NVSDK_NGX_PerfQuality_Value_UltraPerformance; // Ultra Performance
        default: return NVSDK_NGX_PerfQuality_Value_DLAA;             // 0 = DLAA (native)
    }
}

// Force Preset F on every quality tier so whichever we create gets it.
static void set_preset_F()
{
    const char* tiers[] = {
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
    };
    for (const char* t : tiers)
        NVSDK_NGX_Parameter_SetUI(g_params, t, NVSDK_NGX_RayReconstruction_Hint_Render_Preset_F);
}

bool dlss_query_render_res(int outW, int outH, int quality, int* renderW, int* renderH)
{
    if (!g_avail || !g_params) return false;
    unsigned int oW, oH, mxW, mxH, mnW, mnH; float sharp;
    NVSDK_NGX_Result r = NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        g_params, outW, outH, quality_value(quality), &oW, &oH, &mxW, &mxH, &mnW, &mnH, &sharp);
    if (NVSDK_NGX_FAILED(r) || oW == 0 || oH == 0) { printf("[dlss] GET_OPTIMAL_SETTINGS failed (0x%08x)\n", r); return false; }
    *renderW = (int)oW; *renderH = (int)oH;
    return true;
}

bool dlss_create(ID3D12GraphicsCommandList* cl, int renderW, int renderH, int outW, int outH, int quality)
{
    if (!g_avail || g_ready) return g_ready;
    g_rw = renderW; g_rh = renderH;
    set_preset_F();

    NVSDK_NGX_DLSSD_Create_Params cp = {};
    cp.InDenoiseMode   = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    cp.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;   // separate roughness texture
    cp.InUseHWDepth    = NVSDK_NGX_DLSS_Depth_Type_HW;             // projective z/w depth (matches viewToClip)
    cp.InWidth = renderW; cp.InHeight = renderH;
    cp.InTargetWidth = outW; cp.InTargetHeight = outH;            // upscale render -> output
    cp.InPerfQualityValue = quality_value(quality);
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR;  // linear HDR color

    NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSSD_EXT(cl, 1, 1, &g_feat, g_params, &cp);
    if (NVSDK_NGX_FAILED(r) || !g_feat) { printf("[dlss] CREATE_DLSSD failed (0x%08x)\n", r); g_avail = false; return false; }
    g_ready = true;
    printf("[dlss] RR created: render %dx%d -> output %dx%d (Preset F, q=%d)\n", renderW, renderH, outW, outH, quality);
    return true;
}

bool dlss_evaluate(ID3D12GraphicsCommandList4* cl, const DlssFrame& f)
{
    if (!g_ready) return false;
    NVSDK_NGX_D3D12_DLSSD_Eval_Params ep = {};
    ep.pInColor        = f.color;
    ep.pInOutput       = f.output;
    ep.pInDepth        = f.depth;
    ep.pInMotionVectors= f.motion;
    ep.pInNormals      = f.normal;
    ep.pInRoughness    = f.roughness;
    ep.pInDiffuseAlbedo  = f.albedo;
    ep.pInSpecularAlbedo = f.specAlbedo;
    ep.InJitterOffsetX = f.jitterX;
    ep.InJitterOffsetY = f.jitterY;
    ep.InReset         = f.reset;
    ep.InMVScaleX      = 1.0f;   // motion vectors already in pixels
    ep.InMVScaleY      = 1.0f;
    ep.InRenderSubrectDimensions.Width  = g_rw;
    ep.InRenderSubrectDimensions.Height = g_rh;
    ep.pInWorldToViewMatrix = f.worldToView;
    ep.pInViewToClipMatrix  = f.viewToClip;

    NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSSD_EXT(cl, g_feat, g_params, &ep);
    if (NVSDK_NGX_FAILED(r)) { printf("[dlss] EVALUATE_DLSSD failed (0x%08x)\n", r); return false; }
    return true;
}

void dlss_shutdown()
{
    if (g_feat)   { NVSDK_NGX_D3D12_ReleaseFeature(g_feat); g_feat = nullptr; }
    if (g_params) { NVSDK_NGX_D3D12_DestroyParameters(g_params); g_params = nullptr; }
    if (g_init)   { if (g_dev) NVSDK_NGX_D3D12_Shutdown1(g_dev); else NVSDK_NGX_D3D12_Shutdown1(nullptr); g_init = false; }
    g_ready = g_avail = false;
}
