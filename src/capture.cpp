// wolf-native-rt : M1a scene capture (metadata dump)
//
// We patch the two geometry-submitting slots on the device vtable:
//   DrawPrimitive        (vtbl[81])
//   DrawIndexedPrimitive (vtbl[82])
//
// When a capture is armed (Ctrl+F10 in-game), the NEXT frame's draws are logged.
// Rather than shadowing every SetStreamSource/SetTexture/etc., each draw hook
// simply queries the device for its current state via the REAL getters (those
// vtable slots are unpatched), so we read exactly what the draw will use:
//   GetStreamSource / GetIndices / GetTexture / GetVertexShader /
//   GetPixelShader / GetVertexShaderConstantF (c0..c3 = camera MVP, usually).
//
// M1b will Lock the bound VB/IB and write real geometry (positions * MVP) out.

#include <windows.h>
#include <d3d9.h>
#include <d3dcompiler.h>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "capture.h"
#include "ctab.h"
#include "bridge.h"
#include "bridge_client.h"
#include "vsinterp.h"
#include "log.h"

// vtable slot patch (local copy; vtables live in read-only .rdata)
static void* patch_vtable(void* obj, int index, void* detour)
{
    void** vtbl = *reinterpret_cast<void***>(obj);
    DWORD old;
    VirtualProtect(&vtbl[index], sizeof(void*), PAGE_READWRITE, &old);
    void* orig = vtbl[index];
    vtbl[index] = detour;
    VirtualProtect(&vtbl[index], sizeof(void*), old, &old);
    return orig;
}

static IDirect3DDevice9* g_dev       = nullptr;
static bool     g_armed     = false;  // capture requested, starts next frame
static bool     g_capturing = false;  // currently logging this frame's draws
static unsigned g_frame     = 0;
static unsigned g_draw      = 0;
static bool     g_hotkeyDown = false;
static bool     g_originLogged = false; // camera origin logged this frame yet?
static bool     g_v2wLogged  = false;   // view->world basis logged this frame yet?
static float    g_seenProj[8][2];       // distinct projections seen (xScale,yScale)
static int      g_nProj      = 0;
static int      g_nDerivedP  = 0;        // count of P=MVP*inv(MV) derivations logged
static unsigned g_skyDumpFrame = 0xFFFFFFFFu;
static IDirect3DSurface9* g_skyDownRT = nullptr;
static IDirect3DSurface9* g_skyDownCPU = nullptr;
static UINT g_skyDownW = 0, g_skyDownH = 0;
static D3DFORMAT g_skyDownFmt = D3DFMT_UNKNOWN;
static unsigned g_skyPublishTick = 0;
// Particle billboards are expanded on the CPU by re-running the game's own particle
// VS (see vsinterp) into ray-traceable world-space geometry. Programs are cached by
// D3D9 VS pointer.
static std::unordered_map<void*, VSProgram*> g_partProgs;

// --- always-on live camera for the bridge (independent of Ctrl+F10 capture) ---
static bool     g_liveProj   = false;    // world projection grabbed this frame?
static bool     g_liveOrigin = false;    // camera origin grabbed this frame?
static float    g_curProj[16];
static bool     g_projEver = false;       // projection recovered at least once (persists)
static float    g_curOrigin[3] = {0, 0, 0};
static unsigned g_frameDraws = 0;        // total draws this frame (all, not just captured)

// Log at most this many draw records per captured frame (safety on log size).
static const unsigned kMaxDrawLog = 4000;
// Deep-dump (buffer descs + read-back) only the first few draws.
static const unsigned kDeepDump   = 8;

// Read n float4 registers (n*4 floats) from the VS constant file.
static void read_vsc(IDirect3DDevice9* dev, int reg, int n, float* out)
{
    for (int i = 0; i < n * 4; ++i) out[i] = 0.0f;
    if (reg >= 0) dev->GetVertexShaderConstantF(reg, out, n);
}
static void read_psc(IDirect3DDevice9* dev, int reg, int n, float* out);   // defined below

// C = A*B, row-major 4x4 (clip = M*local with M rows = the shader registers).
static void mat_mul(const float* A, const float* B, float* C)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            C[r * 4 + c] = A[r*4+0]*B[0*4+c] + A[r*4+1]*B[1*4+c] +
                           A[r*4+2]*B[2*4+c] + A[r*4+3]*B[3*4+c];
}

// General 4x4 inverse (MESA gluInvertMatrix). Returns false if singular.
static bool mat_inv(const float* m, float* invOut)
{
    float inv[16], det;
    inv[0]  =  m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
    det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
    if (det == 0.0f) return false;
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) invOut[i] = inv[i] * det;
    return true;
}

static void log_mat4(const char* tag, const float* c)  // 4 rows of 4
{
    log_printf("        %s r0=[% .4f % .4f % .4f % .4f] r1=[% .4f % .4f % .4f % .4f]",
               tag, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    log_printf("        %s r2=[% .4f % .4f % .4f % .4f] r3=[% .4f % .4f % .4f % .4f]",
               tag, c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]);
}

// Always-on: opportunistically grab the frame's camera (world projection via
// P = MVP*inv(MV) from a world draw; origin from any $vWorldSpaceViewOrigin)
// and publish it to the bridge. Cheap -- stops probing once both are found.
static void capture_live_camera(IDirect3DDevice9* dev, IDirect3DVertexShader9* vs)
{
    const ShaderConstMap& m = ctab_get(vs);
    if (!g_liveProj && m.mvp >= 0 && m.modelView >= 0) {
        float mvp[16], mv[16] = {0}, inv[16];
        read_vsc(dev, m.mvp, 4, mvp);
        read_vsc(dev, m.modelView, 3, mv);
        mv[12] = 0; mv[13] = 0; mv[14] = 0; mv[15] = 1;
        if (mat_inv(mv, inv)) { mat_mul(mvp, inv, g_curProj); g_liveProj = true; g_projEver = true; }
    }
    if (!g_liveOrigin && m.viewOrigin >= 0) {
        float o[4]; read_vsc(dev, m.viewOrigin, 1, o);
        g_curOrigin[0] = o[0]; g_curOrigin[1] = o[1]; g_curOrigin[2] = o[2];
        g_liveOrigin = true;
    }
    if (g_liveProj) bridge_set_camera(g_curProj, g_curOrigin);
}

// Always-on: stream one world draw (geometry + local->view transform) to the
// bridge. Only draws whose shader exposes $mModelView (the world/opaque family)
// are streamed for now; skinned/UI come later.
// Resolve the fields consumed by DXR. D3D9 commonly declares an ordinary xyz
// position as either FLOAT3 or FLOAT4 (the latter carries an unused w).  Both are
// safe for the DXR R32G32B32 view.  Keep rejecting genuinely packed positions,
// but retain the old offset-zero path when a legacy/FVF declaration cannot be
// queried at all.
static bool find_vertex_layout(IDirect3DDevice9* self, uint32_t& posOff, uint32_t& uvOff)
{
    posOff = uvOff = 0xFFFFFFFFu;
    IDirect3DVertexDeclaration9* decl = nullptr;
    if (SUCCEEDED(self->GetVertexDeclaration(&decl)) && decl) {
        D3DVERTEXELEMENT9 el[MAXD3DDECLLENGTH + 1]; UINT num = 0;
        if (SUCCEEDED(decl->GetDeclaration(el, &num))) {
            for (UINT i = 0; i < num && el[i].Stream != 0xFF; ++i) {
                if (el[i].Stream != 0 || el[i].UsageIndex != 0) continue;
                if (el[i].Usage == D3DDECLUSAGE_POSITION || el[i].Usage == D3DDECLUSAGE_POSITIONT) {
                    if (el[i].Type == D3DDECLTYPE_FLOAT3 || el[i].Type == D3DDECLTYPE_FLOAT4)
                        posOff = el[i].Offset;
                } else if (el[i].Usage == D3DDECLUSAGE_TEXCOORD && el[i].Type == D3DDECLTYPE_FLOAT2) {
                    uvOff = el[i].Offset;
                }
            }
        }
        decl->Release();
    } else {
        posOff = 0; // legacy FVF: preserve the renderer's previously working convention
    }
    return posOff != 0xFFFFFFFFu;
}

static bool bridge_texture_format(D3DFORMAT in, uint32_t& out)
{
    switch (in) {
        case D3DFMT_DXT1:                         out = BTEX_BC1;     return true;
        case D3DFMT_DXT2: case D3DFMT_DXT3:       out = BTEX_BC2;     return true;
        case D3DFMT_DXT4: case D3DFMT_DXT5:       out = BTEX_BC3;     return true;
        case D3DFMT_A8R8G8B8: case D3DFMT_X8R8G8B8: out = BTEX_BGRA8; return true;
        case D3DFMT_A16B16G16R16F:                out = BTEX_RGBA16F; return true;
        default: return false;
    }
}

static uint32_t bridge_texture_row_bytes(uint32_t fmt, uint32_t width)
{
    if (fmt == BTEX_BC1) return ((width + 3) / 4) * 8;
    if (fmt == BTEX_BC2 || fmt == BTEX_BC3) return ((width + 3) / 4) * 16;
    if (fmt == BTEX_RGBA16F) return width * 8;
    return width * 4;
}

static uint32_t tex_rows(uint32_t fmt, uint32_t h) { return (fmt == BTEX_BGRA8 || fmt == BTEX_RGBA16F) ? h : (h + 3) / 4; }

// Register a 2D or cube texture into the bridge. Cube faces are packed in D3D's
// +X,-X,+Y,-Y,+Z,-Z order so the host can create a TextureCube SRV directly.
// wantMips (2D only): capture the full mip chain so the host can trilinear-filter
// it -- needed to anti-alias the sky texture at the zenith (u=azimuth spins there).
static uint32_t register_texture(IDirect3DBaseTexture9* base, bool wantMips = false)
{
    if (!base || (base->GetType() != D3DRTYPE_TEXTURE && base->GetType() != D3DRTYPE_CUBETEXTURE))
        return 0xFFFFFFFFu;
    uint32_t id = bridge_tex_lookup(base);
    if (id != 0xFFFFFFFFu) return id;

    bool isCube = base->GetType() == D3DRTYPE_CUBETEXTURE;
    IDirect3DTexture9*     t2d = isCube ? nullptr : (IDirect3DTexture9*)base;
    IDirect3DCubeTexture9* tcu = isCube ? (IDirect3DCubeTexture9*)base : nullptr;
    D3DSURFACE_DESC d;
    if (isCube) { if (FAILED(tcu->GetLevelDesc(0, &d))) return 0xFFFFFFFFu; }
    else        { if (FAILED(t2d->GetLevelDesc(0, &d))) return 0xFFFFFFFFu; }

    // A8 (the atmos cloud $AlphaMap) isn't a host format; expand to BGRA8 (alpha in all channels).
    bool a8 = (d.Format == D3DFMT_A8);
    uint32_t fmt;
    if (a8) fmt = BTEX_BGRA8;
    else if (!bridge_texture_format(d.Format, fmt)) return 0xFFFFFFFFu;

    uint32_t faces = isCube ? 6u : 1u;
    uint32_t levels = 1;
    if (wantMips && !isCube) { levels = t2d->GetLevelCount(); if (levels < 1) levels = 1; if (levels > 14) levels = 14;
        static int s_ml = 0; if (s_ml < 4) { s_ml++; log_printf("[miptex] %ux%u fmt=%d GetLevelCount=%u", d.Width, d.Height, d.Format, t2d->GetLevelCount()); } }

    // Total = sum over faces × mips.
    uint64_t total = 0;
    for (uint32_t f = 0; f < faces; ++f)
        for (uint32_t m = 0; m < levels; ++m) {
            uint32_t w = d.Width >> m; if (!w) w = 1; uint32_t h = d.Height >> m; if (!h) h = 1;
            total += (uint64_t)bridge_texture_row_bytes(fmt, w) * tex_rows(fmt, h);
        }
    if (total == 0 || total > 0xFFFFFFFFu) return 0xFFFFFFFFu;
    std::vector<uint8_t> pixels((size_t)total);

    uint64_t dstOff = 0;
    for (uint32_t face = 0; face < faces; ++face) {
        for (uint32_t m = 0; m < levels; ++m) {
            uint32_t w = d.Width >> m; if (!w) w = 1; uint32_t h = d.Height >> m; if (!h) h = 1;
            uint32_t rp = bridge_texture_row_bytes(fmt, w), rw = tex_rows(fmt, h);
            D3DLOCKED_RECT lr = {};
            HRESULT hr = isCube ? tcu->LockRect((D3DCUBEMAP_FACES)face, m, &lr, nullptr, D3DLOCK_READONLY)
                                : t2d->LockRect(m, &lr, nullptr, D3DLOCK_READONLY);
            if (FAILED(hr) || !lr.pBits) return 0xFFFFFFFFu;
            uint8_t* dst = pixels.data() + dstOff;
            if (a8) {
                for (uint32_t row = 0; row < rw; ++row) {
                    const uint8_t* s = (const uint8_t*)lr.pBits + (uint64_t)row * lr.Pitch;
                    uint8_t* dr = dst + (uint64_t)row * rp;
                    for (uint32_t x = 0; x < w; ++x) { uint8_t a = s[x]; dr[x*4+0]=a; dr[x*4+1]=a; dr[x*4+2]=a; dr[x*4+3]=a; }
                }
            } else {
                uint32_t copyBytes = (uint32_t)lr.Pitch < rp ? (uint32_t)lr.Pitch : rp;
                for (uint32_t row = 0; row < rw; ++row)
                    memcpy(dst + (uint64_t)row * rp, (const uint8_t*)lr.pBits + (uint64_t)row * lr.Pitch, copyBytes);
            }
            if (isCube) tcu->UnlockRect((D3DCUBEMAP_FACES)face, m); else t2d->UnlockRect(m);
            dstOff += (uint64_t)rp * rw;
        }
    }

    uint32_t rows0 = tex_rows(fmt, d.Height), rowPitch0 = bridge_texture_row_bytes(fmt, d.Width);
    return bridge_register_tex(base, pixels.data(), (uint32_t)total, d.Width, d.Height,
                               fmt, rowPitch0, rows0, faces, levels);
}

static void capture_stream_draw(IDirect3DDevice9* self, IDirect3DVertexShader9* vs,
                                INT base, UINT minIdx, UINT numV, UINT startIdx, UINT primCount)
{
    const ShaderConstMap& m = ctab_get(vs);
    // Skinned positions in the source VB are bind-pose data; without replaying
    // the bone palette they appear as a second, frozen model beside animation.
    if (m.skin >= 0) return;

    // Skip idTech4 stencil-shadow volumes and depth/stencil-only passes: they
    // carry geometry + a transform but write NO colour, so they must not be
    // ray-traced (otherwise they appear as big solid boxes over the scene).
    DWORD cw = 0xF;
    self->GetRenderState(D3DRS_COLORWRITEENABLE, &cw);
    if ((cw & 7) == 0) return;

    // Skip alpha-blended / additive draws (decals, glass, glows, fog): rendering
    // them as opaque flat geometry makes them big solid slabs. They return in BR3
    // with real textures + alpha. Leaves just the opaque world surfaces.
    // Additive glow geometry (self-illuminated: signs/screens/filaments/particle sprites)
    // is normally skipped. Capture it as EMISSIVE instead so it emits in the RT scene.
    // (Deferred light-accumulation passes are also additive but carry GBufferNormalDepth;
    // exclude those -- they're handled as real lights.)
    bool emissiveDraw = false;
    bool emissiveUsesAlpha = false;
    int emissiveColorSampler = 0;
    DWORD ab = 0;
    self->GetRenderState(D3DRS_ALPHABLENDENABLE, &ab);
    if (ab) {
        DWORD sb = 0, db = 0;
        self->GetRenderState(D3DRS_SRCBLEND, &sb);
        self->GetRenderState(D3DRS_DESTBLEND, &db);
        IDirect3DPixelShader9* eps = nullptr; self->GetPixelShader(&eps);
        if ((sb == D3DBLEND_ONE || sb == D3DBLEND_SRCALPHA) && db == D3DBLEND_ONE && eps) {
            const ShaderConstMap& epm = ctab_get_ps(eps);
            emissiveDraw = (epm.gbufferDepth < 0 && epm.colorMap >= 0);   // forward glow, not a light pass
            emissiveColorSampler = epm.colorMap;
            emissiveUsesAlpha = (sb == D3DBLEND_SRCALPHA);
        }
        if (eps) eps->Release();
        if (!emissiveDraw) return;   // other alpha (decals/glass/fog): still skipped
    }

    float xf[12];
    if (m.modelView >= 0) {
        read_vsc(self, m.modelView, 3, xf);   // 3x4 local->view (row-major)
    } else {
        // Some forward emissive materials expose only MVP. Recover their
        // local->view matrix using the projection already learned from opaque
        // geometry. Restrict this fallback to non-sky additive ColorMap draws.
        if (!emissiveDraw || m.mvp < 0 || m.texXform >= 0 || !g_projEver) return;
        float mvp[16], invP[16], mv4[16];
        read_vsc(self, m.mvp, 4, mvp);
        if (!mat_inv(g_curProj, invP)) return;
        mat_mul(invP, mvp, mv4);
        for (int i = 0; i < 12; ++i) xf[i] = mv4[i];
        if (g_capturing) log_printf("        [EMISSIVE] recovered modelView from MVP (ColorMap s%d)", emissiveColorSampler);
    }
    for (int i = 0; i < 12; ++i) { float f = xf[i]; if (f != f || f > 1e30f || f < -1e30f) return; }

    IDirect3DVertexBuffer9* vb = nullptr; UINT off = 0, stride = 0;
    self->GetStreamSource(0, &vb, &off, &stride);
    IDirect3DIndexBuffer9* ib = nullptr; self->GetIndices(&ib);
    if (!vb || !ib || stride == 0) { if (vb) vb->Release(); if (ib) ib->Release(); return; }

    uint32_t vbId = bridge_vb_lookup(vb);
    if (vbId == 0xFFFFFFFFu) {
        D3DVERTEXBUFFER_DESC vd; vb->GetDesc(&vd);
        void* p = nullptr;
        if (SUCCEEDED(vb->Lock(0, 0, &p, D3DLOCK_READONLY)) && p) {
            vbId = bridge_register_vb(vb, p, vd.Size, stride);
            vb->Unlock();
        }
    }
    uint32_t ibId = bridge_ib_lookup(ib);
    if (ibId == 0xFFFFFFFFu) {
        D3DINDEXBUFFER_DESC id; ib->GetDesc(&id);
        uint32_t istride = (id.Format == D3DFMT_INDEX16) ? 2u : 4u;
        void* p = nullptr;
        if (SUCCEEDED(ib->Lock(0, 0, &p, D3DLOCK_READONLY)) && p) {
            ibId = bridge_register_ib(ib, p, id.Size, istride, id.Size / istride);
            ib->Unlock();
        }
    }

    uint32_t matId = 0, texId = 0xFFFFFFFFu;
    uint32_t posOffset = 0xFFFFFFFFu, uvOffset = 0xFFFFFFFFu;
    if (!find_vertex_layout(self, posOffset, uvOffset) || posOffset + 12u > stride) {
        vb->Release(); ib->Release(); return;
    }
    IDirect3DBaseTexture9* tex = nullptr;
    self->GetTexture((DWORD)(emissiveDraw ? emissiveColorSampler : 0), &tex);
    if (tex) {
        matId = (uint32_t)(uintptr_t)tex;
        if (uvOffset != 0xFFFFFFFFu) texId = register_texture(tex);
        tex->Release();
    }

    // Material inputs from the G-buffer pass. Wolf exposes its parallax source as
    // a dedicated DepthMap plus $vParallaxScaleBias; preserve those exact inputs.
    uint32_t specTexId = 0xFFFFFFFFu, normalTexId = 0xFFFFFFFFu, depthTexId = 0xFFFFFFFFu;
    float specParams[4] = { 0, 0, 0, 0 };
    float parallaxParams[4] = { 0, 0, 0, 0 };
    IDirect3DPixelShader9* ps = nullptr; self->GetPixelShader(&ps);
    if (ps) {
        const ShaderConstMap& pm = ctab_get_ps(ps);
        if (pm.specMap >= 0 && uvOffset != 0xFFFFFFFFu) {
            IDirect3DBaseTexture9* st = nullptr; self->GetTexture((DWORD)pm.specMap, &st);
            if (st) { specTexId = register_texture(st); st->Release(); }
        }
        if (pm.normalMap >= 0 && uvOffset != 0xFFFFFFFFu) {
            IDirect3DBaseTexture9* nt = nullptr; self->GetTexture((DWORD)pm.normalMap, &nt);
            if (nt) { normalTexId = register_texture(nt); nt->Release(); }
        }
        if (pm.depthMap >= 0 && uvOffset != 0xFFFFFFFFu) {
            IDirect3DBaseTexture9* dt = nullptr; self->GetTexture((DWORD)pm.depthMap, &dt);
            if (dt) { depthTexId = register_texture(dt); dt->Release(); }
        }
        if (pm.specParams >= 0) { float c[4]; read_psc(self, pm.specParams, 1, c); memcpy(specParams, c, sizeof(c)); }
        if (pm.parallaxParams >= 0) { float c[4]; read_psc(self, pm.parallaxParams, 1, c); memcpy(parallaxParams, c, sizeof(c)); }
        ps->Release();
    }

    uint32_t drawFlags = emissiveDraw ? DRAW_FLAG_EMISSIVE : 0u;
    if (emissiveUsesAlpha) drawFlags |= DRAW_FLAG_EMISSIVE_ALPHA;
    bridge_add_draw(vbId, ibId, startIdx, primCount, (uint32_t)base, minIdx, numV, xf,
                    matId, texId, uvOffset, off, posOffset, drawFlags, specTexId, normalTexId,
                    depthTexId, specParams, parallaxParams);
    vb->Release();
    ib->Release();
}

// Recover + log the camera and per-object world matrix from VS constants,
// using the bound shader's CTAB layout. This is the M2 camera source.
static void log_camera(IDirect3DDevice9* dev, IDirect3DVertexShader9* vs, unsigned n)
{
    if (!vs) return;
    const ShaderConstMap& m = ctab_get(vs);

    if (n < kDeepDump)
        log_printf("        [ctab] mvp=%d proj=%d m2w=%d mview=%d v2w=%d origin=%d skin=%d",
                   m.mvp, m.proj, m.modelToWorld, m.modelView, m.viewToWorld, m.viewOrigin, m.skin);

    float buf[16];
    // World draws expose BOTH $mModelViewProjection and $mModelView. Since
    // MVP = P * MV, solve P = MVP * inverse(MV4) to get the true world
    // projection (convention-independent, no viewmodel contamination).
    if (m.mvp >= 0 && m.modelView >= 0 && g_nDerivedP < 3) {
        float mvp[16], mv[16] = {0};
        read_vsc(dev, m.mvp, 4, mvp);
        read_vsc(dev, m.modelView, 3, mv);   // rows 0..2 = c[mview..+2]
        mv[12] = 0; mv[13] = 0; mv[14] = 0; mv[15] = 1;  // affine 4th row
        float mvInv[16], P[16];
        if (mat_inv(mv, mvInv)) {
            mat_mul(mvp, mvInv, P);
            float asp = P[0] != 0.0f ? P[5] / P[0] : 0.0f;
            log_printf("        [CAMERA] derived P = MVP*inv(MV) (draw %u) xScale=%.4f yScale=%.4f aspect=%.4f",
                       n, P[0], P[5], asp);
            log_mat4("Pd", P);
            ++g_nDerivedP;
        }
    }
    // Log every DISTINCT projection (viewmodel vs world differ) with its aspect.
    if (m.proj >= 0 && g_nProj < 8) {
        read_vsc(dev, m.proj, 4, buf);
        bool seen = false;
        for (int i = 0; i < g_nProj; ++i)
            if (fabsf(g_seenProj[i][0] - buf[0]) < 1e-4f &&
                fabsf(g_seenProj[i][1] - buf[5]) < 1e-4f) seen = true;
        if (!seen) {
            g_seenProj[g_nProj][0] = buf[0];
            g_seenProj[g_nProj][1] = buf[5];
            ++g_nProj;
            float asp = buf[0] != 0.0f ? buf[5] / buf[0] : 0.0f;
            log_printf("        [CAMERA] $mProjection @c%d (draw %u, skin=%d mvp=%d) xScale=%.4f yScale=%.4f aspect=%.4f",
                       m.proj, n, m.skin, m.mvp, buf[0], buf[5], asp);
            log_mat4("PROJ", buf);
        }
    }
    // View->world basis: its translation column should equal the eye origin.
    if (!g_v2wLogged && m.viewToWorld >= 0) {
        read_vsc(dev, m.viewToWorld, 3, buf);  // float4x3 = 3 registers
        log_printf("        [CAMERA] $mViewToWorld @c%d (draw %u):", m.viewToWorld, n);
        log_printf("        V2W c+0=[% .3f % .3f % .3f % .3f] c+1=[% .3f % .3f % .3f % .3f] c+2=[% .3f % .3f % .3f % .3f]",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]);
        g_v2wLogged = true;
    }
    if (!g_originLogged && m.viewOrigin >= 0) {
        read_vsc(dev, m.viewOrigin, 1, buf);
        log_printf("        [CAMERA] $vWorldSpaceViewOrigin @c%d = (% .3f % .3f % .3f)",
                   m.viewOrigin, buf[0], buf[1], buf[2]);
        g_originLogged = true;
    }
    if (n < kDeepDump && m.modelToWorld >= 0) {
        read_vsc(dev, m.modelToWorld, 3, buf);  // float4x3 -> 3 rows
        log_printf("        $mModelToWorld @c%d r0=[% .3f % .3f % .3f % .3f]", m.modelToWorld, buf[0], buf[1], buf[2], buf[3]);
        log_printf("                            r1=[% .3f % .3f % .3f % .3f] r2=[% .3f % .3f % .3f % .3f]",
                   buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]);
    }
}

// Read n float4 registers from the PIXEL shader constant file.
static void read_psc(IDirect3DDevice9* dev, int reg, int n, float* out)
{
    for (int i = 0; i < n * 4; ++i) out[i] = 0.0f;
    if (reg >= 0) dev->GetPixelShaderConstantF(reg, out, n);
}

static bool dump_render_target_bmp(IDirect3DDevice9* dev, IDirect3DBaseTexture9* base, DWORD slot)
{
    if (!base || base->GetType() != D3DRTYPE_TEXTURE) return false;
    IDirect3DTexture9* tex = (IDirect3DTexture9*)base;
    D3DSURFACE_DESC d = {};
    if (FAILED(tex->GetLevelDesc(0, &d)) || !(d.Usage & D3DUSAGE_RENDERTARGET) ||
        (d.Format != D3DFMT_A8R8G8B8 && d.Format != D3DFMT_X8R8G8B8)) return false;
    IDirect3DSurface9 *src = nullptr, *cpu = nullptr;
    if (FAILED(tex->GetSurfaceLevel(0, &src)) || !src) return false;
    HRESULT hr = dev->CreateOffscreenPlainSurface(d.Width, d.Height, d.Format, D3DPOOL_SYSTEMMEM, &cpu, nullptr);
    if (SUCCEEDED(hr)) hr = dev->GetRenderTargetData(src, cpu);
    src->Release();
    if (FAILED(hr) || !cpu) { if (cpu) cpu->Release(); return false; }
    D3DLOCKED_RECT lr = {};
    if (FAILED(cpu->LockRect(&lr, nullptr, D3DLOCK_READONLY)) || !lr.pBits) { cpu->Release(); return false; }

    char name[64]; sprintf_s(name, "wolf_sky_s%lu.bmp", slot);
    HANDLE f = CreateFileA(name, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    bool ok = f != INVALID_HANDLE_VALUE;
    if (ok) {
        BITMAPFILEHEADER fh = {}; BITMAPINFOHEADER ih = {};
        uint32_t bytes = d.Width * d.Height * 4u;
        fh.bfType = 0x4D42; fh.bfOffBits = sizeof(fh) + sizeof(ih); fh.bfSize = fh.bfOffBits + bytes;
        ih.biSize = sizeof(ih); ih.biWidth = (LONG)d.Width; ih.biHeight = (LONG)d.Height;
        ih.biPlanes = 1; ih.biBitCount = 32; ih.biCompression = BI_RGB; ih.biSizeImage = bytes;
        DWORD wrote = 0; ok = WriteFile(f, &fh, sizeof(fh), &wrote, nullptr) && wrote == sizeof(fh);
        ok = ok && WriteFile(f, &ih, sizeof(ih), &wrote, nullptr) && wrote == sizeof(ih);
        for (int y = (int)d.Height - 1; ok && y >= 0; --y) {
            const uint8_t* row = (const uint8_t*)lr.pBits + (uint64_t)y * lr.Pitch;
            ok = WriteFile(f, row, d.Width * 4u, &wrote, nullptr) && wrote == d.Width * 4u;
        }
        CloseHandle(f);
    }
    cpu->UnlockRect(); cpu->Release();
    log_printf("        [SKY] RT readback s%lu -> %s (%s)", slot, name, ok ? "ok" : "failed");
    return ok;
}

static bool dump_texture_bmp(IDirect3DDevice9* dev, IDirect3DBaseTexture9* base, const char* name)
{
    if (!base || base->GetType() != D3DRTYPE_TEXTURE) return false;
    IDirect3DTexture9* tex = (IDirect3DTexture9*)base;
    D3DSURFACE_DESC d = {};
    if (FAILED(tex->GetLevelDesc(0, &d)) || !d.Width || !d.Height) return false;
    IDirect3DSurface9 *src = nullptr, *rt = nullptr, *cpu = nullptr;
    if (FAILED(tex->GetSurfaceLevel(0, &src)) || !src) return false;
    HRESULT hr = dev->CreateRenderTarget(d.Width, d.Height, D3DFMT_A8R8G8B8,
                                         D3DMULTISAMPLE_NONE, 0, FALSE, &rt, nullptr);
    if (SUCCEEDED(hr)) hr = dev->StretchRect(src, nullptr, rt, nullptr, D3DTEXF_NONE);
    if (SUCCEEDED(hr)) hr = dev->CreateOffscreenPlainSurface(d.Width, d.Height, D3DFMT_A8R8G8B8,
                                                              D3DPOOL_SYSTEMMEM, &cpu, nullptr);
    if (SUCCEEDED(hr)) hr = dev->GetRenderTargetData(rt, cpu);
    src->Release(); if (rt) rt->Release();
    if (FAILED(hr) || !cpu) { if (cpu) cpu->Release(); return false; }
    D3DLOCKED_RECT lr = {};
    if (FAILED(cpu->LockRect(&lr, nullptr, D3DLOCK_READONLY)) || !lr.pBits) { cpu->Release(); return false; }
    HANDLE f = CreateFileA(name, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    bool ok = f != INVALID_HANDLE_VALUE;
    if (ok) {
        BITMAPFILEHEADER fh = {}; BITMAPINFOHEADER ih = {};
        uint32_t bytes = d.Width * d.Height * 4u;
        fh.bfType=0x4D42; fh.bfOffBits=sizeof(fh)+sizeof(ih); fh.bfSize=fh.bfOffBits+bytes;
        ih.biSize=sizeof(ih); ih.biWidth=(LONG)d.Width; ih.biHeight=(LONG)d.Height;
        ih.biPlanes=1; ih.biBitCount=32; ih.biCompression=BI_RGB; ih.biSizeImage=bytes;
        DWORD wrote=0; ok=WriteFile(f,&fh,sizeof(fh),&wrote,nullptr) && wrote==sizeof(fh);
        ok=ok && WriteFile(f,&ih,sizeof(ih),&wrote,nullptr) && wrote==sizeof(ih);
        for (int y=(int)d.Height-1; ok && y>=0; --y) {
            const uint8_t* row=(const uint8_t*)lr.pBits+(uint64_t)y*lr.Pitch;
            ok=WriteFile(f,row,d.Width*4u,&wrote,nullptr) && wrote==d.Width*4u;
        }
        CloseHandle(f);
    }
    cpu->UnlockRect(); cpu->Release();
    log_printf("        [VEIL] texture -> %s (%ux%u fmt=%d, %s)", name, d.Width, d.Height, d.Format, ok?"ok":"failed");
    return ok;
}

template <class S> static void dump_shader_asm(S* sh, const char* name)
{
    if (!sh) return;
    UINT bytes=0; if (FAILED(sh->GetFunction(nullptr,&bytes)) || bytes<8) return;
    std::vector<uint8_t> code(bytes);
    if (FAILED(sh->GetFunction(code.data(),&bytes))) return;
    ID3DBlob* asmText=nullptr;
    if (FAILED(D3DDisassemble(code.data(), bytes, D3D_DISASM_ENABLE_INSTRUCTION_NUMBERING, nullptr, &asmText)) || !asmText) return;
    HANDLE f=CreateFileA(name,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if (f!=INVALID_HANDLE_VALUE) { DWORD wrote=0; WriteFile(f,asmText->GetBufferPointer(),(DWORD)asmText->GetBufferSize(),&wrote,nullptr); CloseHandle(f); }
    log_printf("        [SHADER] assembly -> %s", name);
    asmText->Release();
}

static bool is_particle_shader(IDirect3DVertexShader9* vs, IDirect3DPixelShader9* ps)
{
    if (!vs || !ps) return false;
    const ShaderConstMap& vm = ctab_get(vs);
    const ShaderConstMap& pm = ctab_get_ps(ps);
    return vm.particleBounds >= 0 && vm.particlePlayback >= 0 &&
           vm.particleFilmstrip >= 0 && pm.particleFadeMap >= 0;
}

// A captured particle frame dumps the exact GPU program, declaration and raw
// per-particle records. D3D9 has no stream output, so these are the inputs we
// need to reproduce Wolf's billboard expansion on the bridge/host side.
static void dump_particle_diagnostics(IDirect3DDevice9* dev,
                                      IDirect3DVertexShader9* vs,
                                      IDirect3DPixelShader9* ps,
                                      unsigned draw)
{
    static std::unordered_set<void*> dumpedVS, dumpedPS;
    if (!is_particle_shader(vs, ps)) return;

    log_printf("        [PARTICLE] draw=%u exact GPU particle shader", draw);
    if (dumpedVS.insert(vs).second) {
        char name[96]; wsprintfA(name, "wolf_particle_vs_%p.txt", vs);
        dump_shader_asm(vs, name);
    }
    if (dumpedPS.insert(ps).second) {
        char name[96]; wsprintfA(name, "wolf_particle_ps_%p.txt", ps);
        dump_shader_asm(ps, name);
    }

    IDirect3DVertexDeclaration9* decl = nullptr;
    if (SUCCEEDED(dev->GetVertexDeclaration(&decl)) && decl) {
        UINT count = 0;
        if (SUCCEEDED(decl->GetDeclaration(nullptr, &count)) && count) {
            std::vector<D3DVERTEXELEMENT9> e(count);
            if (SUCCEEDED(decl->GetDeclaration(e.data(), &count))) {
                for (UINT i = 0; i < count && e[i].Stream != 0xFF; ++i)
                    log_printf("        [PARTICLE] decl%u stream=%u off=%u type=%u method=%u usage=%u index=%u",
                               i, e[i].Stream, e[i].Offset, e[i].Type, e[i].Method,
                               e[i].Usage, e[i].UsageIndex);
            }
        }
        decl->Release();
    }

    IDirect3DVertexBuffer9* vb = nullptr; UINT off = 0, stride = 0;
    if (SUCCEEDED(dev->GetStreamSource(0, &vb, &off, &stride)) && vb && stride) {
        void* data = nullptr;
        if (SUCCEEDED(vb->Lock(0, 0, &data, D3DLOCK_READONLY)) && data) {
            const uint8_t* base = (const uint8_t*)data + off;
            for (unsigned v = 0; v < 8; ++v) {
                const uint32_t* u = (const uint32_t*)(base + (size_t)v * stride);
                const float* f = (const float*)u;
                const unsigned words = stride / 4 < 16 ? stride / 4 : 16;
                char line[768]; int n = wsprintfA(line, "        [PARTICLE] v%u", v);
                for (unsigned k = 0; k < words && n < 700; ++k)
                    n += wsprintfA(line + n, " %08X/%g", u[k], f[k]);
                log_printf("%s", line);
            }
            vb->Unlock();
        }
        vb->Release();
    }

    float vc[96] = {};
    dev->GetVertexShaderConstantF(0, vc, 24);
    for (unsigned r = 0; r < 24; ++r)
        log_printf("        [PARTICLE] vc%u=[%g %g %g %g]", r,
                   vc[r*4+0], vc[r*4+1], vc[r*4+2], vc[r*4+3]);
}

static void capture_screen_sky(IDirect3DDevice9* dev, IDirect3DPixelShader9* ps)
{
    if (!ps || ctab_get_ps(ps).skyScale < 0) return;
    static bool dbgOnce = false;
    if (!dbgOnce) { dbgOnce = true; log_printf("[skydbg] skyScale PS pass reached; probing texture stages");
        for (DWORD st = 0; st < 4; ++st) {
            IDirect3DBaseTexture9* t = nullptr;
            if (SUCCEEDED(dev->GetTexture(st, &t)) && t) {
                D3DRESOURCETYPE rt = t->GetType();
                if (rt == D3DRTYPE_TEXTURE) { D3DSURFACE_DESC dd={}; ((IDirect3DTexture9*)t)->GetLevelDesc(0,&dd);
                    log_printf("  stage %lu: 2D %ux%u fmt=%d usage=0x%lx", st, dd.Width, dd.Height, dd.Format, dd.Usage); }
                else log_printf("  stage %lu: type=%d", st, rt);
                t->Release();
            } else log_printf("  stage %lu: (none)", st);
        }
    }
    if ((g_skyPublishTick++ & 1u) != 0) return; // ~30 Hz; avoids a readback stall every frame

    IDirect3DBaseTexture9* base = nullptr;
    if (FAILED(dev->GetTexture(1, &base)) || !base || base->GetType() != D3DRTYPE_TEXTURE) {
        if (base) base->Release(); return;
    }
    IDirect3DTexture9* tex = (IDirect3DTexture9*)base;
    D3DSURFACE_DESC d = {};
    if (FAILED(tex->GetLevelDesc(0, &d)) || !(d.Usage & D3DUSAGE_RENDERTARGET) ||
        (d.Format != D3DFMT_A8R8G8B8 && d.Format != D3DFMT_X8R8G8B8)) { base->Release(); return; }
    UINT w = d.Width > 512u ? 512u : d.Width;
    UINT h = (UINT)(((uint64_t)d.Height * w + d.Width / 2u) / d.Width);
    if (!h) h = 1;
    if (!g_skyDownRT || w != g_skyDownW || h != g_skyDownH || d.Format != g_skyDownFmt) {
        if (g_skyDownRT) g_skyDownRT->Release();
        if (g_skyDownCPU) g_skyDownCPU->Release();
        g_skyDownRT = g_skyDownCPU = nullptr;
        if (FAILED(dev->CreateRenderTarget(w, h, d.Format, D3DMULTISAMPLE_NONE, 0, FALSE, &g_skyDownRT, nullptr)) ||
            FAILED(dev->CreateOffscreenPlainSurface(w, h, d.Format, D3DPOOL_SYSTEMMEM, &g_skyDownCPU, nullptr))) {
            if (g_skyDownRT) g_skyDownRT->Release();
            if (g_skyDownCPU) g_skyDownCPU->Release();
            g_skyDownRT = g_skyDownCPU = nullptr; base->Release(); return;
        }
        g_skyDownW = w; g_skyDownH = h; g_skyDownFmt = d.Format;
    }
    IDirect3DSurface9* src = nullptr;
    HRESULT hr = tex->GetSurfaceLevel(0, &src);
    HRESULT hrGL = hr;
    if (SUCCEEDED(hr)) hr = dev->StretchRect(src, nullptr, g_skyDownRT, nullptr, D3DTEXF_LINEAR);
    HRESULT hrSR = hr;
    if (SUCCEEDED(hr)) hr = dev->GetRenderTargetData(g_skyDownRT, g_skyDownCPU);
    static int dbg2 = 0; if (dbg2 < 3) { dbg2++; log_printf("[skydbg] getlevel=0x%08lx stretch=0x%08lx getrtdata=0x%08lx (%ux%u)", hrGL, hrSR, hr, w, h); }
    if (src) src->Release(); base->Release();
    if (FAILED(hr)) return;
    D3DLOCKED_RECT lr = {};
    if (FAILED(g_skyDownCPU->LockRect(&lr, nullptr, D3DLOCK_READONLY)) || !lr.pBits) return;
    std::vector<uint8_t> pixels((size_t)w * h * 4u);
    for (UINT y = 0; y < h; ++y)
        memcpy(pixels.data() + (uint64_t)y * w * 4u, (const uint8_t*)lr.pBits + (uint64_t)y * lr.Pitch, w * 4u);
    g_skyDownCPU->UnlockRect();
    bridge_publish_screen_sky(pixels.data(), w, h, w * 4u);
}

// Get a shader's bytecode and dump all its CTAB constant names (diagnostic).
template <class S> static void dump_shader_ctab(S* sh, const char* tag)
{
    if (!sh) { log_printf("    [ctab-all] %s : (none)", tag); return; }
    UINT bytes = 0;
    if (SUCCEEDED(sh->GetFunction(nullptr, &bytes)) && bytes >= 8) {
        void* code = malloc(bytes);
        if (code && SUCCEEDED(sh->GetFunction(code, &bytes))) ctab_log_all(code, bytes, tag);
        free(code);
    }
}

// Diagnostic: dump every DISTINCT shader's VS + PS constant names once (deferred
// lighting passes come after the geometry, so we can't limit to the first N draws).
// Wolf 2009 is deferred: geometry draws write a G-buffer (Diffuse/Normal/Specular);
// the light constants live in the separate light-accumulation draws.
static std::unordered_set<void*> g_seenShaders;
static void log_shader_constants(IDirect3DDevice9* self, IDirect3DVertexShader9* vs,
                                 IDirect3DPixelShader9* ps, unsigned n)
{
    DWORD ab = 0, sb = 0, db = 0; self->GetRenderState(D3DRS_ALPHABLENDENABLE, &ab);
    self->GetRenderState(D3DRS_SRCBLEND, &sb); self->GetRenderState(D3DRS_DESTBLEND, &db);
    bool novel = (vs && g_seenShaders.insert(vs).second) | (ps && g_seenShaders.insert(ps).second);
    if (!novel) return;
    log_printf("    --- shader combo (draw %u) blend=%lu src=%lu dst=%lu ---", n, ab, sb, db);
    dump_shader_ctab(vs, "VS");
    dump_shader_ctab(ps, "PS");

    // Print values for any light-ish constants this shader exposes.
    const ShaderConstMap& vm = vs ? ctab_get(vs) : ShaderConstMap{};
    const ShaderConstMap& pm = ps ? ctab_get_ps(ps) : ShaderConstMap{};
    float b[16];
    auto dump = [&](const char* nm, int vsReg, int psReg) {
        if (vsReg >= 0) { read_vsc(self, vsReg, 1, b); log_printf("        [LIGHT] %s VS c%d = [% .3f % .3f % .3f % .3f]", nm, vsReg, b[0], b[1], b[2], b[3]); }
        if (psReg >= 0) { read_psc(self, psReg, 1, b); log_printf("        [LIGHT] %s PS c%d = [% .3f % .3f % .3f % .3f]", nm, psReg, b[0], b[1], b[2], b[3]); }
    };
    dump("sunDir", vm.sunDir, pm.sunDir); dump("sunColor", vm.sunColor, pm.sunColor);
    dump("lightOrigin", vm.lightOrigin, pm.lightOrigin); dump("viewLightPos", vm.viewLightPos, pm.viewLightPos);
    dump("lightColor", vm.lightColor, pm.lightColor);
    dump("invRadius", vm.invRadius, pm.invRadius);
    if (pm.skyScale >= 0) {
        float sky[4]; read_psc(self, pm.skyScale, 1, sky);
        log_printf("        [SKY] scale PS c%d = [% .3f % .3f % .3f % .3f]",
                   pm.skyScale, sky[0], sky[1], sky[2], sky[3]);
        // The late sky-aware composite samples dedicated render targets. Record
        // every bound source descriptor so the next F10 frame identifies which
        // one is the actual sky target instead of a reflection cubemap.
        bool dumpThisPass = g_skyDumpFrame != g_frame;
        for (DWORD slot = 0; slot < 8; ++slot) {
            IDirect3DBaseTexture9* t = nullptr;
            if (FAILED(self->GetTexture(slot, &t)) || !t) continue;
            D3DSURFACE_DESC td = {}; HRESULT hr = E_FAIL;
            if (t->GetType() == D3DRTYPE_TEXTURE)
                hr = ((IDirect3DTexture9*)t)->GetLevelDesc(0, &td);
            else if (t->GetType() == D3DRTYPE_CUBETEXTURE)
                hr = ((IDirect3DCubeTexture9*)t)->GetLevelDesc(0, &td);
            if (SUCCEEDED(hr))
                log_printf("        [SKY] s%lu tex=%p type=%d %ux%u fmt=%d usage=0x%08lx pool=%d",
                           slot, t, t->GetType(), td.Width, td.Height, td.Format, td.Usage, td.Pool);
            if (dumpThisPass && (slot == 0 || slot == 1 || slot == 3)) dump_render_target_bmp(self, t, slot);
            t->Release();
        }
        if (dumpThisPass) g_skyDumpFrame = g_frame;
    }
}

// Wolf's steady Veil pass has a unique CTAB signature: a fullscreen refraction
// composite driven by two scrolling normal maps, a mask, screen color and depth.
// TransitionWipe appears only while crossing the boundary.  Detect names rather
// than shader pointers so this remains stable across launches and driver builds.
static void capture_veil(IDirect3DDevice9* dev, IDirect3DVertexShader9* vs, IDirect3DPixelShader9* ps)
{
    if (!ps) return;
    const ShaderConstMap& p = ctab_get_ps(ps);
    bool active = p.screenMap >= 0 && p.normalMap0 >= 0 && p.normalMap1 >= 0 &&
                  p.maskMap >= 0 && p.gbufferDepth >= 0 &&
                  p.distortionDist >= 0 && p.distortionRamp >= 0;
    bool transition = p.transitionWipe >= 0 && p.gbufferDepth >= 0;
    if (active) {
        float dc[4]={}, rc[4]={}, x0[4]={}, x1[4]={}, ar[4]={};
        read_psc(dev,p.distortionDist,1,dc); read_psc(dev,p.distortionRamp,1,rc);
        const ShaderConstMap& v = ctab_get(vs);
        if (vs && v.normalXform0 >= 0) dev->GetVertexShaderConstantF(v.normalXform0,x0,1);
        if (vs && v.normalXform1 >= 0) dev->GetVertexShaderConstantF(v.normalXform1,x1,1);
        if (vs && v.aspectCorrection >= 0) dev->GetVertexShaderConstantF(v.aspectCorrection,ar,1);
        uint32_t ids[3]={0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
        const int slots[3]={p.normalMap0,p.normalMap1,p.maskMap};
        for(int i=0;i<3;++i) { IDirect3DBaseTexture9* t=nullptr; if(slots[i]>=0 && SUCCEEDED(dev->GetTexture(slots[i],&t)) && t){ids[i]=register_texture(t);t->Release();} }
        bridge_mark_veil(true,transition,ids[0],ids[1],ids[2],dc,rc,x0,x1,ar[0]);
    } else if (transition) bridge_mark_veil(false,true);
    if (g_capturing && (active || transition)) {
        static uint32_t dumpedActiveFrame=0xFFFFFFFFu, dumpedTransitionFrame=0xFFFFFFFFu;
        uint32_t& dumpedFrame = active ? dumpedActiveFrame : dumpedTransitionFrame;
        if (dumpedFrame != g_frame) {
            dumpedFrame=g_frame;
            const char* stem=active ? "wolf_veil_active" : "wolf_veil_transition";
            char file[96]; sprintf_s(file,"%s_vs.asm",stem); dump_shader_asm(vs,file);
            sprintf_s(file,"%s_ps.asm",stem); dump_shader_asm(ps,file);
            float pc[28]={}, vc[32]={}; read_psc(dev,0,7,pc);
            if (vs) dev->GetVertexShaderConstantF(0,vc,8);
            for (int r=0;r<7;++r) log_printf("        [VEIL] PS c%d=[% .6f % .6f % .6f % .6f]",r,pc[r*4],pc[r*4+1],pc[r*4+2],pc[r*4+3]);
            for (int r=0;r<8;++r) log_printf("        [VEIL] VS c%d=[% .6f % .6f % .6f % .6f]",r,vc[r*4],vc[r*4+1],vc[r*4+2],vc[r*4+3]);
            for (DWORD s=0;s<5;++s) { IDirect3DBaseTexture9* t=nullptr; if(SUCCEEDED(dev->GetTexture(s,&t))&&t){ sprintf_s(file,"%s_s%lu.bmp",stem,s); dump_texture_bmp(dev,t,file); t->Release(); } }
        }
    }
}

// Always-on: recover the sun + local lights from the deferred lighting passes and
// publish them to the bridge (independent of the Ctrl+F10 capture).
static bool g_sunSet = false;
static bool g_skySet = false;
static void capture_lighting(IDirect3DDevice9* dev, IDirect3DVertexShader9* vs, IDirect3DPixelShader9* ps,
                             UINT primCount)
{
    const ShaderConstMap& vm = ctab_get(vs);
    const ShaderConstMap& pm = ps ? ctab_get_ps(ps) : ShaderConstMap{};
    // Sun: the global-light pass exposes direction + color in the vertex shader.
    if (!g_sunSet && vm.sunDir >= 0 && vm.sunColor >= 0) {
        float d[4], c[4];
        read_vsc(dev, vm.sunDir, 1, d);
        read_vsc(dev, vm.sunColor, 1, c);
        float len = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        if (len > 1e-3f) {
            d[0] /= len; d[1] /= len; d[2] /= len;
            bridge_set_sun(d, c, c[3] > 0.0f ? c[3] : 1.0f);
            g_sunSet = true;
        }
    }
    // Local lights: variants consistently expose diffuse color, but only some
    // publish $fInverseLightRadius. All point-light variants render a scaled
    // unit volume, so its model-view basis supplies the missing radius.
    if (ps && pm.lightColor >= 0) {
        DWORD blendEnable = 0, dstBlend = 0;
        dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &blendEnable);
        dev->GetRenderState(D3DRS_DESTBLEND, &dstBlend);
        // Genuine deferred lights are additive volume draws.  Inherited light
        // constants on ordinary geometry and fullscreen ambient/GI passes were
        // the main source of the previous white wash.
        bool additiveVolume = blendEnable && dstBlend == D3DBLEND_ONE && primCount > 2;
        float mv[12] = {};
        bool haveMV = vm.modelView >= 0;
        if (haveMV) read_vsc(dev, vm.modelView, 3, mv);
        float o[4] = {}, col[4] = {}, invRadius = 0.0f;
        uint32_t space = BLIGHT_VIEW_SPACE;
        bool haveOrigin = false;
        if (pm.viewLightPos >= 0) {
            read_psc(dev, pm.viewLightPos, 1, o);
            haveOrigin = true;
        } else if (vm.lightOrigin >= 0) {
            read_vsc(dev, vm.lightOrigin, 1, o);
            space = BLIGHT_WORLD_SPACE;
            haveOrigin = true;
        }
        read_psc(dev, pm.lightColor, 1, col);
        if (pm.invRadius >= 0) {
            float ir[4]; read_psc(dev, pm.invRadius, 1, ir);
            invRadius = fabsf(ir[0]);
        } else if (haveMV) {
            // Rotation preserves column lengths; their average is the uniform
            // scale applied to Wolf's unit cube/sphere light volume.
            float sx = sqrtf(mv[0]*mv[0] + mv[4]*mv[4] + mv[8]*mv[8]);
            float sy = sqrtf(mv[1]*mv[1] + mv[5]*mv[5] + mv[9]*mv[9]);
            float sz = sqrtf(mv[2]*mv[2] + mv[6]*mv[6] + mv[10]*mv[10]);
            float radius = (sx + sy + sz) / 3.0f;
            if (radius > 1e-4f && std::isfinite(radius)) invRadius = 1.0f / radius;
        }
        // Keep finite, localized light volumes.  Large neutral dim volumes are
        // Wolf's fake ambient/GI; SHARC replaces those and must not double-count them.
        float radius = invRadius > 1e-6f ? 1.0f / invRadius : 0.0f;
        float maxc = fmaxf(col[0], fmaxf(col[1], col[2]));
        float minc = fminf(col[0], fminf(col[1], col[2]));
        float saturation = maxc > 1e-5f ? (maxc - minc) / maxc : 0.0f;
        bool tooDim      = maxc < 0.08f;
        bool badVolume   = radius < 5.0f || radius > 900.0f;
        bool ambientFill = radius > 450.0f && saturation < 0.12f && maxc < 0.75f;
        bool accepted = additiveVolume && haveOrigin && invRadius > 0.0f && std::isfinite(invRadius) &&
                        !tooDim && !badVolume && !ambientFill;
        if (accepted)
            // RGB is already the live, animated diffuse radiance.  Alpha is not a
            // second intensity term (multiplying it again caused overexposure).
            bridge_add_light(o, col, 1.0f, invRadius, space);
        if (g_capturing)
            log_printf("        [LIGHT-CAP] %s prim=%u additive=%d origin=%d radius=%.1f rgb=[%.3f %.3f %.3f] sat=%.2f",
                       accepted ? "ACCEPT" : "reject", primCount, additiveVolume ? 1 : 0, haveOrigin ? 1 : 0,
                       radius, col[0], col[1], col[2], saturation);
    }

    // The F10 shader dump shows $vSkyColorScale belongs to the final postprocess,
    // not the sky asset. The actual environment path is identified by the
    // EnvironmentMap sampler plus the VS environment-rotation constants.
    if (!g_skySet && ps && pm.envMap >= 0 && vm.envRotation >= 0) {
        IDirect3DBaseTexture9* env = nullptr;
        if (SUCCEEDED(dev->GetTexture((DWORD)pm.envMap, &env)) && env) {
            uint32_t texId = register_texture(env);
            if (texId != 0xFFFFFFFFu)
            {
                const float scale[4] = { 1, 1, 1, 1 };
                bridge_set_sky(texId, env->GetType() == D3DRTYPE_CUBETEXTURE ? 1u : 0u, scale);
                log_printf("[bridge] captured persistent environment texture id=%u type=%s",
                           texId, env->GetType() == D3DRTYPE_CUBETEXTURE ? "cube" : "2D");
                g_skySet = true;
            }
            env->Release();
        }
    }
}

// M1b diagnostic: describe the bound VB/IB and try a read-back Lock. Tells us
// whether the game's geometry is CPU-readable (managed pool) or write-only /
// default pool (in which case M2 must shadow buffer writes instead).
static void dump_transform(IDirect3DDevice9* self, unsigned n, const char* which, D3DTRANSFORMSTATETYPE ts)
{
    D3DMATRIX m;
    if (SUCCEEDED(self->GetTransform(ts, &m))) {
        log_printf("    [xf%u] %s r0=[% .4f % .4f % .4f % .4f] r1=[% .4f % .4f % .4f % .4f]",
                   n, which, m._11, m._12, m._13, m._14, m._21, m._22, m._23, m._24);
        log_printf("    [xf%u] %s r2=[% .4f % .4f % .4f % .4f] r3=[% .4f % .4f % .4f % .4f]",
                   n, which, m._31, m._32, m._33, m._34, m._41, m._42, m._43, m._44);
    }
}

static void dump_geometry(IDirect3DDevice9* self, unsigned n)
{
    // Sample the fixed-function camera slots AT DRAW TIME (not frame begin) --
    // this is exactly where Remix reads worldToView / viewToProjection.
    dump_transform(self, n, "VIEW", D3DTS_VIEW);
    dump_transform(self, n, "PROJ", D3DTS_PROJECTION);
    dump_transform(self, n, "WORLD", D3DTS_WORLD);

    IDirect3DVertexBuffer9* vb = nullptr; UINT off = 0, stride = 0;
    self->GetStreamSource(0, &vb, &off, &stride);
    if (vb) {
        D3DVERTEXBUFFER_DESC vd; vb->GetDesc(&vd);
        log_printf("    [geo%u] VB size=%u usage=0x%08x pool=%d fvf=0x%08x off=%u stride=%u",
                   n, vd.Size, vd.Usage, vd.Pool, vd.FVF, off, stride);
        void* p = nullptr;
        HRESULT hr = vb->Lock(0, 0, &p, D3DLOCK_READONLY);
        if (SUCCEEDED(hr) && p) {
            const unsigned char* base = (const unsigned char*)p + off;
            for (int v = 0; v < 6; ++v) {
                const float* f = (const float*)(base + v * stride);
                log_printf("        v%d pos=(% .3f % .3f % .3f)", v, f[0], f[1], f[2]);
            }
            vb->Unlock();
        } else {
            log_printf("        VB Lock(READONLY) failed hr=0x%08x", hr);
        }
        vb->Release();
    }

    IDirect3DIndexBuffer9* ib = nullptr;
    self->GetIndices(&ib);
    if (ib) {
        D3DINDEXBUFFER_DESC id; ib->GetDesc(&id);
        log_printf("    [geo%u] IB size=%u usage=0x%08x pool=%d fmt=%d",
                   n, id.Size, id.Usage, id.Pool, id.Format);
        void* p = nullptr;
        HRESULT hr = ib->Lock(0, 0, &p, D3DLOCK_READONLY);
        if (SUCCEEDED(hr) && p) {
            if (id.Format == D3DFMT_INDEX16) {
                const unsigned short* q = (const unsigned short*)p;
                log_printf("        idx16: %u %u %u %u %u %u %u %u %u %u %u %u",
                           q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11]);
            } else {
                const unsigned* q = (const unsigned*)p;
                log_printf("        idx32: %u %u %u %u %u %u %u %u %u %u %u %u",
                           q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11]);
            }
            ib->Unlock();
        } else {
            log_printf("        IB Lock(READONLY) failed hr=0x%08x", hr);
        }
        ib->Release();
    }
}

static void log_draw_state(IDirect3DDevice9* self, const char* kind,
                           D3DPRIMITIVETYPE prim, UINT primCount,
                           INT baseVtx, UINT minIdx, UINT numVtx, UINT startIdx)
{
    unsigned n = g_draw++;
    if (n >= kMaxDrawLog) return;

    IDirect3DVertexBuffer9* vb = nullptr; UINT off = 0, stride = 0;
    self->GetStreamSource(0, &vb, &off, &stride);
    IDirect3DIndexBuffer9*  ib = nullptr; self->GetIndices(&ib);
    IDirect3DBaseTexture9*  tex = nullptr; self->GetTexture(0, &tex);
    IDirect3DVertexShader9* vs = nullptr; self->GetVertexShader(&vs);
    IDirect3DPixelShader9*  ps = nullptr; self->GetPixelShader(&ps);
    float c[16] = {0};
    self->GetVertexShaderConstantF(0, c, 4);   // c0..c3

    log_printf("  [d%04u] %s prim=%d nP=%u base=%d minI=%u nV=%u startI=%u vb=%p off=%u str=%u ib=%p tex0=%p vs=%p ps=%p",
               n, kind, prim, primCount, baseVtx, minIdx, numVtx, startIdx,
               vb, off, stride, ib, tex, vs, ps);
    log_printf("        c0=[% .3f % .3f % .3f % .3f] c1=[% .3f % .3f % .3f % .3f] c2=[% .3f % .3f % .3f % .3f] c3=[% .3f % .3f % .3f % .3f]",
               c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
               c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]);

    // Render-state fingerprint + whether this draw is streamed to the RT host.
    // Used to identify the spurious "cube" overlays (shadow/fog/decal passes).
    {
        DWORD cw = 0, se = 0, ab = 0, sb = 0, db = 0, zf = 0, cull = 0;
        self->GetRenderState(D3DRS_COLORWRITEENABLE, &cw);
        self->GetRenderState(D3DRS_STENCILENABLE, &se);
        self->GetRenderState(D3DRS_ALPHABLENDENABLE, &ab);
        self->GetRenderState(D3DRS_SRCBLEND, &sb);
        self->GetRenderState(D3DRS_DESTBLEND, &db);
        self->GetRenderState(D3DRS_ZFUNC, &zf);
        self->GetRenderState(D3DRS_CULLMODE, &cull);
        int mview = vs ? ctab_get(vs).modelView : -1;
        int streamed = (vs && mview >= 0 && (cw & 7) != 0) ? 1 : 0;
        log_printf("        rs: cw=0x%lx stencil=%lu blend=%lu src=%lu dst=%lu zfunc=%lu cull=%lu mview=%d STREAM=%d",
                   cw, se, ab, sb, db, zf, cull, mview, streamed);
    }

    log_camera(self, vs, n);
    log_shader_constants(self, vs, ps, n);   // dump VS+PS constant names + light values
    dump_particle_diagnostics(self, vs, ps, n);

    if (vb) vb->Release();
    if (ib) ib->Release();
    if (tex) tex->Release();
    if (vs) vs->Release();
    if (ps) ps->Release();

    if (n < kDeepDump) dump_geometry(self, n);
}

// Fit the dome's direction->UV mapping as a QUADRATIC in dir = normalize(model pos):
// basis b = [1, x, y, z, xx, yy, zz, xy, yz, zx] (10 terms), uv = Mu.b / Mv.b, least
// squares over all dome verts. A linear fit smears the (nonlinear, wrapping) azimuth;
// the quadratic follows the true mapping closely -> smooth AND accurate. The dome's own
// per-vertex UVs barycentric-interpolated across coarse apex triangles give a radial
// pinch at the zenith; evaluating this analytic fit per pixel removes it. outM = u[10] then v[10].
// The dome UV is a clean CYLINDRICAL mapping: u = az/(2*pi) + ou (linear in azimuth,
// wrapping), v = cubic(dir.z) (a smooth elevation curve). Computing this analytically
// per pixel is smooth through the zenith (no mesh-fan seam) AND accurate (matches the
// real mapping, no approximation streaks). Fit ou (circular mean) and the v cubic
// (least squares over the upper hemisphere). outM: [0]=ou, [1]=su(=1/2pi), [2..5]=v cubic.
static bool fit_sky_uv_matrix(const uint8_t* vb, uint32_t off, uint32_t stride,
                              uint32_t posOffset, uint32_t uvOffset, uint32_t minIdx, uint32_t numV, float outM[20])
{
    const double TWO_PI = 6.283185307179586, su = 1.0 / TWO_PI;
    double A[4][4] = {}, bv[4] = {}, sc = 0, ss = 0; uint32_t nv = 0;
    for (uint32_t i = 0; i < numV; ++i) {
        const uint8_t* vtx = vb + off + (size_t)(minIdx + i) * stride;
        const float* pos = (const float*)(vtx + posOffset);
        const float* uv  = (const float*)(vtx + uvOffset);
        float len = sqrtf(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
        if (len < 1e-3f) continue;
        double dx = pos[0]/len, dy = pos[1]/len, dz = pos[2]/len;
        if (dz < -0.15) continue;                       // skip the lower/ground cap (different UV island)
        double z = dz, f[4] = { 1.0, z, z*z, z*z*z }, v = uv[1];
        for (int r = 0; r < 4; ++r) { for (int c = 0; c < 4; ++c) A[r][c] += f[r]*f[c]; bv[r] += v*f[r]; }
        double az = atan2(dy, dx), d = (double)uv[0] - az*su;   // u offset for this vert
        sc += cos(TWO_PI*d); ss += sin(TWO_PI*d);               // circular mean of the offset
        ++nv;
    }
    if (nv < 12) return false;
    // Solve A * vc = bv (4x4 Gaussian elimination) for the v cubic.
    double M[4][5];
    for (int r = 0; r < 4; ++r) { for (int c = 0; c < 4; ++c) M[r][c] = A[r][c]; M[r][4] = bv[r]; }
    for (int col = 0; col < 4; ++col) {
        int piv = col; for (int r = col+1; r < 4; ++r) if (fabs(M[r][col]) > fabs(M[piv][col])) piv = r;
        if (fabs(M[piv][col]) < 1e-12) return false;
        if (piv != col) for (int c = 0; c < 5; ++c) { double t = M[col][c]; M[col][c] = M[piv][c]; M[piv][c] = t; }
        double dd = M[col][col]; for (int c = 0; c < 5; ++c) M[col][c] /= dd;
        for (int r = 0; r < 4; ++r) if (r != col) { double ff = M[r][col]; for (int c = 0; c < 5; ++c) M[r][c] -= ff * M[col][c]; }
    }
    double ou = atan2(ss, sc) / TWO_PI; if (ou < 0) ou += 1.0;
    outM[0] = (float)ou; outM[1] = (float)su;
    for (int r = 0; r < 4; ++r) outM[2+r] = (float)M[r][4];
    for (int i = 6; i < 20; ++i) outM[i] = 0.0f;
    return true;
}

// Diagnostic: identify the sky-dome (atmos) layer draw and log its texture +
// direction->UV transform + tint + geometry, so we can replicate the mapping.
// The atmos sky is drawn as real world-space geometry (a dome mesh) whose VS
// exposes $mModelViewProjection + $mTextureTransform (no $mModelView). We can't
// let capture_stream_draw handle it (that path requires $mModelView and rejects
// the sky), so capture it here and stream it as an EMISSIVE surface: the host
// puts it in the TLAS, so primary rays, reflections AND GI all see the real sky
// texture -- exactly why we want a dome, not a screen-space render target.
//
// Placement trick: we only have MVP (model->clip). The projection is already
// recovered as g_curProj, so modelView = proj^-1 * MVP puts the sky on the same
// local->view footing every other object uses, and the host's view->world lands
// it in the shared world space with no special case.
static void capture_sky_geometry(IDirect3DDevice9* dev, IDirect3DVertexShader9* vs, IDirect3DPixelShader9* ps,
                                 INT base, UINT minIdx, UINT numV, UINT startIdx, UINT primCount)
{
    if (!vs || !ps || !g_projEver) return;
    const ShaderConstMap& vm = ctab_get(vs);
    const ShaderConstMap& pm = ctab_get_ps(ps);
    if (vm.texXform < 0 || vm.mvp < 0 || pm.colorMap < 0) return;   // not the sky-dome layer

    float mvp[16], invP[16], mv4[16];
    read_vsc(dev, vm.mvp, 4, mvp);
    if (!mat_inv(g_curProj, invP)) return;
    mat_mul(invP, mvp, mv4);            // model->view = proj^-1 * (model->clip)
    float xf[12]; for (int i = 0; i < 12; ++i) xf[i] = mv4[i];   // rows 0..2 (3x4)
    for (int i = 0; i < 12; ++i) { float f = xf[i]; if (f != f || f > 1e30f || f < -1e30f) return; }

    // Alpha-blended sky draws are the scrolling CLOUD overlays; the opaque draw is
    // the base dome. We put only the base into the TLAS as geometry (so reflections
    // hit real sky) and pass the cloud layers as globals, composited "over" the base
    // in the dome's hit shader (all layers map the same hemisphere -> share base UV).
    DWORD abe = 0; dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &abe);
    if (abe) {
        float tx[8] = {}; read_vsc(dev, vm.texXform, 2, tx);
        float xform6[6] = { tx[0], tx[1], tx[2], tx[4], tx[5], tx[6] };   // 2x3 texture affine
        uint32_t colId = 0xFFFFFFFFu, alpId = 0xFFFFFFFFu;
        IDirect3DBaseTexture9* ct = nullptr; dev->GetTexture((DWORD)pm.colorMap, &ct);
        if (ct) { colId = register_texture(ct, true); ct->Release(); }
        IDirect3DBaseTexture9* at = nullptr; dev->GetTexture(1, &at);   // s1 = $AlphaMap
        if (at) { alpId = register_texture(at, true); at->Release(); }
        bridge_add_sky_layer(colId, alpId, xform6);
        return;
    }

    IDirect3DVertexBuffer9* vb = nullptr; UINT off = 0, stride = 0;
    dev->GetStreamSource(0, &vb, &off, &stride);
    IDirect3DIndexBuffer9* ib = nullptr; dev->GetIndices(&ib);
    if (!vb || !ib || stride == 0) { if (vb) vb->Release(); if (ib) ib->Release(); return; }

    uint32_t vbId = bridge_vb_lookup(vb);
    if (vbId == 0xFFFFFFFFu) {
        D3DVERTEXBUFFER_DESC vd; vb->GetDesc(&vd);
        void* p = nullptr;
        if (SUCCEEDED(vb->Lock(0, 0, &p, D3DLOCK_READONLY)) && p) { vbId = bridge_register_vb(vb, p, vd.Size, stride); vb->Unlock(); }
    }
    uint32_t ibId = bridge_ib_lookup(ib);
    if (ibId == 0xFFFFFFFFu) {
        D3DINDEXBUFFER_DESC id; ib->GetDesc(&id);
        uint32_t istride = (id.Format == D3DFMT_INDEX16) ? 2u : 4u;
        void* p = nullptr;
        if (SUCCEEDED(ib->Lock(0, 0, &p, D3DLOCK_READONLY)) && p) { ibId = bridge_register_ib(ib, p, id.Size, istride, id.Size / istride); ib->Unlock(); }
    }

    uint32_t posOffset = 0xFFFFFFFFu, uvOffset = 0xFFFFFFFFu;
    if (!find_vertex_layout(dev, posOffset, uvOffset) || posOffset + 12u > stride) { vb->Release(); ib->Release(); return; }

    // Fit the dome's smooth analytic direction->UV matrix once per map (dome geometry
    // is static). Published to the shader to sample the sky per pixel without the pinch.
    static uint32_t s_fitVb = 0xFFFFFFFFu;
    if (vbId != s_fitVb && uvOffset != 0xFFFFFFFFu) {
        void* p = nullptr;
        if (SUCCEEDED(vb->Lock(0, 0, &p, D3DLOCK_READONLY)) && p) {
            float M[20];
            if (fit_sky_uv_matrix((const uint8_t*)p, off, stride, posOffset, uvOffset, minIdx, numV, M)) {
                bridge_set_sky_uv_matrix(M); s_fitVb = vbId;
                log_printf("[skydome] cylindrical UV fit: ou=%.4f su=%.4f vcubic=[%.4f %.4f %.4f %.4f]",
                           M[0], M[1], M[2], M[3], M[4], M[5]);
            }
            vb->Unlock();
        }
    }

    uint32_t texId = 0xFFFFFFFFu, matId = 0;
    IDirect3DBaseTexture9* tex = nullptr; dev->GetTexture((DWORD)pm.colorMap, &tex);
    if (tex) { matId = (uint32_t)(uintptr_t)tex; if (uvOffset != 0xFFFFFFFFu) texId = register_texture(tex, true); tex->Release(); }

    bridge_add_draw(vbId, ibId, startIdx, primCount, (uint32_t)base, minIdx, numV,
                    xf, matId, texId, uvOffset, off, posOffset, DRAW_FLAG_SKY);
    vb->Release(); ib->Release();

    static bool logged = false;
    if (!logged) { logged = true;
        log_printf("[skydome] base dome captured: vb=%u ib=%u prim=%u tex=%u (cloud layers stream as globals)",
                   vbId, ibId, primCount, texId); }
}

// --- DrawIndexedPrimitive (vtbl index 82) ------------------------------------
typedef HRESULT(STDMETHODCALLTYPE* DIP_t)(IDirect3DDevice9*, D3DPRIMITIVETYPE, INT, UINT, UINT, UINT, UINT);
static DIP_t s_realDIP = nullptr;

static VSProgram* get_particle_program(IDirect3DVertexShader9* vs)
{
    auto it = g_partProgs.find(vs);
    if (it != g_partProgs.end()) return it->second;
    VSProgram* prog = nullptr;
    UINT n = 0;
    if (SUCCEEDED(vs->GetFunction(nullptr, &n)) && n >= 8) {
        std::vector<uint32_t> code(n / 4);
        if (SUCCEEDED(vs->GetFunction(code.data(), &n))) {
            prog = new VSProgram();
            if (!prog->load(code.data(), n)) { delete prog; prog = nullptr; }
        }
    }
    g_partProgs[vs] = prog;
    log_printf("[part] cached VS program %p -> %s (%d instrs)", vs, prog ? "ok" : "FAILED", prog ? prog->insCount : 0);
    return prog;
}

// Read one vertex attribute out of the raw VB per its declaration type.
static void read_vertex_attr(const uint8_t* vp, int off, int type, float out[4])
{
    out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f;
    if (off < 0) return;
    const float* f = (const float*)(vp + off);
    switch (type) {
        case D3DDECLTYPE_FLOAT1: out[0]=f[0]; break;
        case D3DDECLTYPE_FLOAT2: out[0]=f[0]; out[1]=f[1]; break;
        case D3DDECLTYPE_FLOAT3: out[0]=f[0]; out[1]=f[1]; out[2]=f[2]; break;
        case D3DDECLTYPE_FLOAT4: out[0]=f[0]; out[1]=f[1]; out[2]=f[2]; out[3]=f[3]; break;
        case D3DDECLTYPE_D3DCOLOR: { uint32_t cc=*(const uint32_t*)(vp+off);
            out[0]=((cc>>16)&0xff)/255.0f; out[1]=((cc>>8)&0xff)/255.0f; out[2]=(cc&0xff)/255.0f; out[3]=((cc>>24)&0xff)/255.0f; } break;
        default: break;
    }
}

// Emitters closer to the camera than this (game units) are the first-person weapon;
// its particles/muzzle-flash must not enter the RT world (there is no gun mesh there).
static const float PARTICLE_VIEWMODEL_DEPTH = 24.0f;

// Re-run the game's particle VS on the CPU to recover the expanded, animated
// billboards as world-space triangles, and stream them to the bridge for ray-tracing.
static void capture_particle_geometry(IDirect3DDevice9* dev, IDirect3DVertexShader9* vs,
                                      IDirect3DPixelShader9* ps, INT base, UINT minIdx,
                                      UINT numV, UINT startIdx, UINT primCount)
{
    if (!vs || !ps || !is_particle_shader(vs, ps) || primCount == 0) return;

    DWORD ab = 0, sb = 0, db = 0;
    dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &ab);
    if (!ab) return;
    dev->GetRenderState(D3DRS_SRCBLEND, &sb);
    dev->GetRenderState(D3DRS_DESTBLEND, &db);
    // Wolf uses three materially different particle blend families. In
    // particular, src=ONE/dst=INVSRCALPHA is premultiplied soft fire, not
    // smoke; dropping it was the reason some flames disappeared in RT.
    uint32_t cls;
    if (db == D3DBLEND_ONE)
        cls = PART_EMISSIVE;
    else if (db == D3DBLEND_INVSRCALPHA && sb == D3DBLEND_ONE)
        cls = PART_EMISSIVE_ALPHA;
    else if (db == D3DBLEND_INVSRCALPHA && sb == D3DBLEND_SRCALPHA)
        cls = PART_VOLUMETRIC;
    else
        return; // unsupported compositing mode; do not misclassify it as fog

    const ShaderConstMap& pm = ctab_get_ps(ps);
    if (pm.screenMap >= 0 && pm.colorMap < 0) return;   // refraction-only distortion, skip

    VSProgram* prog = get_particle_program(vs);
    if (!prog || !prog->valid) return;

    // c0-3 projection, c4-6 modelView, c7-9 worldView, c13 bounding sphere.
    static float c[256][4];
    memset(c, 0, sizeof(c));
    dev->GetVertexShaderConstantF(0, &c[0][0], 32);

    float proj[16], invProj[16], wv[16], invWV[16], mv[16];
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) proj[i*4+k] = c[i][k];
    for (int k = 0; k < 4; ++k) { wv[0*4+k]=c[7][k]; wv[1*4+k]=c[8][k]; wv[2*4+k]=c[9][k]; }
    wv[12]=wv[13]=wv[14]=0; wv[15]=1;
    for (int k = 0; k < 4; ++k) { mv[0*4+k]=c[4][k]; mv[1*4+k]=c[5][k]; mv[2*4+k]=c[6][k]; }
    mv[12]=mv[13]=mv[14]=0; mv[15]=1;
    if (!mat_inv(proj, invProj) || !mat_inv(wv, invWV)) return;

    float ec[4] = { c[13][0], c[13][1], c[13][2], 1.0f }, ev[4];
    for (int i = 0; i < 4; ++i) ev[i]=mv[i*4]*ec[0]+mv[i*4+1]*ec[1]+mv[i*4+2]*ec[2]+mv[i*4+3]*ec[3];
    float emitterDepth = sqrtf(ev[0]*ev[0]+ev[1]*ev[1]+ev[2]*ev[2]);
    if (g_capturing) log_printf("        [part] emitterDepth=%.1f cls=%u prim=%u", emitterDepth, cls, primCount);
    if (emitterDepth < PARTICLE_VIEWMODEL_DEPTH) return;   // first-person weapon: exclude

    // Bind VS input registers to declaration elements (by usage + usageIndex).
    struct Bind { int off; int type; } bind[16];
    for (int i = 0; i < 16; ++i) { bind[i].off = -1; bind[i].type = 0; }
    IDirect3DVertexDeclaration9* decl = nullptr;
    if (FAILED(dev->GetVertexDeclaration(&decl)) || !decl) return;
    D3DVERTEXELEMENT9 el[MAXD3DDECLLENGTH + 1]; UINT nEl = 0;
    HRESULT dhr = decl->GetDeclaration(el, &nEl); decl->Release();
    if (FAILED(dhr)) return;
    for (int ri = 0; ri < 16; ++ri) {
        if (!prog->in[ri].used) continue;
        for (UINT j = 0; j < nEl && el[j].Stream != 0xFF; ++j)
            if (el[j].Stream == 0 && el[j].Usage == prog->in[ri].usage && el[j].UsageIndex == prog->in[ri].usageIndex) {
                bind[ri].off = el[j].Offset; bind[ri].type = el[j].Type; break;
            }
    }

    IDirect3DVertexBuffer9* vb = nullptr; UINT vbOff = 0, stride = 0;
    dev->GetStreamSource(0, &vb, &vbOff, &stride);
    IDirect3DIndexBuffer9* ib = nullptr; dev->GetIndices(&ib);
    if (!vb || stride == 0) { if (vb) vb->Release(); if (ib) ib->Release(); return; }
    uint8_t* vbData = nullptr;
    if (FAILED(vb->Lock(0, 0, (void**)&vbData, D3DLOCK_READONLY)) || !vbData) {
        vb->Release(); if (ib) ib->Release(); return;
    }
    uint8_t* ibData = nullptr; bool ib16 = true;
    if (ib) {
        D3DINDEXBUFFER_DESC id; ib->GetDesc(&id); ib16 = (id.Format == D3DFMT_INDEX16);
        if (FAILED(ib->Lock(0, 0, (void**)&ibData, D3DLOCK_READONLY))) ibData = nullptr;
    }

    uint32_t texId = 0xFFFFFFFFu;
    IDirect3DBaseTexture9* tex = nullptr;
    if (SUCCEEDED(dev->GetTexture((DWORD)(pm.colorMap >= 0 ? pm.colorMap : 0), &tex)) && tex) {
        texId = register_texture(tex); tex->Release();
    }

    static std::vector<BridgePartVert> out;
    out.clear();
    uint32_t idxCount = primCount * 3;
    for (uint32_t t = 0; t < idxCount; ++t) {
        uint32_t vi = ibData ? (ib16 ? (uint32_t)((const uint16_t*)ibData)[startIdx + t]
                                     : ((const uint32_t*)ibData)[startIdx + t])
                             : (startIdx + t);
        vi = (uint32_t)((int)vi + base);
        const uint8_t* vp = vbData + vbOff + (size_t)vi * stride;
        float v[16][4]; memset(v, 0, sizeof(v));
        for (int ri = 0; ri < 16; ++ri) if (prog->in[ri].used) read_vertex_attr(vp, bind[ri].off, bind[ri].type, v[ri]);
        float clip[4], uv[4], col[4];
        vs_run(*prog, v, c, clip, uv, col);
        if (clip[3] == 0.0f || clip[3] != clip[3]) continue;
        float vh[4];
        for (int i = 0; i < 4; ++i) vh[i]=invProj[i*4]*clip[0]+invProj[i*4+1]*clip[1]+invProj[i*4+2]*clip[2]+invProj[i*4+3]*clip[3];
        if (vh[3] == 0.0f) continue;
        float iw = 1.0f/vh[3];
        float vin[4] = { vh[0]*iw, vh[1]*iw, vh[2]*iw, 1.0f }, wh[4];
        for (int i = 0; i < 4; ++i) wh[i]=invWV[i*4]*vin[0]+invWV[i*4+1]*vin[1]+invWV[i*4+2]*vin[2]+invWV[i*4+3]*vin[3];
        float ww = wh[3] != 0.0f ? 1.0f/wh[3] : 1.0f;
        BridgePartVert pv;
        pv.pos[0]=wh[0]*ww; pv.pos[1]=wh[1]*ww; pv.pos[2]=wh[2]*ww;
        pv.uv[0]=uv[0]; pv.uv[1]=uv[1];
        auto b8=[](float x){ int q=(int)(x*255.0f+0.5f); return (uint32_t)(q<0?0:q>255?255:q); };
        pv.color = b8(col[0]) | (b8(col[1])<<8) | (b8(col[2])<<16) | (b8(col[3])<<24);
        bool bad=false; for(int k=0;k<3;++k) if (pv.pos[k]!=pv.pos[k] || fabsf(pv.pos[k])>1e8f) bad=true;
        if (!bad) out.push_back(pv);
    }
    vb->Unlock(); vb->Release();
    if (ib) { if (ibData) ib->Unlock(); ib->Release(); }

    if (out.size() >= 3)
        bridge_particles_add_range(out.data(), (uint32_t)out.size(), texId, cls);
}


static HRESULT STDMETHODCALLTYPE Hook_DIP(IDirect3DDevice9* self, D3DPRIMITIVETYPE prim,
                                          INT base, UINT minIdx, UINT numV, UINT startIdx, UINT primCount)
{
    ++g_frameDraws;
    if (bridge_active() && prim == D3DPT_TRIANGLELIST) {
        IDirect3DVertexShader9* vs = nullptr; self->GetVertexShader(&vs);
        IDirect3DPixelShader9* ps = nullptr; self->GetPixelShader(&ps);
        capture_veil(self, vs, ps);
        capture_screen_sky(self, ps);   // sky/postprocess pass may be DIP, not DP
        capture_sky_geometry(self, vs, ps, base, minIdx, numV, startIdx, primCount);  // emissive sky dome -> TLAS
        capture_particle_geometry(self, vs, ps, base, minIdx, numV, startIdx, primCount);  // particles -> ray-traced quads
        if (vs) {
            if (!g_liveProj || !g_liveOrigin) capture_live_camera(self, vs);
            capture_stream_draw(self, vs, base, minIdx, numV, startIdx, primCount);
            capture_lighting(self, vs, ps, primCount);
            vs->Release();
        }
        if (ps) ps->Release();
    }
    if (g_capturing) log_draw_state(self, "DIP", prim, primCount, base, minIdx, numV, startIdx);
    return s_realDIP(self, prim, base, minIdx, numV, startIdx, primCount);
}

// --- DrawPrimitive (vtbl index 81) -------------------------------------------
typedef HRESULT(STDMETHODCALLTYPE* DP_t)(IDirect3DDevice9*, D3DPRIMITIVETYPE, UINT, UINT);
static DP_t s_realDP = nullptr;
static HRESULT STDMETHODCALLTYPE Hook_DP(IDirect3DDevice9* self, D3DPRIMITIVETYPE prim,
                                         UINT start, UINT primCount)
{
    ++g_frameDraws;
    if (bridge_active()) {
        IDirect3DVertexShader9* vs = nullptr; self->GetVertexShader(&vs);
        IDirect3DPixelShader9* ps = nullptr; self->GetPixelShader(&ps);
        capture_veil(self, vs, ps);
        capture_screen_sky(self, ps);
        if (vs) vs->Release();
        if (ps) ps->Release();
    }
    if (g_capturing) log_draw_state(self, "DP ", prim, primCount, 0, 0, 0, start);
    return s_realDP(self, prim, start, primCount);
}

// Dump the fixed-function VIEW / PROJECTION matrices. If the game sets these
// (some idTech4 paths do), we get the shared camera directly instead of having
// to decompose per-draw MVPs. Called at capture BEGIN.
static void dump_ffp_camera(const char* which, D3DTRANSFORMSTATETYPE ts)
{
    if (!g_dev) return;
    D3DMATRIX m;
    if (SUCCEEDED(g_dev->GetTransform(ts, &m))) {
        log_printf("[camera] %s row0=[% .4f % .4f % .4f % .4f] row1=[% .4f % .4f % .4f % .4f]",
                   which, m._11, m._12, m._13, m._14, m._21, m._22, m._23, m._24);
        log_printf("[camera] %s row2=[% .4f % .4f % .4f % .4f] row3=[% .4f % .4f % .4f % .4f]",
                   which, m._31, m._32, m._33, m._34, m._41, m._42, m._43, m._44);
    }
}

void capture_install(IDirect3DDevice9* dev)
{
    g_dev = dev;
    s_realDIP = (DIP_t)patch_vtable(dev, 82, (void*)Hook_DIP);
    s_realDP  = (DP_t) patch_vtable(dev, 81, (void*)Hook_DP);
    log_printf("[capture] draw hooks installed (DIP real=%p DP real=%p) -- Ctrl+F10 in-game dumps one frame",
               s_realDIP, s_realDP);
}

void capture_present_tick()
{
    // Ctrl+F10 rising edge arms a one-frame capture. Ctrl guards against any
    // stray game binding on F10 and the Windows menu-bar activation.
    bool down = (GetAsyncKeyState(VK_CONTROL) & 0x8000) &&
                (GetAsyncKeyState(VK_F10)     & 0x8000);
    if (down && !g_hotkeyDown) g_armed = true;
    g_hotkeyDown = down;

    // Close a capture that ran through the frame just presented.
    if (g_capturing) {
        log_printf("[capture] END frame %u : %u draws", g_frame, g_draw);
        g_capturing = false;
    }
    // Open a new capture for the frame about to be drawn.
    if (g_armed) {
        g_armed = false;
        g_capturing = true;
        g_draw = 0;
        g_originLogged = false;
        g_v2wLogged = false;
        g_nProj = 0;
        g_nDerivedP = 0;
        g_seenShaders.clear();
        log_printf("[capture] BEGIN frame %u", g_frame);
        dump_ffp_camera("VIEW", D3DTS_VIEW);
        dump_ffp_camera("PROJ", D3DTS_PROJECTION);
    }
    // Publish this frame to the bridge and reset per-frame probes.
    bridge_particles_commit();       // publish this frame's expanded particle geometry
    bridge_end_frame();
    bridge_begin_frame();
    bridge_particles_begin();        // start accumulating next frame's particles
    g_frameDraws = 0;
    g_liveProj = false;
    g_liveOrigin = false;
    g_sunSet = false;   // re-detect the sun each frame
    // Sky selection persists: environment materials are not guaranteed visible
    // in every frame, but their registered texture remains valid in the bridge.

    ++g_frame;
}
