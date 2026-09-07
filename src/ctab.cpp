// CTAB (constant table) parser for D3D9 vertex shaders.
//
// Shader bytecode = a DWORD token stream. The constant table is stored in a
// comment token (opcode 0xFFFE) whose payload begins with the FourCC 'CTAB'.
// After the FourCC comes a D3DXSHADER_CONSTANTTABLE header; all offsets in the
// table are byte offsets from the start of that header. We walk the constant
// array, read each constant's name + register index, and record the matrices
// we care about. See [[wolf-camera-solved]] for why this is the camera source.

#include <windows.h>
#include <d3d9.h>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include "ctab.h"
#include "log.h"

#pragma pack(push, 1)
struct CTHeader {
    DWORD Size, Creator, Version, Constants, ConstantInfo, Flags, Target;
};
struct CInfo {
    DWORD Name;
    WORD  RegisterSet, RegisterIndex, RegisterCount, Reserved;
    DWORD TypeInfo, DefaultValue;
};
#pragma pack(pop)

static std::unordered_map<void*, ShaderConstMap> g_cache;

static void classify(const char* name, WORD reg, WORD /*cnt*/, ShaderConstMap& m)
{
    // Exclusive matrix classification, specific -> general (names are substrings
    // of each other: "ModelViewProjection" contains "ModelView", etc.).
    if      (strstr(name, "ModelViewProjection"))   m.mvp          = reg;
    else if (strstr(name, "ModelToWorldTransform")) m.modelToWorld = reg;
    else if (strstr(name, "ViewToWorldTransform"))  m.viewToWorld  = reg;
    else if (strstr(name, "ModelView"))             m.modelView    = reg;
    else if (strstr(name, "Projection"))            m.proj         = reg;
    // Vectors / arrays (independent of the matrix slot above).
    if      (strstr(name, "WorldSpaceViewOrigin"))  m.viewOrigin   = reg;
    if      (strstr(name, "SkinToViewTransforms"))  m.skin         = reg;
    // Lights: global (sun) + local (Wolf2009 exact names).
    if      (strstr(name, "GlobalLightDirection"))  m.sunDir       = reg;
    if      (strstr(name, "GlobalLightColor"))      m.sunColor     = reg;
    if      (strstr(name, "LightOrigin"))           m.lightOrigin  = reg;
    if      (strstr(name, "ViewLightPosition"))     m.viewLightPos = reg;
    if      (strstr(name, "LightDiffuseColor"))     m.lightColor   = reg;
    if      (strstr(name, "InverseLightRadius"))    m.invRadius    = reg;
    if      (strstr(name, "SkyColorScale"))         m.skyScale     = reg;
    if      (strstr(name, "EnvironmentMap"))        m.envMap       = reg;
    if      (strstr(name, "EnvRotation0"))          m.envRotation  = reg;
    // Material G-buffer pass: specular lobe inputs.
    if      (strstr(name, "SpecularMap"))           m.specMap      = reg;
    if      (strstr(name, "NormalMap") && !strstr(name, "Detail")) m.normalMap = reg;
    if      (!strcmp(name, "$DepthMap") || !strcmp(name, "DepthMap")) m.depthMap = reg;
    if      (strstr(name, "SpecularPowerIntensityBiasScale")) m.specParams = reg;
    if      (strstr(name, "ParallaxScaleBias"))     m.parallaxParams = reg;
    // Sky-dome (atmos) layer constants.
    if      (strstr(name, "TextureTransform"))      m.texXform     = reg;
    if      (strstr(name, "HPosBias"))              m.hposBias     = reg;
    if      (strstr(name, "ColorModulate"))         m.colorMod     = reg;
    if      (strstr(name, "ColorAdd"))              m.colorAdd     = reg;
    if      (!strcmp(name, "$ScreenMap") || !strcmp(name, "ScreenMap")) m.screenMap = reg;
    if      (!strcmp(name, "$NormalMap0") || !strcmp(name, "NormalMap0")) m.normalMap0 = reg;
    if      (!strcmp(name, "$NormalMap1") || !strcmp(name, "NormalMap1")) m.normalMap1 = reg;
    if      (!strcmp(name, "$MaskMap") || !strcmp(name, "MaskMap")) m.maskMap = reg;
    if      (!strcmp(name, "GBufferNormalDepth") || !strcmp(name, "$GBufferNormalDepth")) m.gbufferDepth = reg;
    if      (strstr(name, "DistortionDistanceWeightControl")) m.distortionDist = reg;
    if      (strstr(name, "DistortionRampControl")) m.distortionRamp = reg;
    if      (!strcmp(name, "TransitionWipe") || !strcmp(name, "$TransitionWipe")) m.transitionWipe = reg;
    if      (!strcmp(name, "NoiseMap") || !strcmp(name, "$NoiseMap")) m.noiseMap = reg;
    if      (strstr(name, "NormalMap0TranslateScale")) m.normalXform0 = reg;
    if      (strstr(name, "NormalMap1TranslateScale")) m.normalXform1 = reg;
    if      (strstr(name, "AspectRatioCorrection")) m.aspectCorrection = reg;
    if      (strstr(name, "BoundingSphereCenterRadiusSqrd")) m.particleBounds = reg;
    if      (strstr(name, "PlaybackStopCycleTime")) m.particlePlayback = reg;
    if      (strstr(name, "PivotPointFilmstripScale")) m.particleFilmstrip = reg;
    if      (strstr(name, "ParticleColorFadeMap")) m.particleFadeMap = reg;
    if (strstr(name, "ColorMap") && !strstr(name, "Ambient") && !strstr(name, "Hemisphere") && !strstr(name, "GBuffer"))
        m.colorMap = reg;   // $ColorMap = the sky texture (avoid ambient/deferred look-alikes)
}

static void parse_bytecode(const DWORD* code, size_t nDwords, ShaderConstMap& m)
{
    const DWORD kCTAB = MAKEFOURCC('C', 'T', 'A', 'B');
    for (size_t i = 1; i + 1 < nDwords; ++i) {
        if ((code[i] & 0x0000FFFF) != 0x0000FFFE) continue;   // not a comment
        DWORD size = (code[i] >> 16) & 0x7FFF;                 // payload DWORDs
        if (size < 2 || i + 1 + size > nDwords) continue;
        if (code[i + 1] != kCTAB) { i += size; continue; }

        const char* base = (const char*)&code[i + 2];          // header start
        const CTHeader* h = (const CTHeader*)base;
        const CInfo* ci = (const CInfo*)(base + h->ConstantInfo);
        for (DWORD k = 0; k < h->Constants; ++k) {
            const char* name = base + ci[k].Name;
            classify(name, ci[k].RegisterIndex, ci[k].RegisterCount, m);
        }
        return;
    }
}

const ShaderConstMap& ctab_get(IDirect3DVertexShader9* vs)
{
    auto it = g_cache.find(vs);
    if (it != g_cache.end()) return it->second;

    ShaderConstMap m;
    UINT bytes = 0;
    if (vs && SUCCEEDED(vs->GetFunction(nullptr, &bytes)) && bytes >= 8) {
        DWORD* code = (DWORD*)malloc(bytes);
        if (code && SUCCEEDED(vs->GetFunction(code, &bytes))) {
            parse_bytecode(code, bytes / 4, m);
            m.parsed = true;
        }
        free(code);
    }
    auto res = g_cache.emplace(vs, m);
    return res.first->second;
}

const ShaderConstMap& ctab_get_ps(IDirect3DPixelShader9* ps)
{
    auto it = g_cache.find(ps);
    if (it != g_cache.end()) return it->second;

    ShaderConstMap m;
    UINT bytes = 0;
    if (ps && SUCCEEDED(ps->GetFunction(nullptr, &bytes)) && bytes >= 8) {
        DWORD* code = (DWORD*)malloc(bytes);
        if (code && SUCCEEDED(ps->GetFunction(code, &bytes))) {
            parse_bytecode(code, bytes / 4, m);
            m.parsed = true;
        }
        free(code);
    }
    auto res = g_cache.emplace(ps, m);
    return res.first->second;
}

void ctab_log_all(const void* code, unsigned bytes, const char* tag)
{
    const DWORD* dw = (const DWORD*)code; size_t nDwords = bytes / 4;
    const DWORD kCTAB = MAKEFOURCC('C', 'T', 'A', 'B');
    static const char* kSet[] = { "bool", "int4", "float4", "sampler" };
    for (size_t i = 1; i + 1 < nDwords; ++i) {
        if ((dw[i] & 0x0000FFFF) != 0x0000FFFE) continue;
        DWORD size = (dw[i] >> 16) & 0x7FFF;
        if (size < 2 || i + 1 + size > nDwords) continue;
        if (dw[i + 1] != kCTAB) { i += size; continue; }
        const char* base = (const char*)&dw[i + 2];
        const CTHeader* h = (const CTHeader*)base;
        const CInfo* ci = (const CInfo*)(base + h->ConstantInfo);
        log_printf("    [ctab-all] %s : %u constants", tag, h->Constants);
        for (DWORD k = 0; k < h->Constants; ++k) {
            const char* name = base + ci[k].Name;
            WORD rs = ci[k].RegisterSet;
            log_printf("        %-40s set=%s idx=%u cnt=%u",
                       name, rs < 4 ? kSet[rs] : "?", ci[k].RegisterIndex, ci[k].RegisterCount);
        }
        return;
    }
    log_printf("    [ctab-all] %s : no CTAB", tag);
}
