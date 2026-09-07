// Shared memory contract between the 32-bit client (proxy d3d9.dll) and the
// 64-bit host (our DXR path tracer). POD only, fixed layout, no pointers.
#pragma once
#include <cstdint>

#define BRIDGE_SHM_NAME   "Local\\WolfRT_Bridge"
#define BRIDGE_MAGIC      0x57524233u   // 'WRB3'
#define BRIDGE_VERSION    20u           // BR20: world-space emissive + volumetric particle classes

// Per-draw flags (BridgeDraw.flags).
#define DRAW_FLAG_SKY      0x1u         // atmos sky-dome layer: shade as emissive sky texture
#define DRAW_FLAG_EMISSIVE 0x2u         // self-illuminated glow geometry: emit texId.rgb * intensity
#define DRAW_FLAG_EMISSIVE_ALPHA 0x4u   // modulate emission by ColorMap alpha (SRCALPHA + ONE)
#define DRAW_FLAG_PARTICLE 0x8u         // expanded particle billboard: pure emitter (no diffuse albedo)

// Particle billboard class (BridgePartRange.cls). The proxy expands the game's GPU
// particle VS on the CPU into world-space quads and tags each by blend mode.
#define PART_EMISSIVE       0u         // additive (dst=ONE): fire/sparks/glow -> emit
#define PART_VOLUMETRIC     1u         // src-alpha over blend: smoke/steam -> participating media
#define PART_EMISSIVE_ALPHA 2u         // premultiplied over blend: soft fire/glow -> emit

#define BRIDGE_MAX_MESHES 8192u
#define BRIDGE_MAX_TEX    8192u
#define BRIDGE_MAX_DRAWS  8192u
#define BRIDGE_MAX_LIGHTS 1024u
#define BRIDGE_MAX_PART_RANGES 4096u                // per-frame particle sub-draws (grouped by tex+class)
#define BRIDGE_SCREEN_SKY_BYTES (4u * 1024u * 1024u)
#define BRIDGE_PARTICLE_BYTES (32u * 1024u * 1024u) // per-frame expanded particle vertices (world space)
#define BRIDGE_ARENA_SIZE (320u * 1024u * 1024u)   // geometry + texture bytes, append-only

// Texture pixel format codes (D3D9 -> our code -> DXGI on the host).
enum { BTEX_BGRA8 = 0, BTEX_BC1 = 1, BTEX_BC2 = 2, BTEX_BC3 = 3, BTEX_RGBA16F = 4 };
enum { BLIGHT_WORLD_SPACE = 0, BLIGHT_VIEW_SPACE = 1 };

struct BridgeCamera {
    float    proj[16];
    float    viewOrigin[4];
    uint32_t valid;
    uint32_t _pad[3];
};

struct BridgeVB { uint32_t arenaOff, size, stride, _pad; };
struct BridgeIB { uint32_t arenaOff, size, indexStride, count; };
struct BridgeTex {
    uint32_t arenaOff, size, width, height;
    uint32_t fmt, rowPitch, rows, faces;  // mip-0 rowPitch/rows; faces is 1 (2D) or 6 (cubemap)
    uint32_t mipCount, _pad3[3];          // >1 means the arena holds the full mip chain (mip0 first)
};

struct BridgeDraw {
    uint32_t vbId, ibId;
    uint32_t startIndex, primCount, baseVertex, minIndex, numVertices, materialId;
    uint32_t texId, uvOffset;      // texId: 0xFFFFFFFF = none; uvOffset: byte offset of float2 UV
    uint32_t vbOffset, posOffset;  // SetStreamSource byte offset + POSITION0 offset within each vertex
    uint32_t flags;                // DRAW_FLAG_* bits (0 = ordinary opaque surface)
    uint32_t specTexId;            // SpecularMap texture id (0xFFFFFFFF = none) -> F0/gloss
    uint32_t normalTexId;          // NormalMap texture id (0xFFFFFFFF = none) -> detail normal
    uint32_t depthTexId;           // DepthMap texture id (0xFFFFFFFF = none) -> parallax height
    uint32_t _materialPad;
    float    specParams[4];        // $vSpecularPowerIntensityBiasScale (power, intensity, bias, scale)
    float    parallaxParams[4];    // $vParallaxScaleBias (scale, bias, ...)
    float    transform[12];        // local -> view (3x4 row-major)
};

// One deferred local light (point or spot) recovered from the light pass.
struct BridgeLight {
    float    origin[4];   // position (xyz), in the coordinate space below
    float    color[4];    // diffuse rgb (xyz), w = intensity scale
    float    invRadius;   // 1 / light radius (attenuation)
    uint32_t space;       // BLIGHT_*_SPACE
    uint32_t _pad[2];
};

// One alpha-blended sky cloud layer drawn over the opaque base dome. Sampled in
// the base dome's hit shader (all layers map the same sky hemisphere), composited
// "over" the base, scrolled/rotated by its own texture affine.
struct BridgeSkyLayer {
    uint32_t colorTexId;      // ColorMap texture id (0xFFFFFFFF = none)
    uint32_t alphaTexId;      // AlphaMap texture id (0xFFFFFFFF = use colorMap.a)
    float    xform[6];        // 2x3 texture affine: rows [x.xyz][y.xyz], uv' = M*(uv,1)
};

struct BridgeLighting {
    uint32_t sunValid;
    float    sunDir[4];       // world dir toward the sun (idTech Z-up)
    float    sunColor[4];     // rgb (xyz), w = intensity
    uint32_t skyTexId;        // env-map texture id (0xFFFFFFFF = none)
    uint32_t skyIsCube;       // 1 if the env map is a cubemap
    float    skyColorScale[4];
    uint32_t lightCount;
    uint32_t skyLayerCount;   // active cloud layers (0..2)
    BridgeSkyLayer skyLayers[2];
    float    skyUVMat[20];    // quadratic dome dir->UV: u=Mu.b, v=Mv.b, b=[1,x,y,z,xx,yy,zz,xy,yz,zx]; u[0..9] then v[0..9]
    uint32_t skyUVValid;      // 1 once fitted
    uint32_t _pad[3];
};

struct BridgeScreenSky {
    uint32_t width, height, rowPitch, size;
    uint32_t sequence;   // odd while client writes, even when a frame is complete
    uint32_t valid;
    uint32_t _pad[2];
};

// One expanded particle billboard vertex, in WORLD space. The proxy runs the
// game's particle VS on the CPU (interpreter) and writes these; the host builds
// one BLAS per frame from the vertex stream and ray-traces it.
struct BridgePartVert {
    float    pos[3];     // world-space position
    float    uv[2];      // filmstrip-resolved texture coordinate
    uint32_t color;      // modulated RGBA8 (0xAABBGGRR); a = coverage for volumetric
};

// A contiguous run of particle triangles (vertCount % 3 == 0) sharing one texture
// and blend class. Grouped so the host can shade each run with its own material.
struct BridgePartRange {
    uint32_t firstVert;  // start index into the per-frame vertex stream
    uint32_t vertCount;  // number of vertices (multiple of 3)
    uint32_t texId;      // ColorMap texture id (bridge tex table; 0xFFFFFFFF = none)
    uint32_t cls;        // PART_EMISSIVE, PART_VOLUMETRIC, or PART_EMISSIVE_ALPHA
};

// Per-frame expanded-particle geometry. Ranges live in their own fixed array; the
// vertex stream lives in the particle byte region. sequence gates a torn read.
struct BridgeParticles {
    uint32_t rangeCount; // committed sub-draws this frame (<= BRIDGE_MAX_PART_RANGES)
    uint32_t vertCount;  // committed vertices this frame
    uint32_t sequence;   // odd while the client writes, even when the frame is complete
    uint32_t valid;
    uint32_t _pad[4];
};

// Veil vision is identified from Wolf's dedicated fullscreen distortion shader
// (ScreenMap + two normal maps + mask + G-buffer depth).  The proxy publishes
// the state; the host recreates the effect after Ray Reconstruction.
struct BridgeVeil {
    uint32_t active;        // steady compositor was submitted this frame
    uint32_t transitioning; // TransitionWipe/NoiseMap pass was submitted
    uint32_t normal0TexId, normal1TexId;
    uint32_t maskTexId, _pad0;
    float distanceControl[4]; // PS c0: near/far/base blend
    float rampControl[4];     // PS c1: scale/bias/power
    float normalXform0[4];    // VS c4: translate.xy, scale.zw
    float normalXform1[4];    // VS c5
    float aspect;
    float _pad[3];
};

struct BridgeHeader {
    uint32_t     magic, version, frameIndex, pid;
    BridgeCamera camera;
    uint32_t     vbCount, ibCount, texCount, arenaUsed;
    uint32_t     readyList;
    uint32_t     drawCount[2];
    BridgeLighting lighting;
    BridgeScreenSky screenSky;
    BridgeParticles particles;
    BridgeVeil   veil;
    uint32_t     _pad2[3];
};

inline uint64_t bridge_off_vbs()   { return sizeof(BridgeHeader); }
inline uint64_t bridge_off_ibs()   { return bridge_off_vbs() + (uint64_t)BRIDGE_MAX_MESHES * sizeof(BridgeVB); }
inline uint64_t bridge_off_texs()  { return bridge_off_ibs() + (uint64_t)BRIDGE_MAX_MESHES * sizeof(BridgeIB); }
inline uint64_t bridge_off_draws()  { return bridge_off_texs() + (uint64_t)BRIDGE_MAX_TEX * sizeof(BridgeTex); }
inline uint64_t bridge_off_lights() { return bridge_off_draws() + 2ull * BRIDGE_MAX_DRAWS * sizeof(BridgeDraw); }
inline uint64_t bridge_off_screen_sky() { return bridge_off_lights() + (uint64_t)BRIDGE_MAX_LIGHTS * sizeof(BridgeLight); }
inline uint64_t bridge_off_part_ranges() { return bridge_off_screen_sky() + BRIDGE_SCREEN_SKY_BYTES; }
inline uint64_t bridge_off_part_verts()  { return bridge_off_part_ranges() + (uint64_t)BRIDGE_MAX_PART_RANGES * sizeof(BridgePartRange); }
inline uint64_t bridge_off_arena()  { return bridge_off_part_verts() + BRIDGE_PARTICLE_BYTES; }
inline uint64_t bridge_total_size() { return bridge_off_arena() + BRIDGE_ARENA_SIZE; }

inline BridgeVB*    bridge_vbs(void* b)              { return (BridgeVB*)((char*)b + bridge_off_vbs()); }
inline BridgeIB*    bridge_ibs(void* b)              { return (BridgeIB*)((char*)b + bridge_off_ibs()); }
inline BridgeTex*   bridge_texs(void* b)             { return (BridgeTex*)((char*)b + bridge_off_texs()); }
inline BridgeDraw*  bridge_draws(void* b, int which) { return (BridgeDraw*)((char*)b + bridge_off_draws()) + (uint64_t)which * BRIDGE_MAX_DRAWS; }
inline BridgeLight* bridge_lights(void* b)           { return (BridgeLight*)((char*)b + bridge_off_lights()); }
inline uint8_t*     bridge_screen_sky(void* b)       { return (uint8_t*)b + bridge_off_screen_sky(); }
inline BridgePartRange* bridge_part_ranges(void* b)  { return (BridgePartRange*)((char*)b + bridge_off_part_ranges()); }
inline BridgePartVert*  bridge_part_verts(void* b)   { return (BridgePartVert*)((char*)b + bridge_off_part_verts()); }
inline uint8_t*     bridge_arena(void* b)            { return (uint8_t*)b + bridge_off_arena(); }
