#include <windows.h>
#include <cstring>
#include <unordered_map>
#include "bridge.h"
#include "bridge_client.h"
#include "log.h"

static HANDLE        g_map = nullptr;
static BridgeHeader* g_hdr = nullptr;
static void*         g_base = nullptr;
static int           g_writeList = 0;     // draw list currently being filled
static uint32_t      g_writeCount = 0;

static std::unordered_map<void*, uint32_t> g_vbIds;
static std::unordered_map<void*, uint32_t> g_ibIds;
static std::unordered_map<void*, uint32_t> g_texIds;

// Sky layers are staged, then committed to the shared header in bridge_end_frame
// (atomically with the frame). Writing them live + resetting the count in
// begin_frame raced the host's async read -> it always caught 0 (same failure the
// live lighting has). Staging publishes a stable count for the whole frame.
static BridgeSkyLayer g_stagedSky[2];
static uint32_t       g_stagedSkyCount = 0;
// Local lights are staged too (same reason as the sky) and committed atomically with
// the frame + camera in bridge_end_frame -- otherwise the host caught a drifting partial
// set at view-space positions converted with a mismatched camera, so they never deduped
// and piled up in its cache (washout).
static BridgeLight    g_stagedLights[BRIDGE_MAX_LIGHTS];
static uint32_t       g_stagedLightCount = 0;
static BridgeVeil     g_stagedVeil = {};

static uint32_t arena_alloc(uint32_t size)
{
    uint32_t off = (g_hdr->arenaUsed + 15u) & ~15u;   // 16-byte align
    if ((uint64_t)off + size > BRIDGE_ARENA_SIZE) return 0xFFFFFFFFu;
    g_hdr->arenaUsed = off + size;
    return off;
}

void bridge_client_init()
{
    if (g_hdr) return;
    uint64_t total = bridge_total_size();
    g_map = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                               (DWORD)(total >> 32), (DWORD)(total & 0xFFFFFFFF), BRIDGE_SHM_NAME);
    if (!g_map) { log_printf("[bridge] CreateFileMapping failed %lu", GetLastError()); return; }
    g_base = MapViewOfFile(g_map, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T)total);
    if (!g_base) { log_printf("[bridge] MapViewOfFile(%llu) failed %lu", total, GetLastError()); return; }
    g_hdr = (BridgeHeader*)g_base;

    memset(g_hdr, 0, sizeof(*g_hdr));
    g_hdr->version = BRIDGE_VERSION;
    g_hdr->pid     = GetCurrentProcessId();
    g_hdr->lighting.skyTexId = 0xFFFFFFFFu;
    g_hdr->screenSky.valid = 0;
    g_hdr->screenSky.sequence = 0;
    g_hdr->particles.valid = 0;
    g_hdr->particles.sequence = 0;
    g_hdr->veil.normal0TexId = g_hdr->veil.normal1TexId = g_hdr->veil.maskTexId = 0xFFFFFFFFu;
    for (int i = 0; i < 4; ++i) g_hdr->lighting.skyColorScale[i] = 1.0f;
    g_hdr->magic   = BRIDGE_MAGIC;   // last: host waits on this
    g_writeList = 0; g_writeCount = 0;
    log_printf("[bridge] shared memory ready '%s' (%llu bytes, arena=%u MB)",
               BRIDGE_SHM_NAME, total, BRIDGE_ARENA_SIZE >> 20);
}

bool bridge_active() { return g_hdr != nullptr; }

// Camera is STAGED here during the frame and published atomically with the draws
// in bridge_end_frame -- otherwise the host reads a live (newer) camera against a
// frame-old draw list, so the geometry lags/stutters behind the camera.
static float    g_camProj[16] = {};
static float    g_camOrigin[3] = {};
static uint32_t g_camValid = 0;

void bridge_set_camera(const float* proj16, const float* origin3)
{
    memcpy(g_camProj, proj16, 16 * sizeof(float));
    g_camOrigin[0] = origin3[0]; g_camOrigin[1] = origin3[1]; g_camOrigin[2] = origin3[2];
    g_camValid = 1;
}

uint32_t bridge_vb_lookup(void* key)
{
    auto it = g_vbIds.find(key);
    return it != g_vbIds.end() ? it->second : 0xFFFFFFFFu;
}
uint32_t bridge_ib_lookup(void* key)
{
    auto it = g_ibIds.find(key);
    return it != g_ibIds.end() ? it->second : 0xFFFFFFFFu;
}
uint32_t bridge_tex_lookup(void* key)
{
    auto it = g_texIds.find(key);
    return it != g_texIds.end() ? it->second : 0xFFFFFFFFu;
}
uint32_t bridge_register_tex(void* key, const void* data, uint32_t size,
                             uint32_t w, uint32_t h, uint32_t fmt, uint32_t rowPitch,
                             uint32_t rows, uint32_t faces, uint32_t mipCount)
{
    if (!g_hdr) return 0xFFFFFFFFu;
    auto it = g_texIds.find(key);
    if (it != g_texIds.end()) return it->second;
    if (g_hdr->texCount >= BRIDGE_MAX_TEX) return 0xFFFFFFFFu;
    uint32_t off = arena_alloc(size);
    if (off == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    memcpy(bridge_arena(g_base) + off, data, size);
    uint32_t id = g_hdr->texCount;
    BridgeTex& t = bridge_texs(g_base)[id];
    t.arenaOff = off; t.size = size; t.width = w; t.height = h;
    t.fmt = fmt; t.rowPitch = rowPitch; t.rows = rows; t.faces = faces;
    t.mipCount = mipCount < 1 ? 1 : mipCount; t._pad3[0] = t._pad3[1] = t._pad3[2] = 0;
    g_hdr->texCount = id + 1;
    g_texIds.emplace(key, id);
    // Registry keys are raw COM addresses. Keep the object alive for the life of
    // the bridge so D3D cannot recycle that address for unrelated texture data.
    ((IUnknown*)key)->AddRef();
    return id;
}

uint32_t bridge_register_vb(void* key, const void* data, uint32_t size, uint32_t stride)
{
    if (!g_hdr) return 0xFFFFFFFFu;
    auto it = g_vbIds.find(key);
    if (it != g_vbIds.end()) return it->second;
    if (g_hdr->vbCount >= BRIDGE_MAX_MESHES) return 0xFFFFFFFFu;
    uint32_t off = arena_alloc(size);
    if (off == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    memcpy(bridge_arena(g_base) + off, data, size);
    uint32_t id = g_hdr->vbCount;
    BridgeVB& v = bridge_vbs(g_base)[id];
    v.arenaOff = off; v.size = size; v.stride = stride; v._pad = 0;
    g_hdr->vbCount = id + 1;          // publish after the entry is filled
    g_vbIds.emplace(key, id);
    // Without this reference Wolf can release the VB and D3D may later allocate
    // a different resource at the same COM address, making our snapshot lookup
    // return unrelated vertex data.
    ((IUnknown*)key)->AddRef();
    return id;
}

uint32_t bridge_register_ib(void* key, const void* data, uint32_t size, uint32_t indexStride, uint32_t count)
{
    if (!g_hdr) return 0xFFFFFFFFu;
    auto it = g_ibIds.find(key);
    if (it != g_ibIds.end()) return it->second;
    if (g_hdr->ibCount >= BRIDGE_MAX_MESHES) return 0xFFFFFFFFu;
    uint32_t off = arena_alloc(size);
    if (off == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    memcpy(bridge_arena(g_base) + off, data, size);
    uint32_t id = g_hdr->ibCount;
    BridgeIB& b = bridge_ibs(g_base)[id];
    b.arenaOff = off; b.size = size; b.indexStride = indexStride; b.count = count;
    g_hdr->ibCount = id + 1;
    g_ibIds.emplace(key, id);
    ((IUnknown*)key)->AddRef();
    return id;
}

void bridge_begin_frame()
{
    if (!g_hdr) return;
    g_writeList  = g_hdr->readyList ^ 1;   // fill the list the host isn't reading
    g_writeCount = 0;
    g_hdr->lighting.sunValid  = 0;         // re-detect sun each frame
    g_stagedLightCount = 0;                 // re-collect local lights each frame (staged, committed at end_frame)
    g_stagedSkyCount = 0;                   // re-collect cloud layers each frame (staged, committed at end_frame)
    g_stagedVeil = {};
    g_stagedVeil.normal0TexId = g_stagedVeil.normal1TexId = g_stagedVeil.maskTexId = 0xFFFFFFFFu;
    // The environment material is visibility-dependent and may not be submitted
    // every frame. Keep the last valid sky instead of flashing back to fallback.
}

void bridge_mark_veil(bool activeComposite, bool transitionPass,
                      uint32_t normal0TexId, uint32_t normal1TexId, uint32_t maskTexId,
                      const float* distance4, const float* ramp4, const float* xform0,
                      const float* xform1, float aspect)
{
    if (activeComposite) g_stagedVeil.active = 1;
    if (transitionPass)  g_stagedVeil.transitioning = 1;
    if (activeComposite) {
        g_stagedVeil.normal0TexId=normal0TexId; g_stagedVeil.normal1TexId=normal1TexId;
        g_stagedVeil.maskTexId=maskTexId; g_stagedVeil.aspect=aspect;
        if(distance4) memcpy(g_stagedVeil.distanceControl,distance4,16);
        if(ramp4) memcpy(g_stagedVeil.rampControl,ramp4,16);
        if(xform0) memcpy(g_stagedVeil.normalXform0,xform0,16);
        if(xform1) memcpy(g_stagedVeil.normalXform1,xform1,16);
    }
}

void bridge_set_sun(const float* dir3, const float* color3, float intensity)
{
    if (!g_hdr) return;
    BridgeLighting& L = g_hdr->lighting;
    L.sunDir[0] = dir3[0]; L.sunDir[1] = dir3[1]; L.sunDir[2] = dir3[2]; L.sunDir[3] = 0;
    L.sunColor[0] = color3[0]; L.sunColor[1] = color3[1]; L.sunColor[2] = color3[2]; L.sunColor[3] = intensity;
    L.sunValid = 1;
}

void bridge_set_sky(uint32_t texId, uint32_t isCube, const float* colorScale4)
{
    if (!g_hdr) return;
    BridgeLighting& L = g_hdr->lighting;
    L.skyTexId = texId; L.skyIsCube = isCube;
    for (int i = 0; i < 4; ++i) L.skyColorScale[i] = colorScale4 ? colorScale4[i] : 1.0f;
}

void bridge_add_sky_layer(uint32_t colorTexId, uint32_t alphaTexId, const float* xform6)
{
    if (g_stagedSkyCount >= 2) return;
    BridgeSkyLayer& s = g_stagedSky[g_stagedSkyCount++];
    s.colorTexId = colorTexId; s.alphaTexId = alphaTexId;
    for (int i = 0; i < 6; ++i) s.xform[i] = xform6[i];
}

void bridge_set_sky_uv_matrix(const float* m20)
{
    if (!g_hdr || !m20) return;
    for (int i = 0; i < 20; ++i) g_hdr->lighting.skyUVMat[i] = m20[i];
    g_hdr->lighting.skyUVValid = 1;   // static per map; set once, never reset -> no publish race
}

void bridge_publish_screen_sky(const void* data, uint32_t width, uint32_t height, uint32_t rowPitch)
{
    if (!g_hdr || !data || !width || !height || rowPitch < width * 4u) return;
    uint64_t size = (uint64_t)rowPitch * height;
    if (size > BRIDGE_SCREEN_SKY_BYTES) return;
    uint32_t seq = g_hdr->screenSky.sequence;
    g_hdr->screenSky.sequence = seq + 1u;
    MemoryBarrier();
    memcpy(bridge_screen_sky(g_base), data, (size_t)size);
    g_hdr->screenSky.width = width; g_hdr->screenSky.height = height;
    g_hdr->screenSky.rowPitch = rowPitch; g_hdr->screenSky.size = (uint32_t)size;
    g_hdr->screenSky.valid = 1;
    MemoryBarrier();
    g_hdr->screenSky.sequence = seq + 2u;
}

// Particle geometry is accumulated into a CPU staging buffer during the frame, then
// committed atomically (sequence odd->even) so the host never reads a torn frame.
static const uint32_t PART_VERT_CAP = BRIDGE_PARTICLE_BYTES / (uint32_t)sizeof(BridgePartVert);
static BridgePartVert g_partVerts[PART_VERT_CAP];
static BridgePartRange g_partRanges[BRIDGE_MAX_PART_RANGES];
static uint32_t g_partVertCount = 0, g_partRangeCount = 0;

void bridge_particles_begin()
{
    g_partVertCount = 0; g_partRangeCount = 0;
}

void bridge_particles_add_range(const BridgePartVert* verts, uint32_t vertCount,
                                uint32_t texId, uint32_t cls)
{
    if (!verts || vertCount < 3) return;
    vertCount -= vertCount % 3;
    if (g_partRangeCount >= BRIDGE_MAX_PART_RANGES) return;
    if (g_partVertCount + vertCount > PART_VERT_CAP) return;
    BridgePartRange& r = g_partRanges[g_partRangeCount++];
    r.firstVert = g_partVertCount;
    r.vertCount = vertCount;
    r.texId = texId;
    r.cls = cls;
    memcpy(&g_partVerts[g_partVertCount], verts, (size_t)vertCount * sizeof(BridgePartVert));
    g_partVertCount += vertCount;
}

void bridge_particles_commit()
{
    if (!g_hdr) return;
    uint32_t seq = g_hdr->particles.sequence;
    g_hdr->particles.sequence = seq + 1u;      // odd: writing
    MemoryBarrier();
    if (g_partRangeCount) {
        memcpy(bridge_part_ranges(g_base), g_partRanges, (size_t)g_partRangeCount * sizeof(BridgePartRange));
        memcpy(bridge_part_verts(g_base),  g_partVerts,  (size_t)g_partVertCount * sizeof(BridgePartVert));
    }
    g_hdr->particles.rangeCount = g_partRangeCount;
    g_hdr->particles.vertCount  = g_partVertCount;
    g_hdr->particles.valid      = g_partRangeCount > 0 ? 1u : 0u;
    MemoryBarrier();
    g_hdr->particles.sequence = seq + 2u;      // even: complete
}

void bridge_add_light(const float* origin3, const float* color3, float intensity,
                      float invRadius, uint32_t space)
{
    // Deferred renderers can submit several material/shadow variants for one light.
    // Identity is position + radius, while RGB remains live per frame for fire/flicker.
    for (uint32_t i = 0; i < g_stagedLightCount; ++i) {
        BridgeLight& e = g_stagedLights[i];
        if (e.origin[0] == origin3[0] && e.origin[1] == origin3[1] && e.origin[2] == origin3[2] &&
            e.invRadius == invRadius && e.space == space) {
            e.color[0] = color3[0]; e.color[1] = color3[1]; e.color[2] = color3[2]; e.color[3] = intensity;
            return;
        }
    }
    if (g_stagedLightCount >= BRIDGE_MAX_LIGHTS) return;
    BridgeLight& l = g_stagedLights[g_stagedLightCount++];
    l.origin[0] = origin3[0]; l.origin[1] = origin3[1]; l.origin[2] = origin3[2]; l.origin[3] = 0;
    l.color[0] = color3[0]; l.color[1] = color3[1]; l.color[2] = color3[2]; l.color[3] = intensity;
    l.invRadius = invRadius; l.space = space; l._pad[0] = l._pad[1] = 0;
}

void bridge_add_draw(uint32_t vbId, uint32_t ibId, uint32_t startIndex, uint32_t primCount,
                     uint32_t baseVertex, uint32_t minIndex, uint32_t numVertices,
                     const float* transform12, uint32_t materialId,
                     uint32_t texId, uint32_t uvOffset, uint32_t vbOffset, uint32_t posOffset,
                     uint32_t flags, uint32_t specTexId, uint32_t normalTexId, uint32_t depthTexId,
                     const float* specParams4, const float* parallaxParams4)
{
    if (!g_hdr || g_writeCount >= BRIDGE_MAX_DRAWS) return;
    if (vbId == 0xFFFFFFFFu || ibId == 0xFFFFFFFFu) return;
    BridgeDraw& d = bridge_draws(g_base, g_writeList)[g_writeCount++];
    d.vbId = vbId; d.ibId = ibId;
    d.startIndex = startIndex; d.primCount = primCount; d.baseVertex = baseVertex;
    d.minIndex = minIndex; d.numVertices = numVertices; d.materialId = materialId;
    d.texId = texId; d.uvOffset = uvOffset;
    d.vbOffset = vbOffset; d.posOffset = posOffset; d.flags = flags;
    d.specTexId = specTexId; d.normalTexId = normalTexId; d.depthTexId = depthTexId;
    d._materialPad = 0;
    for (int i = 0; i < 4; ++i) d.specParams[i] = specParams4 ? specParams4[i] : 0.0f;
    for (int i = 0; i < 4; ++i) d.parallaxParams[i] = parallaxParams4 ? parallaxParams4[i] : 0.0f;
    memcpy(d.transform, transform12, 12 * sizeof(float));
}

void bridge_end_frame()
{
    if (!g_hdr) return;
    g_hdr->drawCount[g_writeList] = g_writeCount;
    g_hdr->readyList  = g_writeList;         // publish the finished list
    // Publish this frame's camera together with its draws (before frameIndex++, which
    // is what the host waits on) so the camera + geometry are always the SAME frame.
    if (g_camValid) {
        memcpy(g_hdr->camera.proj, g_camProj, 16 * sizeof(float));
        g_hdr->camera.viewOrigin[0] = g_camOrigin[0];
        g_hdr->camera.viewOrigin[1] = g_camOrigin[1];
        g_hdr->camera.viewOrigin[2] = g_camOrigin[2];
        g_hdr->camera.valid = 1;
    }
    // Commit sky layers atomically with the frame (see bridge_add_sky_layer).
    for (uint32_t i = 0; i < g_stagedSkyCount; ++i) g_hdr->lighting.skyLayers[i] = g_stagedSky[i];
    g_hdr->lighting.skyLayerCount = g_stagedSkyCount;
    // Commit local lights atomically with the frame + camera (see bridge_add_light).
    BridgeLight* lights = bridge_lights(g_base);
    for (uint32_t i = 0; i < g_stagedLightCount; ++i) lights[i] = g_stagedLights[i];
    g_hdr->lighting.lightCount = g_stagedLightCount;
    g_hdr->veil = g_stagedVeil;
    g_hdr->frameIndex = g_hdr->frameIndex + 1;
}
