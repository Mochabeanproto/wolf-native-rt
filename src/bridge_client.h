#pragma once
#include <cstdint>
// 32-bit client side of the bridge: publishes the live scene into shared memory
// for the 64-bit host to ray-trace.
void bridge_client_init();
bool bridge_active();
void bridge_set_camera(const float* proj16, const float* origin3);

// Geometry streaming. VBs/IBs are deduped by their D3D9 pointer and copied into
// the shared arena once; returns a stable id (or 0xFFFFFFFF if the arena/registry
// is full). `data` is only read on first registration.
uint32_t bridge_vb_lookup(void* key);   // existing id, or 0xFFFFFFFF if not yet registered
uint32_t bridge_ib_lookup(void* key);
uint32_t bridge_tex_lookup(void* key);
uint32_t bridge_register_vb(void* key, const void* data, uint32_t size, uint32_t stride);
uint32_t bridge_register_ib(void* key, const void* data, uint32_t size, uint32_t indexStride, uint32_t count);
uint32_t bridge_register_tex(void* key, const void* data, uint32_t size,
                             uint32_t w, uint32_t h, uint32_t fmt, uint32_t rowPitch,
                             uint32_t rows, uint32_t faces, uint32_t mipCount = 1);

void bridge_begin_frame();
void bridge_add_draw(uint32_t vbId, uint32_t ibId, uint32_t startIndex, uint32_t primCount,
                     uint32_t baseVertex, uint32_t minIndex, uint32_t numVertices,
                     const float* transform12, uint32_t materialId,
                     uint32_t texId, uint32_t uvOffset, uint32_t vbOffset, uint32_t posOffset,
                     uint32_t flags = 0, uint32_t specTexId = 0xFFFFFFFFu,
                     uint32_t normalTexId = 0xFFFFFFFFu, uint32_t depthTexId = 0xFFFFFFFFu,
                     const float* specParams4 = nullptr, const float* parallaxParams4 = nullptr);
void bridge_end_frame();

// Lighting (BR4). Sun + local lights are re-published every frame; sky env map persists.
void bridge_set_sun(const float* dir3, const float* color3, float intensity);
void bridge_set_sky(uint32_t texId, uint32_t isCube, const float* colorScale4);
void bridge_add_sky_layer(uint32_t colorTexId, uint32_t alphaTexId, const float* xform6);
void bridge_set_sky_uv_matrix(const float* m6);
void bridge_publish_screen_sky(const void* data, uint32_t width, uint32_t height, uint32_t rowPitch);

// Expanded particle geometry (world space), streamed per frame. begin() resets the
// staging cursor; add_range() appends one sub-draw's triangles (already expanded from
// the game's particle VS on the CPU); commit() publishes atomically. Verts are
// BridgePartVert (see shared/bridge.h).
struct BridgePartVert;
void bridge_particles_begin();
void bridge_particles_add_range(const BridgePartVert* verts, uint32_t vertCount,
                                uint32_t texId, uint32_t cls);
void bridge_particles_commit();
void bridge_mark_veil(bool activeComposite, bool transitionPass,
                      uint32_t normal0TexId = 0xFFFFFFFFu, uint32_t normal1TexId = 0xFFFFFFFFu,
                      uint32_t maskTexId = 0xFFFFFFFFu, const float* distance4 = nullptr,
                      const float* ramp4 = nullptr, const float* xform0 = nullptr,
                      const float* xform1 = nullptr, float aspect = 1.0f);
void bridge_add_light(const float* origin3, const float* color3, float intensity,
                      float invRadius, uint32_t space);
