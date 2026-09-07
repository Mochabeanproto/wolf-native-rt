#pragma once
// D3D12 presentation + DXR path tracer for the host window.
#include <windows.h>
struct BridgeHeader;
bool gfx_init(HWND hwnd, int w, int h);
void gfx_clear_present(float r, float g, float b);
bool gfx_rt_init();                          // BR2b-2: build the RT pipeline
void gfx_render_scene(const BridgeHeader* h);// BR2b-2: ray-trace one frame + present
void gfx_shutdown();
