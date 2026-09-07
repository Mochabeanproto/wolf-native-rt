// DLSS Ray Reconstruction (DLSS-D) wrapper around NVIDIA NGX. Denoises + anti-aliases
// the path tracer's noisy G-buffer. Native resolution (DLAA), Preset F.
#pragma once
#include <d3d12.h>

// One frame's inputs. All resources at render resolution; output at the same size (DLAA).
struct DlssFrame {
    ID3D12Resource* color;        // linear HDR noisy radiance
    ID3D12Resource* output;       // RR result (linear HDR)
    ID3D12Resource* depth;        // linear view depth
    ID3D12Resource* motion;       // screen-space motion vectors (pixels)
    ID3D12Resource* normal;       // world-space normals
    ID3D12Resource* albedo;       // diffuse albedo
    ID3D12Resource* specAlbedo;   // specular albedo (F0)
    ID3D12Resource* roughness;    // roughness
    float  jitterX, jitterY;      // sub-pixel jitter applied this frame (pixels)
    int    reset;                 // 1 = discard history (level change / teleport)
    float* worldToView;           // 16 floats, row-major
    float* viewToClip;            // 16 floats, row-major
};

// Quality tiers (render:output linear ratio): 0=DLAA(1:1) 1=Quality(~1:1.5) 2=Balanced 3=Performance(1:2) 4=UltraPerf(1:3)
bool dlss_init(ID3D12Device* dev);                 // NGX init + capability probe
bool dlss_available();                             // RR usable on this GPU/driver
bool dlss_query_render_res(int outW, int outH, int quality, int* renderW, int* renderH);  // optimal render res for tier
bool dlss_create(ID3D12GraphicsCommandList* cl, int renderW, int renderH, int outW, int outH, int quality);
bool dlss_ready();                                 // feature created
bool dlss_evaluate(ID3D12GraphicsCommandList4* cl, const DlssFrame& f);     // denoise+upscale one frame
void dlss_shutdown();
