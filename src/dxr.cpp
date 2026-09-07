// M2a: D3D12 + DXR device bring-up, in-process with the game's D3D9 device.
//
// Picks the highest-performance hardware adapter, creates an ID3D12Device5
// (the DXR-capable interface), verifies D3D12_RAYTRACING_TIER, and creates a
// direct command queue. Everything here is logged; nothing renders yet.

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "dxr.h"
#include "log.h"

// Kept for later milestones (BLAS/TLAS, raygen, blit).
static IDXGIFactory6*      g_factory = nullptr;
static IDXGIAdapter1*      g_adapter = nullptr;
static ID3D12Device5*      g_device  = nullptr;
static ID3D12CommandQueue* g_queue   = nullptr;

static const char* rt_tier_name(D3D12_RAYTRACING_TIER t)
{
    switch (t) {
        case D3D12_RAYTRACING_TIER_NOT_SUPPORTED: return "NONE";
        case D3D12_RAYTRACING_TIER_1_0:           return "1_0";
        case D3D12_RAYTRACING_TIER_1_1:           return "1_1";
        default:                                  return "?";
    }
}

bool dxr_init()
{
    if (g_device) return true;

    UINT flags = 0;
#ifdef _DEBUG
    ID3D12Debug* dbg = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) { dbg->EnableDebugLayer(); dbg->Release(); }
    flags |= DXGI_CREATE_FACTORY_DEBUG;
#endif

    if (FAILED(CreateDXGIFactory2(flags, IID_PPV_ARGS(&g_factory)))) {
        log_printf("[dxr] CreateDXGIFactory2 failed");
        return false;
    }

    // Pick the first adapter that yields a D3D12 device with hardware RT.
    for (UINT i = 0; g_factory->EnumAdapterByGpuPreference(
             i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&g_adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 ad; g_adapter->GetDesc1(&ad);
        if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { g_adapter->Release(); g_adapter = nullptr; continue; }

        ID3D12Device5* dev = nullptr;
        if (SUCCEEDED(D3D12CreateDevice(g_adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
            dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5));
            log_printf("[dxr] adapter %u: %ls  VRAM=%lluMB  RaytracingTier=%s",
                       i, ad.Description, (unsigned long long)(ad.DedicatedVideoMemory >> 20),
                       rt_tier_name(o5.RaytracingTier));
            if (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0) {
                g_device = dev;
                break;
            }
            dev->Release();
        }
        g_adapter->Release(); g_adapter = nullptr;
    }

    if (!g_device) {
        log_printf("[dxr] no hardware-raytracing D3D12 device found");
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue)))) {
        log_printf("[dxr] CreateCommandQueue failed");
        return false;
    }

    log_printf("[dxr] D3D12 device + DXR ready (device=%p queue=%p) -- M2a OK", g_device, g_queue);
    return true;
}
