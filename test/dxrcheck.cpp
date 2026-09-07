// Standalone probe: does D3D12 report hardware raytracing in THIS process arch?
// Build x86 and x64, run both. Answers whether DXR is architecture-gated.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>

int main()
{
    printf("=== dxrcheck (%zu-bit) ===\n", sizeof(void*) * 8);

    IDXGIFactory6* f = nullptr;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f)))) { printf("factory fail\n"); return 1; }

    IDXGIAdapter1* a = nullptr;
    for (UINT i = 0; f->EnumAdapterByGpuPreference(
             i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&a)) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        bool sw = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        ID3D12Device5* dev = nullptr;
        HRESULT hrDev = D3D12CreateDevice(a, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev));
        printf("adapter %u: %ls  sw=%d  createDevice hr=0x%08lX\n", i, d.Description, sw, hrDev);
        if (SUCCEEDED(hrDev)) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
            HRESULT hrq = dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5));
            printf("    OPTIONS5 hr=0x%08lX  RaytracingTier=%d  (10=tier1_0, 11=tier1_1)\n",
                   hrq, (int)o5.RaytracingTier);
            dev->Release();
        }
        a->Release(); a = nullptr;
    }
    f->Release();
    return 0;
}
