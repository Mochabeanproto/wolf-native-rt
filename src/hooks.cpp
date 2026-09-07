// wolf-native-rt : M0 proxy d3d9.dll
//
// The game (Wolf2.exe, x86) imports exactly one symbol from d3d9.dll:
// Direct3DCreate9. We export that, forward to the real system d3d9.dll, then walk
// down the COM vtables by patching only the slots we care about -- no full wrapper.
//
//   Direct3DCreate9  -> real -> patch IDirect3D9::CreateDevice (vtbl[16])
//   CreateDevice     -> real -> patch IDirect3DDevice9::Present (vtbl[17])
//   Present          -> log a heartbeat, call real
//
// M1 will add DrawIndexedPrimitive / SetTexture / SetStreamSource / SetTransform
// detours to the same device vtable to start extracting the scene.

#include <windows.h>
#include <d3d9.h>
#include <cstring>
#include "log.h"
#include "capture.h"
#include "bridge_client.h"

// --- generic vtable slot patch (vtables live in read-only .rdata) -------------
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

// --- IDirect3DDevice9::Present (vtbl index 17) --------------------------------
typedef HRESULT(STDMETHODCALLTYPE* Present_t)(IDirect3DDevice9*, const RECT*, const RECT*, HWND, const RGNDATA*);
static Present_t s_realPresent = nullptr;
static unsigned  s_frame = 0;

static HRESULT STDMETHODCALLTYPE Hook_Present(IDirect3DDevice9* self, const RECT* src, const RECT* dst, HWND wnd, const RGNDATA* dirty)
{
    if (s_frame < 5 || (s_frame % 600) == 0)
        log_printf("[present] frame %u", s_frame);
    ++s_frame;
    capture_present_tick();
    return s_realPresent(self, src, dst, wnd, dirty);
}

// --- IDirect3D9::CreateDevice (vtbl index 16) ---------------------------------
typedef HRESULT(STDMETHODCALLTYPE* CreateDevice_t)(IDirect3D9*, UINT, D3DDEVTYPE, HWND, DWORD, D3DPRESENT_PARAMETERS*, IDirect3DDevice9**);
static CreateDevice_t s_realCreateDevice = nullptr;

static HRESULT STDMETHODCALLTYPE Hook_CreateDevice(IDirect3D9* self, UINT adapter, D3DDEVTYPE type, HWND focus, DWORD flags, D3DPRESENT_PARAMETERS* pp, IDirect3DDevice9** ppDev)
{
    if (pp)
        log_printf("[createdevice] %ux%u fmt=%d windowed=%d flags=0x%08x swap=%d",
                   pp->BackBufferWidth, pp->BackBufferHeight, pp->BackBufferFormat,
                   pp->Windowed, flags, pp->SwapEffect);
    HRESULT hr = s_realCreateDevice(self, adapter, type, focus, flags, pp, ppDev);
    if (SUCCEEDED(hr) && ppDev && *ppDev && !s_realPresent) {
        s_realPresent = (Present_t)patch_vtable(*ppDev, 17, (void*)Hook_Present);
        log_printf("[createdevice] hr=0x%08x device=%p present hooked (real=%p)", hr, *ppDev, s_realPresent);
        capture_install(*ppDev);
        bridge_client_init();
    } else {
        log_printf("[createdevice] hr=0x%08x device=%p", hr, ppDev ? *ppDev : nullptr);
    }
    return hr;
}

// --- exported Direct3DCreate9 -------------------------------------------------
typedef IDirect3D9*(WINAPI* Direct3DCreate9_t)(UINT);
static Direct3DCreate9_t s_realCreate = nullptr;
static HMODULE s_realDll = nullptr;

static void load_real_d3d9()
{
    if (s_realCreate) return;
    char path[MAX_PATH];
    UINT n = GetSystemDirectoryA(path, MAX_PATH);
    strcpy_s(path + n, MAX_PATH - n, "\\d3d9.dll");
    s_realDll = LoadLibraryA(path);
    s_realCreate = s_realDll ? (Direct3DCreate9_t)GetProcAddress(s_realDll, "Direct3DCreate9") : nullptr;
    log_printf("[init] real d3d9 = %s (mod=%p create=%p)", path, s_realDll, s_realCreate);
}

extern "C" IDirect3D9 * WINAPI Direct3DCreate9(UINT sdkVersion)
{
    load_real_d3d9();
    if (!s_realCreate) return nullptr;
    IDirect3D9* d3d = s_realCreate(sdkVersion);
    if (d3d && !s_realCreateDevice) {
        s_realCreateDevice = (CreateDevice_t)patch_vtable(d3d, 16, (void*)Hook_CreateDevice);
        log_printf("[init] IDirect3D9=%p createdevice hooked (real=%p)", d3d, s_realCreateDevice);
    }
    return d3d;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(inst);
        log_printf("[dllmain] proxy d3d9 attached, pid=%lu", GetCurrentProcessId());
    }
    return TRUE;
}
