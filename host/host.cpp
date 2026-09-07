// wolf-native-rt : 64-bit host.
//   BR1  : read the live scene over the bridge (camera/geometry).            [done]
//   BR2b-1: open a window + D3D12 swapchain, clear to a colour driven by the
//           live camera -- proves the display path before adding DXR.        [here]
//   BR2b-2: BLAS/TLAS from the streamed geometry + raygen -> ray-traced image.
#include <windows.h>
#include "bridge.h"
#include "gfx.h"
#include <cstdio>
#include <cmath>

static bool g_quit = false;

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    if (m == WM_CLOSE || m == WM_DESTROY) { g_quit = true; PostQuitMessage(0); return 0; }
    if (m == WM_KEYDOWN && w == VK_END)   { g_quit = true; return 0; }
    return DefWindowProcA(h, m, w, l);
}

static void pump()
{
    MSG msg;
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    WNDCLASSA wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "WolfRTHost";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassA(&wc);
    HWND hwnd = CreateWindowA("WolfRTHost", "wolf-native-rt host  [END to quit]",
                              WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                              1280, 960, nullptr, nullptr, wc.hInstance, nullptr);
    RECT rc; GetClientRect(hwnd, &rc);
    if (!gfx_init(hwnd, rc.right, rc.bottom)) return 1;
    bool rt = gfx_rt_init();
    if (!rt) printf("[host] RT init failed -- falling back to colour clear\n");

    printf("[host] waiting for client shared memory '%s' ...\n", BRIDGE_SHM_NAME);
    HANDLE map = nullptr;
    while (!map && !g_quit) {
        map = OpenFileMappingA(FILE_MAP_READ, FALSE, BRIDGE_SHM_NAME);
        if (!map) { pump(); gfx_clear_present(0.05f, 0.05f, 0.08f); Sleep(60); }
    }
    const BridgeHeader* h = map ? (const BridgeHeader*)MapViewOfFile(map, FILE_MAP_READ, 0, 0,
                                                                     (SIZE_T)bridge_total_size()) : nullptr;
    if (h) {
        while (h->magic != BRIDGE_MAGIC && !g_quit) { pump(); gfx_clear_present(0.05f, 0.05f, 0.08f); Sleep(30); }
        printf("[host] connected: client pid=%u version=%u\n", h ? h->pid : 0, h ? h->version : 0);
        if (!g_quit && h->version != BRIDGE_VERSION) {
            printf("[host] bridge version mismatch: host=%u client=%u; rebuild/deploy d3d9.dll\n",
                   BRIDGE_VERSION, h->version);
            UnmapViewOfFile(h);
            CloseHandle(map);
            gfx_shutdown();
            return 2;
        }
    }
    printf("[host] press END (or close the window) to quit.\n");

    uint32_t last = 0xFFFFFFFFu;
    DWORD t0 = GetTickCount(); uint32_t f0 = 0;
    while (!g_quit) {
        pump();
        if (!rt || !h) { gfx_clear_present(0.05f, 0.06f, 0.09f); continue; }

        // Pace to the GAME: render EXACTLY one frame per new client frame. Re-rendering
        // the same bridge state (host fps > game fps) then jumping when it updates is
        // what made the whole world stutter. Waiting for frameIndex removes duplicate
        // renders (and, since the client bumps frameIndex last, guarantees the draws +
        // camera for that frame are fully published before we read them).
        uint32_t fi = h->frameIndex;
        if (fi == last) { Sleep(1); continue; }
        last = fi;
        gfx_render_scene(h);

        if ((fi % 60) == 0) {
            uint32_t rl = h->readyList & 1;
            DWORD now = GetTickCount();
            float fps = (now > t0) ? (fi - f0) * 1000.0f / (now - t0) : 0.0f;
            t0 = now; f0 = fi;
            printf("[host] frame %u draws=%u VBs=%u IBs=%u arena=%.1fMB eye=(%.0f %.0f %.0f) client~%.0ffps\n",
                   fi, h->drawCount[rl], h->vbCount, h->ibCount, h->arenaUsed / 1048576.0f,
                   h->camera.viewOrigin[0], h->camera.viewOrigin[1], h->camera.viewOrigin[2], fps);
        }
    }
    gfx_shutdown();
    printf("[host] bye.\n");
    return 0;
}
