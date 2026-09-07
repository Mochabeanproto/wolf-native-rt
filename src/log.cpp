#include "log.h"
#include <windows.h>
#include <cstdio>
#include <cstdarg>
#include <share.h>

// Log file lives in the process working directory (the game folder). One shared
// handle, guarded by a critical section since D3D9 draws can come from >1 thread.
static CRITICAL_SECTION g_lock;
static bool g_init = false;
static FILE* g_fp = nullptr;

static void ensure_open()
{
    if (g_init) return;
    InitializeCriticalSection(&g_lock);
    // _SH_DENYNO so external readers can tail the log while the game holds it open.
    g_fp = _fsopen("wolf_rt.log", "w", _SH_DENYNO);
    g_init = true;
    if (g_fp) {
        fprintf(g_fp, "=== wolf-native-rt proxy d3d9 ===\n");
        fflush(g_fp);
    }
}

void log_printf(const char* fmt, ...)
{
    ensure_open();
    if (!g_fp) return;
    EnterCriticalSection(&g_lock);
    SYSTEMTIME st; GetLocalTime(&st);
    fprintf(g_fp, "[%02d:%02d:%02d.%03d] ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    va_list ap; va_start(ap, fmt);
    vfprintf(g_fp, fmt, ap);
    va_end(ap);
    fputc('\n', g_fp);
    fflush(g_fp);
    LeaveCriticalSection(&g_lock);
}
