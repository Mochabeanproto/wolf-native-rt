#pragma once
// Tiny append-only file logger. Writes next to the game exe as wolf_rt.log so we
// can confirm injection and watch the frame loop without a debugger attached.
void log_printf(const char* fmt, ...);
