#pragma once
// M1 scene capture. Installed on the device vtable; dumps one frame on hotkey.
struct IDirect3DDevice9;

void capture_install(IDirect3DDevice9* dev);  // patch draw slots on the device
void capture_present_tick();                  // call once per Present (frame edge)
