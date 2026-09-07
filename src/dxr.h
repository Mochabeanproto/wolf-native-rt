#pragma once
// M2a: bring up a D3D12 device with DXR alongside the game's D3D9 device,
// in-process (the game is 32-bit; D3D12/DXR are not architecture-gated).
// Returns true if a device with hardware raytracing was created.
bool dxr_init();
