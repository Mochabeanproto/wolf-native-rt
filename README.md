# wolf-native-rt

A from-scratch **native DXR (D3D12) path tracer** for Wolfenstein (2009), Raven's
D3D9 idTech4 fork. The game keeps rendering through
Direct3D 9; we intercept its draw stream, reconstruct the scene each frame, and
re-render it with hardware ray tracing.

## Architecture

Two processes joined by a shared-memory bridge:

```
Wolf2.exe (x86)                             wolfrt_host.exe (x64)
  proxy d3d9.dll  ──►  Local\WolfRT_Bridge  ──►  D3D12 / DXR path tracer
  (scene extractor)     (POD arena, versioned)     (own window, DLSS-RR, SHARC)
```

The game is 32-bit and NVIDIA's DLSS/NGX runtime is 64-bit-only, so the renderer
can't live in-process. Instead a proxy `d3d9.dll` dropped next to the exe
**extracts** the scene — geometry, materials, camera, lights, sky, particles —
and publishes it into a versioned shared-memory arena. A separate 64-bit host
maps that memory and path-traces it in its own window.

The game imports exactly one symbol from d3d9 — `Direct3DCreate9` — so the proxy
exports that, forwards the rest to the real system `d3d9.dll`, and patches down
the COM vtables to reach the draw path:

```
Direct3DCreate9 -> real d3d9 -> patch IDirect3D9::CreateDevice        (vtbl[16])
CreateDevice    -> real       -> patch IDirect3DDevice9::Present       (vtbl[17])
                              -> patch DrawIndexedPrimitive / DrawPrimitive (vtbl[82]/[81])
```

The bridge contract (`shared/bridge.h`) is POD-only, fixed-layout, and versioned
(`BRIDGE_VERSION`); the host refuses to run against a mismatched client.

## Layout

### Client — 32-bit scene extractor (`src/`)
- `hooks.cpp` — the proxy `d3d9.dll`: `Direct3DCreate9` + vtable patching.
- `capture.cpp` — the core. Per draw, reads live device state to recover the
  camera (`P = MVP·inv(MV)`), streams world geometry + materials, classifies
  emissive / sky-dome / particle / Veil draws, and recovers deferred lights.
- `ctab.cpp` — parses each shader's D3D9 constant table (CTAB) so constants are
  found by name, not by assumed register.
- `vsinterp.cpp` — a tiny SM1-3 vertex-shader interpreter. D3D9 has no
  stream-out, so the game's GPU particle VS is re-run on the CPU to recover the
  expanded, animated billboards as ray-traceable world-space geometry.
- `bridge_client.cpp` — publishes the scene into shared memory with staged,
  torn-read-safe commits.

### Host — 64-bit DXR path tracer (`host/`)
- `gfx.cpp` — D3D12 device + swapchain, per-surface BLAS and a persistent world
  TLAS (static geometry is promoted so off-screen surfaces still cast shadows /
  GI), bindless PBR materials, local-light + sun lighting, quarter-res fog
  volumes, and frame pacing to the game.
- `shader.hlsl` — SM 6.6 raygen / miss / closest-hit / any-hit library: primary
  G-buffer, multi-bounce diffuse GI, NEE + ray-traced shadows, sky.
- `sharc_resolve.hlsl` — resolve pass for NVIDIA **SHARC** world-space radiance
  cache (GI).
- `fog.hlsl` — analytic ellipsoid smoke volumes (converted from smoke sprites).
- `tonemap.hlsl` — tonemap + Veil-vision composite to the backbuffer.
- `dlss.cpp` — **DLSS Ray Reconstruction** integration (denoise + upscale).

Also included: `bridge_diag.cpp` / `particle_tex_dump.cpp` (standalone
diagnostics) and `test/` (DXR / Vulkan-RT capability probes).

## Dependencies

These are **not** vendored in this repo — fetch them yourself:

- **NVIDIA SHARC** (RTX SDK) — headers included by `host/shader.hlsl` from
  `third_party/SHARC/include`. Clone it into `third_party/SHARC`
  (<https://github.com/NVIDIA-RTX/SHARC>). Ships under the NVIDIA RTX SDK license.
- **NVIDIA DLSS / NGX SDK** — for Ray Reconstruction. `host/build_host.ps1`
  expects the NGX headers/lib and stages the runtime DLLs
  (`nvngx_dlss.dll`, `nvngx_dlssd.dll`, `nvngx_dlssg.dll`) next to the host exe.
- **Windows 10 SDK** — D3D12 headers + `dxc` (shaders are SM 6.6).
- **Visual Studio 2022+** with the x86 and x64 toolsets.

The build scripts contain machine-specific absolute paths (game folder, vcvars,
NGX cache, DLSS DLL source) — edit those to match your environment before building.

## Build

```powershell
.\build.ps1                 # -> build\d3d9.dll        (32-bit proxy client)
.\build.ps1 -Deploy         # build + copy into the game folder

cd host; .\build_host.ps1   # -> host\build\wolfrt_host.exe  (64-bit host)
```

## Run

1. Launch `wolfrt_host.exe` — it opens a window and waits for the bridge.
2. Run the game **from its own folder** so the local proxy `d3d9.dll` wins the
   DLL search over `system32`. The client creates the shared memory; the host
   connects and starts path-tracing. Press **END** (or close the host window) to
   quit.

`Ctrl+F10` in-game dumps one frame's draws / shader constants to `wolf_rt.log`
for extraction debugging. A number of `WOLF_*` environment variables toggle
diagnostics and features (e.g. `WOLF_DBG`, `WOLF_PERF`, `WOLF_NOSHARC`,
`WOLF_NORR`, `WOLF_DLSS=<dlaa|quality|balanced|performance|ultraperf>`).

## Status / notes

- Implemented: geometry + material extraction, primary rays, PBR direct lighting
  (sun + recovered local lights), sky dome + cloud layers + live sky env map,
  path-traced diffuse GI (SHARC), ReSTIR DI, particles (CPU-expanded billboards),
  smoke as fog volumes, the Veil effect, and DLSS-RR denoise/upscale.
- `IDirect3DDevice9::Reset` is **not** hooked yet; D3D12 interop resources must be
  released/recreated across a device reset before this is robust.
- Present/draw hooks are installed on the first device only (the game creates one).
