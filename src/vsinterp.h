// A tiny software interpreter for Shader Model 1-3 vertex shaders. Wolf's particle
// billboards are expanded entirely on the GPU by a ~160-instruction vs_2_0 program
// (physics sim + rotation + filmstrip + soft-particle + billboard corner); D3D9 has
// no stream-out, so to ray-trace the particles we re-run that exact program on the
// CPU per vertex and read back the expanded, animated world-space geometry.
//
// Only the opcodes Wolf's particle shaders use are implemented; an unknown opcode is
// logged once and treated as a nop so we fail visibly rather than silently wrong.
#pragma once
#include <cstdint>

struct VSProgram {
    // Executable instructions (def/dcl folded into setup below).
    struct Src { uint8_t reg, type, swizzle, mod; };   // type: D3DSPR_*; swizzle: 2 bits/comp; mod: D3DSPSM_*
    struct Dst { uint8_t reg, type, mask; };           // mask: bit0..3 = xyzw; also carries saturate
    struct Ins { uint16_t op; uint8_t nSrc, sat; Dst dst; Src src[4]; };

    Ins      ins[512];
    int      insCount = 0;
    float    def[256][4];        // def cN constants baked into the program
    uint8_t  defSet[256];        // 1 where def[] overrides the device constant
    // Input register -> declaration semantic, so the caller can bind VB data.
    struct InReg { uint8_t used, usage, usageIndex; };
    InReg    in[16];
    int      major = 0, minor = 0;
    bool     valid = false;

    bool load(const uint32_t* tokens, uint32_t sizeBytes);
};

// Execute the loaded program for one vertex.
//   v    : 16 input registers already filled from the VB (per InReg semantics)
//   c    : 256 float constant registers (c0..c255) as set on the device
//   oPos : clip-space output position (oPos / SV_Position)
//   oT0  : output texcoord0 (particle UV)
//   oD0  : output color0 (modulated particle color, 0..1)
// Returns false only if the program is invalid.
bool vs_run(const VSProgram& p, const float v[16][4], const float c[256][4],
            float oPos[4], float oT0[4], float oD0[4]);
