// SM1-3 vertex-shader interpreter. See vsinterp.h. Only the opcodes Wolf's particle
// programs use are handled; anything else is logged once and skipped.
#include "vsinterp.h"
#include <cmath>
#include <cstring>

void log_printf(const char* fmt, ...);   // from log.cpp

// D3D shader bytecode opcodes (D3DSIO_*).
enum {
    OP_NOP=0, OP_MOV=1, OP_ADD=2, OP_SUB=3, OP_MAD=4, OP_MUL=5, OP_RCP=6, OP_RSQ=7,
    OP_DP3=8, OP_DP4=9, OP_MIN=10, OP_MAX=11, OP_SLT=12, OP_SGE=13, OP_EXP=14, OP_LOG=15,
    OP_LIT=16, OP_DST=17, OP_LRP=18, OP_FRC=19, OP_M4x4=20, OP_M4x3=21, OP_M3x4=22,
    OP_M3x3=23, OP_M3x2=24, OP_POW=32, OP_CRS=33, OP_SGN=34, OP_ABS=35, OP_NRM=36,
    OP_SINCOS=37, OP_MOVA=46, OP_DCL=31, OP_DEF=81, OP_COMMENT=0xFFFE, OP_END=0xFFFF
};
// Register types (D3DSHADER_PARAM_REGISTER_TYPE).
enum { RT_TEMP=0, RT_INPUT=1, RT_CONST=2, RT_ADDR=3, RT_RASTOUT=4, RT_ATTROUT=5,
       RT_TEXCRDOUT=6 /* == OUTPUT */ };
// Source modifiers (D3DSHADER_PARAM_SRCMOD_TYPE).
enum { SM_NONE=0, SM_NEG=1, SM_ABS=11, SM_ABSNEG=12 };

static uint8_t reg_type(uint32_t tok) {
    // register type is split: bits [30:28] (high 3) and [12:11] (low 2).
    return (uint8_t)(((tok >> 28) & 0x7) | ((tok >> 11) & 0x3) << 3);
}
static uint8_t reg_num(uint32_t tok) { return (uint8_t)(tok & 0x7FF); }

bool VSProgram::load(const uint32_t* t, uint32_t sizeBytes)
{
    valid = false; insCount = 0;
    memset(defSet, 0, sizeof(defSet));
    memset(in, 0, sizeof(in));
    uint32_t n = sizeBytes / 4;
    if (n < 2) return false;
    uint32_t ver = t[0];
    if ((ver & 0xFFFF0000u) != 0xFFFE0000u) return false;   // not a vertex shader
    major = (ver >> 8) & 0xFF; minor = ver & 0xFF;
    uint32_t i = 1;
    while (i < n) {
        uint32_t tok = t[i];
        uint16_t op = (uint16_t)(tok & 0xFFFF);
        if (op == OP_END) break;
        if (op == OP_COMMENT) { i += 1 + ((tok >> 16) & 0x7FFF); continue; }
        uint32_t len = (tok >> 24) & 0xF;       // SM2+ instruction length (param DWORDs)
        if (op == OP_DEF) {
            uint32_t dtok = t[i+1];
            uint8_t rn = reg_num(dtok);
            for (int k = 0; k < 4; ++k) memcpy(&def[rn][k], &t[i+2+k], 4);
            defSet[rn] = 1;
            i += 1 + len; continue;
        }
        if (op == OP_DCL) {
            uint32_t usageTok = t[i+1];
            uint32_t dtok = t[i+2];
            if (reg_type(dtok) == RT_INPUT) {
                uint8_t rn = reg_num(dtok);
                in[rn].used = 1;
                in[rn].usage = (uint8_t)(usageTok & 0x1F);
                in[rn].usageIndex = (uint8_t)((usageTok >> 16) & 0xF);
            }
            i += 1 + len; continue;
        }
        // Regular executable instruction: 1 dst + (len-1) src tokens.
        if (insCount >= 512) return false;
        Ins& in2 = ins[insCount];
        in2.op = op; in2.nSrc = 0; in2.sat = 0;
        uint32_t p = i + 1;
        uint32_t dtok = t[p++];
        in2.dst.type = reg_type(dtok);
        in2.dst.reg  = reg_num(dtok);
        in2.dst.mask = (uint8_t)((dtok >> 16) & 0xF);
        in2.sat      = (dtok & (1u << 20)) ? 1 : 0;   // D3DSPDM_SATURATE
        uint32_t srcEnd = i + 1 + len;
        while (p < srcEnd && in2.nSrc < 4) {
            uint32_t stok = t[p++];
            Src& s = in2.src[in2.nSrc++];
            s.type = reg_type(stok);
            s.reg  = reg_num(stok);
            s.swizzle = (uint8_t)((stok >> 16) & 0xFF);
            s.mod = (uint8_t)((stok >> 24) & 0xF);
        }
        insCount++;
        i = srcEnd;
        if (len == 0) { i = p; }   // safety: SM1 shaders lack a length field
    }
    valid = insCount > 0;
    return valid;
}

// ---- evaluation ------------------------------------------------------------

static void read_src(const VSProgram::Src& s, const float v[16][4], const float ceff[256][4],
                     const float r[32][4], const float out_reg[16][4], float o[4])
{
    const float* base;
    switch (s.type) {
        case RT_TEMP:  base = r[s.reg]; break;
        case RT_INPUT: base = v[s.reg]; break;
        case RT_CONST: base = ceff[s.reg]; break;
        default:       base = out_reg[s.reg]; break;   // reading an output (rare); tolerate
    }
    float t[4] = { base[0], base[1], base[2], base[3] };
    // swizzle: 2 bits per component
    for (int k = 0; k < 4; ++k) o[k] = t[(s.swizzle >> (k * 2)) & 3];
    switch (s.mod) {
        case SM_NEG:    for (int k=0;k<4;++k) o[k] = -o[k]; break;
        case SM_ABS:    for (int k=0;k<4;++k) o[k] = fabsf(o[k]); break;
        case SM_ABSNEG: for (int k=0;k<4;++k) o[k] = -fabsf(o[k]); break;
        default: break;
    }
}

static void write_dst(const VSProgram::Dst& d, uint8_t sat, const float val[4],
                      float r[32][4], float out_reg[16][4], int* oPosIdx)
{
    float v[4] = { val[0], val[1], val[2], val[3] };
    if (sat) for (int k=0;k<4;++k) v[k] = v[k] < 0 ? 0 : (v[k] > 1 ? 1 : v[k]);
    float* dst;
    switch (d.type) {
        case RT_TEMP:     dst = r[d.reg]; break;
        case RT_RASTOUT:  dst = out_reg[8 + d.reg]; if (d.reg == 0) *oPosIdx = 8; break; // oPos at slot 8
        case RT_ATTROUT:  dst = out_reg[12 + d.reg]; break;  // oD0/oD1 at slots 12,13
        case RT_TEXCRDOUT:dst = out_reg[d.reg]; break;       // oT0.. at slots 0..
        default:          dst = r[d.reg]; break;
    }
    for (int k = 0; k < 4; ++k) if (d.mask & (1 << k)) dst[k] = v[k];
}

bool vs_run(const VSProgram& p, const float v[16][4], const float c[256][4],
            float oPos[4], float oT0[4], float oD0[4])
{
    if (!p.valid) return false;
    // Effective constants = device constants with def[] overlaid.
    static thread_local float ceff[256][4];
    memcpy(ceff, c, sizeof(ceff));
    for (int k = 0; k < 256; ++k) if (p.defSet[k]) memcpy(ceff[k], p.def[k], sizeof(float) * 4);

    float r[32][4];      memset(r, 0, sizeof(r));
    float out[16][4];    memset(out, 0, sizeof(out));  // slots: 0..7 oT, 8 oPos(+RASTOUT), 12/13 oD
    int oPosIdx = 8;

    for (int ii = 0; ii < p.insCount; ++ii) {
        const VSProgram::Ins& in = p.ins[ii];
        float s0[4]={0}, s1[4]={0}, s2[4]={0};
        if (in.nSrc > 0) read_src(in.src[0], v, ceff, r, out, s0);
        if (in.nSrc > 1) read_src(in.src[1], v, ceff, r, out, s1);
        if (in.nSrc > 2) read_src(in.src[2], v, ceff, r, out, s2);
        float res[4] = {0,0,0,0};
        switch (in.op) {
            case OP_MOV: case OP_MOVA: for(int k=0;k<4;++k) res[k]=s0[k]; break;
            case OP_ADD: for(int k=0;k<4;++k) res[k]=s0[k]+s1[k]; break;
            case OP_SUB: for(int k=0;k<4;++k) res[k]=s0[k]-s1[k]; break;
            case OP_MUL: for(int k=0;k<4;++k) res[k]=s0[k]*s1[k]; break;
            case OP_MAD: for(int k=0;k<4;++k) res[k]=s0[k]*s1[k]+s2[k]; break;
            case OP_MIN: for(int k=0;k<4;++k) res[k]=s0[k]<s1[k]?s0[k]:s1[k]; break;
            case OP_MAX: for(int k=0;k<4;++k) res[k]=s0[k]>s1[k]?s0[k]:s1[k]; break;
            case OP_SLT: for(int k=0;k<4;++k) res[k]=s0[k]<s1[k]?1.0f:0.0f; break;
            case OP_SGE: for(int k=0;k<4;++k) res[k]=s0[k]>=s1[k]?1.0f:0.0f; break;
            case OP_FRC: for(int k=0;k<4;++k) res[k]=s0[k]-floorf(s0[k]); break;
            case OP_LRP: for(int k=0;k<4;++k) res[k]=s2[k]+s0[k]*(s1[k]-s2[k]); break;
            case OP_ABS: for(int k=0;k<4;++k) res[k]=fabsf(s0[k]); break;
            case OP_SGN: for(int k=0;k<4;++k) res[k]=(s0[k]>0)-(s0[k]<0); break;
            case OP_RCP: { float d=s0[0]; d=(d==0)?0:1.0f/d; for(int k=0;k<4;++k) res[k]=d; } break;
            case OP_RSQ: { float d=fabsf(s0[0]); d=(d==0)?0:1.0f/sqrtf(d); for(int k=0;k<4;++k) res[k]=d; } break;
            case OP_EXP: { float d=powf(2.0f,s0[0]); for(int k=0;k<4;++k) res[k]=d; } break;
            case OP_LOG: { float a=fabsf(s0[0]); float d=(a==0)?-3.4e38f:log2f(a); for(int k=0;k<4;++k) res[k]=d; } break;
            case OP_POW: { float d=powf(fabsf(s0[0]),s1[0]); for(int k=0;k<4;++k) res[k]=d; } break;
            case OP_DP3: { float d=s0[0]*s1[0]+s0[1]*s1[1]+s0[2]*s1[2]; for(int k=0;k<4;++k) res[k]=d; } break;
            case OP_DP4: { float d=s0[0]*s1[0]+s0[1]*s1[1]+s0[2]*s1[2]+s0[3]*s1[3]; for(int k=0;k<4;++k) res[k]=d; } break;
            case OP_NRM: { float l=sqrtf(s0[0]*s0[0]+s0[1]*s0[1]+s0[2]*s0[2]); l=(l==0)?0:1.0f/l;
                           for(int k=0;k<3;++k) res[k]=s0[k]*l; res[3]=s0[3]; } break;
            case OP_CRS: res[0]=s0[1]*s1[2]-s0[2]*s1[1]; res[1]=s0[2]*s1[0]-s0[0]*s1[2];
                         res[2]=s0[0]*s1[1]-s0[1]*s1[0]; res[3]=0; break;
            case OP_SINCOS: { float a=s0[0]; res[0]=cosf(a); res[1]=sinf(a); res[2]=0; res[3]=0; } break;
            case OP_M4x4: case OP_M4x3: case OP_M3x4: case OP_M3x3: case OP_M3x2: {
                // mNxM dst, src0, src1: dst.i = dot(src0, src1[i]) for consecutive const rows.
                int rows = (in.op==OP_M3x2)?2:(in.op==OP_M4x3||in.op==OP_M3x3)?3:4;
                int comps= (in.op==OP_M3x3||in.op==OP_M3x4||in.op==OP_M3x2)?3:4;
                const VSProgram::Src& mb = in.src[1];
                for (int rrow=0; rrow<rows; ++rrow) {
                    VSProgram::Src rs = mb; rs.reg = mb.reg + rrow; float m[4];
                    read_src(rs, v, ceff, r, out, m);
                    float d=0; for(int k=0;k<comps;++k) d += s0[k]*m[k];
                    res[rrow]=d;
                }
            } break;
            default: {
                static int warned[512] = {0};
                if (in.op < 512 && !warned[in.op]) { warned[in.op]=1; log_printf("[vsinterp] unhandled opcode %u (nop)", in.op); }
            } break;
        }
        write_dst(in.dst, in.sat, res, r, out, &oPosIdx);
    }
    memcpy(oPos, out[8], sizeof(float)*4);
    memcpy(oT0,  out[0], sizeof(float)*4);
    memcpy(oD0,  out[12], sizeof(float)*4);
    return true;
}
