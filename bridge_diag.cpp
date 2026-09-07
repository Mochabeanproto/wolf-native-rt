#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "shared/bridge.h"

struct Finding {
    uint32_t draw, tris, outsideDeclared, outsideBuffer, nonFinite;
    float maxEdge, diagonal;
    BridgeDraw d;
};

static uint32_t read_index(const uint8_t* p, uint32_t stride, uint32_t i) {
    if (stride == 2) { uint16_t v; memcpy(&v, p + (uint64_t)i * 2, 2); return v; }
    uint32_t v; memcpy(&v, p + (uint64_t)i * 4, 4); return v;
}

int main() {
    HANDLE map = OpenFileMappingA(FILE_MAP_READ, FALSE, BRIDGE_SHM_NAME);
    if (!map) { printf("bridge mapping not found (%lu)\n", GetLastError()); return 2; }
    void* base = MapViewOfFile(map, FILE_MAP_READ, 0, 0, (SIZE_T)bridge_total_size());
    if (!base) { printf("map failed (%lu)\n", GetLastError()); CloseHandle(map); return 2; }
    const BridgeHeader* h = (const BridgeHeader*)base;
    if (h->magic != BRIDGE_MAGIC || h->version != BRIDGE_VERSION) {
        printf("bridge mismatch magic=%08x version=%u\n", h->magic, h->version); return 3;
    }
    uint32_t list = h->readyList & 1u;
    uint32_t count = std::min(h->drawCount[list], BRIDGE_MAX_DRAWS);
    const BridgeDraw* draws = bridge_draws(base, list);
    const BridgeVB* vbs = bridge_vbs(base);
    const BridgeIB* ibs = bridge_ibs(base);
    const uint8_t* arena = bridge_arena(base);
    std::vector<Finding> found;
    uint64_t totalOutside = 0, totalInvalid = 0;

    for (uint32_t n = 0; n < count; ++n) {
        const BridgeDraw& d = draws[n];
        if (d.vbId >= h->vbCount || d.ibId >= h->ibCount) continue;
        const BridgeVB& vb = vbs[d.vbId]; const BridgeIB& ib = ibs[d.ibId];
        Finding f = {}; f.draw = n; f.tris = d.primCount; f.d = d;
        float lo[3] = { INFINITY, INFINITY, INFINITY }, hi[3] = { -INFINITY, -INFINITY, -INFINITY };
        int64_t vertexBase = (int64_t)d.vbOffset + (int64_t)(int32_t)d.baseVertex * vb.stride;
        for (uint64_t t = 0; t < (uint64_t)d.primCount; ++t) {
            float p[3][3] = {}; bool valid = true;
            for (int k = 0; k < 3; ++k) {
                uint64_t elem = (uint64_t)d.startIndex + t * 3 + k;
                if ((elem + 1) * ib.indexStride > ib.size) { ++f.outsideBuffer; valid = false; continue; }
                uint32_t idx = read_index(arena + ib.arenaOff, ib.indexStride, (uint32_t)elem);
                if (idx < d.minIndex || (uint64_t)idx >= (uint64_t)d.minIndex + d.numVertices) ++f.outsideDeclared;
                int64_t at = vertexBase + (int64_t)idx * vb.stride + d.posOffset;
                if (at < 0 || (uint64_t)at + 12 > vb.size) { ++f.outsideBuffer; valid = false; continue; }
                memcpy(p[k], arena + vb.arenaOff + at, 12);
                if (!std::isfinite(p[k][0]) || !std::isfinite(p[k][1]) || !std::isfinite(p[k][2])) {
                    ++f.nonFinite; valid = false; continue;
                }
                for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], p[k][a]); hi[a] = std::max(hi[a], p[k][a]); }
            }
            if (valid) for (int e = 0; e < 3; ++e) {
                int q = (e + 1) % 3; float dx=p[e][0]-p[q][0], dy=p[e][1]-p[q][1], dz=p[e][2]-p[q][2];
                f.maxEdge = std::max(f.maxEdge, std::sqrt(dx*dx + dy*dy + dz*dz));
            }
        }
        if (std::isfinite(lo[0])) { float dx=hi[0]-lo[0],dy=hi[1]-lo[1],dz=hi[2]-lo[2]; f.diagonal=std::sqrt(dx*dx+dy*dy+dz*dz); }
        totalOutside += f.outsideDeclared; totalInvalid += f.outsideBuffer + f.nonFinite;
        if (f.outsideDeclared || f.outsideBuffer || f.nonFinite || f.maxEdge > 5000.0f) found.push_back(f);
    }
    std::sort(found.begin(), found.end(), [](const Finding& a, const Finding& b) {
        if (a.outsideBuffer + a.nonFinite != b.outsideBuffer + b.nonFinite)
            return a.outsideBuffer + a.nonFinite > b.outsideBuffer + b.nonFinite;
        if (a.outsideDeclared != b.outsideDeclared) return a.outsideDeclared > b.outsideDeclared;
        return a.maxEdge > b.maxEdge;
    });
    printf("frame=%u list=%u draws=%u vb=%u ib=%u findings=%zu outsideDeclared=%llu invalid=%llu\n",
           h->frameIndex, list, count, h->vbCount, h->ibCount, found.size(), totalOutside, totalInvalid);
    printf("lighting: sun=%u lights=%u skyTex=%u skyCube=%u skyScale=(%.3f %.3f %.3f %.3f)\n",
           h->lighting.sunValid, h->lighting.lightCount, h->lighting.skyTexId, h->lighting.skyIsCube,
           h->lighting.skyColorScale[0], h->lighting.skyColorScale[1],
           h->lighting.skyColorScale[2], h->lighting.skyColorScale[3]);
    uint32_t depthDraws = 0, activeParallax = 0;
    for (uint32_t n = 0; n < count; ++n) {
        const BridgeDraw& d = draws[n];
        if (d.depthTexId == 0xFFFFFFFFu) continue;
        ++depthDraws;
        if (fabsf(d.parallaxParams[0]) > 1e-7f || fabsf(d.parallaxParams[1]) > 1e-7f) ++activeParallax;
        if (depthDraws <= 40)
            printf("height[%u] draw=%u depthTex=%u scaleBias=(%.7f %.7f %.7f %.7f) normalTex=%u baseTex=%u\n",
                   depthDraws-1, n, d.depthTexId,
                   d.parallaxParams[0], d.parallaxParams[1], d.parallaxParams[2], d.parallaxParams[3],
                   d.normalTexId, d.texId);
    }
    printf("height summary: depthDraws=%u activeScaleBias=%u\n", depthDraws, activeParallax);
    {
        const BridgeParticles& pp=h->particles;
        const BridgePartRange* ranges=bridge_part_ranges(base);
        const BridgePartVert* verts=bridge_part_verts(base);
        uint32_t clsCount[3]={},clsVerts[3]={};
        printf("particles: valid=%u seq=%u ranges=%u verts=%u\n",pp.valid,pp.sequence,pp.rangeCount,pp.vertCount);
        for(uint32_t ri=0;ri<pp.rangeCount && ri<BRIDGE_MAX_PART_RANGES;++ri){
            const BridgePartRange& r=ranges[ri];
            if(r.cls<3){++clsCount[r.cls];clsVerts[r.cls]+=r.vertCount;}
            if(ri<80 && (uint64_t)r.firstVert+r.vertCount<=pp.vertCount){
                float u0=INFINITY,v0=INFINITY,u1=-INFINITY,v1=-INFINITY,maxEdge=0;
                for(uint32_t j=0;j<r.vertCount;++j){const BridgePartVert& v=verts[r.firstVert+j];u0=std::min(u0,v.uv[0]);v0=std::min(v0,v.uv[1]);u1=std::max(u1,v.uv[0]);v1=std::max(v1,v.uv[1]);}
                for(uint32_t j=0;j+2<r.vertCount && j<60;j+=3)for(int a=0;a<3;++a)for(int b=a+1;b<3;++b){const float* x=verts[r.firstVert+j+a].pos;const float* y=verts[r.firstVert+j+b].pos;float dx=x[0]-y[0],dy=x[1]-y[1],dz=x[2]-y[2];maxEdge=std::max(maxEdge,sqrtf(dx*dx+dy*dy+dz*dz));}
                printf("part[%u] cls=%u first=%u verts=%u tex=%u uv=(%.4f %.4f)-(%.4f %.4f) edge<=%.2f\n",ri,r.cls,r.firstVert,r.vertCount,r.texId,u0,v0,u1,v1,maxEdge);
            }
        }
        printf("particle classes E/V/A: ranges=%u/%u/%u verts=%u/%u/%u\n",clsCount[0],clsCount[1],clsCount[2],clsVerts[0],clsVerts[1],clsVerts[2]);
        const BridgeTex* texs=bridge_texs(base);
        std::vector<uint32_t> used;
        for(uint32_t ri=0;ri<pp.rangeCount;++ri)if(std::find(used.begin(),used.end(),ranges[ri].texId)==used.end())used.push_back(ranges[ri].texId);
        for(uint32_t id:used)if(id<h->texCount){const BridgeTex& t=texs[id];printf("particle tex[%u]: %ux%u fmt=%u bytes=%u pitch=%u rows=%u mips=%u\n",id,t.width,t.height,t.fmt,t.size,t.rowPitch,t.rows,t.mipCount);}
    }
    uint32_t flaggedDraws = 0;
    for (uint32_t n = 0; n < count; ++n) {
        const BridgeDraw& d = draws[n];
        if (!d.flags) continue;
        if (flaggedDraws++ < 80)
            printf("flagged[%u] draw=%u flags=0x%x tris=%u verts=%u vb=%u ib=%u tex=%u uv=%u stride=%u\n",
                   flaggedDraws-1, n, d.flags, d.primCount, d.numVertices, d.vbId, d.ibId,
                   d.texId, d.uvOffset, d.vbId < h->vbCount ? vbs[d.vbId].stride : 0u);
    }
    printf("flagged summary: draws=%u\n", flaggedDraws);
    const BridgeLight* lights = bridge_lights(base);
    for (uint32_t i = 0; i < h->lighting.lightCount && i < 12; ++i)
        printf("light[%u] pos=(%.2f %.2f %.2f) color=(%.3f %.3f %.3f) intensity=%.3f invR=%.6f space=%u\n",
               i, lights[i].origin[0], lights[i].origin[1], lights[i].origin[2],
               lights[i].color[0], lights[i].color[1], lights[i].color[2], lights[i].color[3],
               lights[i].invRadius, lights[i].space);
    for (size_t i = 0; i < found.size() && i < 80; ++i) {
        const Finding& f=found[i];
        printf("draw=%u tris=%u vb=%u ib=%u start=%u min=%u nV=%u base=%d off=%u pos=%u stride=%u outDecl=%u bad=%u nonfinite=%u maxEdge=%.2f diag=%.2f\n",
               f.draw,f.tris,f.d.vbId,f.d.ibId,f.d.startIndex,f.d.minIndex,f.d.numVertices,(int32_t)f.d.baseVertex,
               f.d.vbOffset,f.d.posOffset,vbs[f.d.vbId].stride,f.outsideDeclared,f.outsideBuffer,f.nonFinite,f.maxEdge,f.diagonal);
    }
    UnmapViewOfFile(base); CloseHandle(map); return 0;
}
