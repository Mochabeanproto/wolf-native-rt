#define SHARC_UPDATE 0
#define SHARC_ENABLE_CACHE_RESAMPLING 0
#define SHARC_PROPAGATION_DEPTH 4
#define SHARC_MATERIAL_DEMODULATION 0
#include "../third_party/SHARC/include/SharcCommon.h"

RWStructuredBuffer<uint64_t>              gSharcHash     : register(u1);
RWStructuredBuffer<SharcAccumulationData> gSharcAccum    : register(u2);
RWStructuredBuffer<SharcPackedData>       gSharcResolved : register(u3);

cbuffer SharcConstants : register(b3)
{
    float4 gSharcCameraScale;
    uint4  gSharcInfo;
    float4 gSharcPrevCamera;
    uint4  gSharcResolveInfo;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    SharcParameters p;
    p.hashGridParameters.cameraPosition = gSharcCameraScale.xyz;
    p.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    p.hashGridParameters.sceneScale = gSharcCameraScale.w;
    p.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    p.hashGridData.capacity = gSharcInfo.x;
    p.hashGridData.hashEntriesBuffer = gSharcHash;
    p.radianceScale = 1000.0f;
    p.accumulationBuffer = gSharcAccum;
    p.resolvedBuffer = gSharcResolved;

    SharcResolveParameters r;
    r.cameraPositionPrev = gSharcPrevCamera.xyz;
    r.accumulationFrameNum = gSharcResolveInfo.x;
    r.responsiveFrameNum = gSharcResolveInfo.y;
    r.staleFrameNumMax = gSharcResolveInfo.z;
    r.frameIndex = gSharcInfo.z;
    SharcResolveEntry(dispatchThreadId.x, p, r);
}
