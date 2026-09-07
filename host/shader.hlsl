// Path tracer feeding DLSS Ray Reconstruction. The primary hit emits a G-buffer
// (linear-HDR noisy color, view depth, motion vectors, world normal, diffuse +
// specular albedo, roughness); RR denoises + anti-aliases it. Multi-bounce diffuse
// GI lit by the sun (NEE + ray-traced shadow) and the sky (miss). World space,
// SM 6.6 bindless. No temporal accumulation here -- RR owns that now.

RaytracingAccelerationStructure Scene : register(t0);
RWTexture2D<float4>              Output : register(u0);   // heap[0]: linear HDR color
SamplerState                    gSamp  : register(s0);

// NVIDIA SHARC world-space radiance cache.  The update and query entry points
// share one library; Resolve is a small compute pass in sharc_resolve.hlsl.
#define SHARC_UPDATE 1
#define SHARC_ENABLE_CACHE_RESAMPLING 0
#define SHARC_PROPAGATION_DEPTH 4
#define SHARC_MATERIAL_DEMODULATION 0
#include "../third_party/SHARC/include/SharcCommon.h"
RWStructuredBuffer<uint64_t>              gSharcHash     : register(u1);
RWStructuredBuffer<SharcAccumulationData> gSharcAccum    : register(u2);
RWStructuredBuffer<SharcPackedData>       gSharcResolved : register(u3);

cbuffer Cam : register(b0)
{
    float4 gVW0;      // xyz = viewToWorld row0, w = proj xScale
    float4 gVW1;      // xyz = row1,            w = proj yScale
    float4 gVW2;      // xyz = row2,            w = unused
    float4 gCamPos;   // xyz = eye,             w = width
    float4 gSun;      // xyz = world sun dir,   w = height
    float4 gFrame;    // x = frame seed, y = unused, z = jitterX(px), w = jitterY(px)
    row_major float4x4 gPrevRelWC;   // last frame's ROTATION-ONLY world->clip (eye-relative, for MVs)
    float4 gCamDelta; // xyz = curEye - prevEye (world units), w unused
    float4 gSunColor; // xyz = sun radiance (game's real sun color * intensity), w unused
    float4 gSkyColor; // captured sky color/exposure scale
    uint4  gLightInfo;// x=light SRV, y=count, z=sky SRV (0=fallback), w=0 latlong/1 cube/2 screen
};

// Multi-layer sky (separate CBV -- the b0 root-constant budget is full).
cbuffer SkyLayers : register(b2)
{
    float4 gSkyL1x0;  // cloud layer 1 texture affine row0 (xyz used); uv' = M*(uv,1)
    float4 gSkyL1x1;  // cloud layer 1 texture affine row1
    float4 gSkyL2x0;  // cloud layer 2 texture affine row0
    float4 gSkyL2x1;  // cloud layer 2 texture affine row1
    uint4  gSkyLayers;// x=L1 colorSRV, y=L1 alphaSRV, z=L2 colorSRV, w=L2 alphaSRV (0=absent)
    float4 gSkyUu[3]; // quadratic u coeffs [0..9] (2 pad): basis [1,x,y,z, xx,yy,zz,xy, yz,zx]
    float4 gSkyVv[3]; // quadratic v coeffs [0..9]
    uint4  gSkyMisc;  // x = skyUV valid(1/0)
};

cbuffer SharcConstants : register(b3)
{
    float4 gSharcCameraScale; // xyz camera, w scene scale
    uint4  gSharcInfo;        // x capacity, y enabled, z frame, w sparse block size
    float4 gSharcPrevCamera;
    uint4  gSharcResolveInfo; // x history frames, y responsive frames, z stale frames
};

cbuffer RestirConstants : register(b4)
{
    uint4 gRestirSlots; // current reservoir, previous reservoir, current surface, previous surface
    uint4 gRestirInfo;  // render width, height, max temporal M, fresh candidate count (0=fallback)
};

struct RestirReservoir
{
    uint  lightIndex;
    float weightSum;
    float target;
    uint  M;
};

SharcParameters make_sharc_parameters()
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
    return p;
}

SharcHitData make_sharc_hit(float3 positionWorld, float3 normalWorld)
{
    SharcHitData h;
    h.positionWorld = positionWorld;
    h.normalWorld = normalWorld;
    return h;
}

ByteAddressBuffer Indices : register(t1);
ByteAddressBuffer Verts   : register(t2);
cbuffer Geo : register(b1)
{
    uint gStride, gIndexStride, gStartIndex, gBaseVertex, gTexHeapIndex, gUvOffset, gPosOffset;
    uint gSpecHeapIndex, gNormalHeapIndex;
    uint gEmissiveHeapIndex;                 // high bit = multiply sampled RGB by alpha
    uint gDepthHeapIndex;
    uint gGeoPad;
    float4 gSpecParams;                       // $vSpecularPowerIntensityBiasScale (power,intensity,bias,scale)
    float4 gParallaxParams;                   // $vParallaxScaleBias (scale,bias,...)
};

struct Payload       { float3 albedo; float3 normal; float3 geomNormal; float t; float emissive; float3 spec; float rough; float3 emission;
                       float3 partAccum;       // additive fire/spark emission gathered along the ray
                       float  smokeTrans;      // product transmittance through volumetric smoke (order-independent)
                       float3 smokeColorSum;   // sum of scene-lit smoke color * alpha
                       float  smokeAlphaSum; }; // sum of smoke alpha (for the alpha-weighted average color)
struct ShadowPayload { float  vis; };
struct LocalLight
{
    float4 origin;
    float4 color;       // rgb radiance, w intensity
    float  invRadius;
    uint3  _pad;
};

static const int   NSAMP      = 1;     // indirect GI samples/pixel/frame (RR denoises 1spp temporally)
static const int   IND_BOUNCES= 3;     // indirect bounce depth
static const float  FIREFLY   = 8.0f;  // per-sample radiance clamp (kills pulsing outliers)
#define SUN_COL (gSunColor.rgb)        // game's real sun radiance (from $vGlobalLightColor)
static const float SKY_EMISSIVE = 2.5f; // brightness of the real atmos sky-dome texture as an emitter
static const float LOCAL_LIGHT_SCALE = 0.12f;  // retain legacy color below Reinhard's whitening shoulder
static const float EMISSIVE_SCALE = 50.0f;     // diagnostic high-output emissive energy before Reinhard
static const float PARTICLE_EMISSIVE_SCALE = 3.0f;  // additive particle billboards (fire/sparks): rgb*alpha emission
static const float PARTICLE_SMOKE_DENSITY = 0.55f;  // optical depth of one smoke card; overlaps build real density
#define DEBUG_SPEC 0                     // PHASE A: 1 = show captured SpecularMap raw (verify), 0 = normal
static const float3 SKY_COL = float3(0.45f, 0.62f, 0.85f) * 1.1f;   // sky dome / fill

uint  pcg(inout uint s) { s = s*747796405u + 2891336453u; uint w = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u; return (w >> 22) ^ w; }
float rndf(inout uint s) { return (pcg(s) >> 8) * (1.0f / 16777216.0f); }
float3 cosHemi(float3 n, inout uint s)
{
    float u1 = rndf(s), u2 = rndf(s);
    float r = sqrt(u1), phi = 6.2831853f * u2;
    float3 up = abs(n.z) < 0.999f ? float3(0,0,1) : float3(1,0,0);
    float3 tx = normalize(cross(up, n)); float3 ty = cross(n, tx);
    return normalize(tx*(r*cos(phi)) + ty*(r*sin(phi)) + n*sqrt(max(0.0f, 1.0f - u1)));
}

// GGX half-vector sample (importance sampling the NDF) around normal n.
float3 sampleGGX(float3 n, float rough, inout uint s)
{
    float a = rough * rough;
    float u1 = rndf(s), u2 = rndf(s);
    float phi = 6.2831853f * u1;
    float cosT = sqrt((1.0f - u2) / (1.0f + (a*a - 1.0f) * u2));
    float sinT = sqrt(max(0.0f, 1.0f - cosT*cosT));
    float3 h = float3(sinT*cos(phi), sinT*sin(phi), cosT);
    float3 up = abs(n.z) < 0.999f ? float3(0,0,1) : float3(1,0,0);
    float3 tx = normalize(cross(up, n)); float3 ty = cross(n, tx);
    return normalize(tx*h.x + ty*h.y + n*h.z);
}
// Fresnel-Schlick.
float3 fresnelS(float3 F0, float cosT) { float m = saturate(1.0f - cosT); float m2 = m*m; return F0 + (1.0f - F0) * (m2*m2*m); }

// Material textures currently use sRGB SRVs. Albedo wants that hardware decode,
// but tangent-space normals are data; invert the decode to recover their stored
// UNORM channels before remapping [0,1] to [-1,1].
float3 linear_to_srgb(float3 c)
{
    return lerp(12.92f*c, 1.055f*pow(max(c, 0.0f), 1.0f/2.4f)-0.055f, step(0.0031308f, c));
}

// EON: A Practical Energy-Preserving Rough Diffuse BRDF (Portsmouth, Kutz, Hill 2024,
// JCGT 14(1)). Energy- and albedo-preserving Oren-Nayar. Transcribed from Listing 1.
// Returns the BRDF value (NOT * NdotL); multiply by NdotL * light radiance.
static const float EON_RCPPI = 1.0f / 3.14159265f;
static const float EON_C1 = 0.5f - 2.0f / (3.0f * 3.14159265f);
static const float EON_C2 = 2.0f / 3.0f - 28.0f / (15.0f * 3.14159265f);
float E_FON_approx(float mu, float r)
{
    float mucomp = 1.0f - mu;
    const float g1 = 0.0571085289f, g2 = 0.491881867f, g3 = -0.332181442f, g4 = 0.0714429953f;
    float GoverPi = mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)));
    return (1.0f + r * GoverPi) / (1.0f + EON_C1 * r);
}
// rho = albedo, r = roughness[0,1]; N,L,V world-space unit vectors (frame-invariant use).
float3 f_EON(float3 rho, float r, float3 N, float3 L, float3 V)
{
    float mu_i = saturate(dot(N, L)), mu_o = saturate(dot(N, V));
    float s = dot(L, V) - mu_i * mu_o;                                   // QON s term
    float sovertF = s > 0.0f ? s / max(mu_i, mu_o) : s;                  // FON s/t
    float AF = 1.0f / (1.0f + EON_C1 * r);
    float3 f_ss = (rho * EON_RCPPI) * AF * (1.0f + r * sovertF);         // single scatter
    float EFo = E_FON_approx(mu_o, r), EFi = E_FON_approx(mu_i, r);
    float avgEF = AF * (1.0f + EON_C2 * r);
    float3 rho_ms = (rho * rho) * avgEF / (1.0f - rho * (1.0f - avgEF));
    const float eps = 1.0e-7f;
    float3 f_ms = (rho_ms * EON_RCPPI) * max(eps, 1.0f - EFo) * max(eps, 1.0f - EFi) / max(eps, 1.0f - avgEF);
    return f_ss + f_ms;
}

// Hammon diffuse (GDC 2017, "PBR Diffuse Lighting for GGX+Smith Microsurfaces"):
// roughness-aware diffuse that's energy-consistent with our GGX specular. Returns the
// diffuse BRDF*NdotL; multiply by light radiance. rough = linear roughness, alpha used
// directly for the smooth<->rough blend + multi-scatter term. The *PI keeps our
// light convention (existing Lambert path omits the 1/pi, folding it into the sun).
float3 hammonDiffuse(float3 albedo, float3 F0, float rough, float NdotL, float NdotV, float VdotL, float NdotH)
{
    float facing = 0.5f + 0.5f * VdotL;
    float roughSurf = facing * (0.9f - 0.4f * facing) * (rsqrt(NdotH*NdotH + 0.01f) + 2.0f);
    float f0 = max(F0.r, max(F0.g, F0.b));
    float ecf = 1.0f - (4.0f*sqrt(f0) + 5.0f*f0*f0) * (1.0f/9.0f);
    float3 fresnelL = 1.0f - fresnelS(F0, NdotL);
    float3 fresnelV = 1.0f - fresnelS(F0, NdotV);
    float3 smoothSurf = (fresnelL * fresnelV) / max(ecf, 1e-3f);
    float3 single = lerp(smoothSurf, roughSurf.xxx, rough) * (1.0f / 3.14159265f);
    float  multi  = 0.1159f * rough;
    return max(0.0f, NdotL) * (albedo * multi + single) * 3.14159265f;   // *PI: match existing (no-1/pi) sun scale
}
// Smith GGX geometry (height-correlated, visibility term ratio ~ G2/G1 for the IS weight).
float smithG1(float ndv, float a) { float a2 = a*a; return 2.0f*ndv / (ndv + sqrt(a2 + (1.0f-a2)*ndv*ndv)); }

static const float PI_    = 3.14159265f;
static const uint  ENVW   = 2048u;
static const uint  ENVH   = 1024u;

// Ray direction -> lat-long UV (idTech Z-up: dir.z is up).
uint2 dir_to_env(float3 d)
{
    d = normalize(d);
    float2 uv = float2(atan2(d.y, d.x) * (0.5f/PI_) + 0.5f, acos(clamp(d.z, -1.0f, 1.0f)) / PI_);
    uint2 t = uint2(uv * float2(ENVW, ENVH));
    return uint2(min(t.x, ENVW-1u), min(t.y, ENVH-1u));
}

// Analytic sky dome driven by the game's REAL sun direction/colour. Gives a
// horizon->zenith gradient, warm horizon haze, and a sun glow + disk. Works in
// every direction (reflections, GI, looking around) at zero capture cost.
float3 analytic_sky(float3 dir, float3 sunDir)
{
    dir = normalize(dir);
    float z = dir.z;                                  // idTech Z-up
    float3 zenith  = float3(0.10f, 0.18f, 0.34f);
    float3 horizon = float3(0.42f, 0.48f, 0.58f);
    float3 sky = lerp(horizon, zenith, pow(saturate(z), 0.55f));
    sky = lerp(float3(0.16f, 0.15f, 0.14f), sky, saturate(z*3.0f + 0.5f));   // ground/haze below horizon
    float cosT = dot(dir, sunDir);
    float glow = pow(saturate(cosT), 6.0f) * 0.25f + pow(saturate(cosT), 350.0f) * 3.0f;
    float disk = smoothstep(0.9994f, 0.9998f, cosT) * 8.0f;
    sky += gSunColor.rgb * (glow + disk) * 0.30f;
    return max(0.0f, sky);
}

// Sky = analytic base everywhere, with the game's real framebuffer sky overlaid.
// Primary sky rays use the live screen sky (sharp) and paint it into a persistent
// lat-long env map by direction; secondary rays read that accumulated real sky
// where we've already seen it, and fall back to the analytic base elsewhere.
float3 sky_radiance(float3 dir, float2 screenUV)
{
    float3 sunDir = normalize(gSun.xyz);
    float3 base = analytic_sky(dir, sunDir);
    uint envIdx  = asuint(gSkyColor.w);   // sky env-map UAV heap index (0 = none)
    uint skyHeap = gLightInfo.z;
    bool primary = screenUV.x >= 0.0f;

    // Primary rays: overlay the live screen-space sky (sharp) + paint it into the env map.
    if (primary && gLightInfo.w == 2u && skyHeap != 0u) {
        Texture2D<float4> ss = ResourceDescriptorHeap[skyHeap];
        float3 c = ss.SampleLevel(gSamp, saturate(screenUV), 0).rgb * gSkyColor.rgb;
        if (envIdx != 0u) {
            RWTexture2D<float4> env = ResourceDescriptorHeap[envIdx];
            uint2 tx = dir_to_env(dir);
            float4 prev = env[tx];
            env[tx] = float4(prev.a > 0.5f ? lerp(prev.rgb, c, 0.08f) : c, 1.0f);
        }
        return max(0.0f, c);
    }

    // Secondary rays: read the accumulated real sky by direction where covered.
    if (envIdx != 0u) {
        RWTexture2D<float4> env = ResourceDescriptorHeap[envIdx];
        float4 e = env[dir_to_env(dir)];
        if (e.a > 0.5f) return max(0.0f, e.rgb);
    }
    return base;
}

float3 eval_local_unshadowed(LocalLight l, float3 pos, float3 n, float3 albedo, float rough, float3 V)
{
    float3 toLight = l.origin.xyz - pos;
    float dist2 = dot(toLight, toLight);
    if (dist2 < 1e-4f) return 0;
    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    float ndl = saturate(dot(n, wi));
    if (ndl <= 0) return 0;

    float attenuation = 1.0f;
    if (l.invRadius > 1e-7f) {
        float r = dist * l.invRadius;
        if (r >= 1.0f) return 0;
        attenuation = saturate(1.0f - r*r);
        attenuation *= attenuation;
    }

    // EON diffuse (like the sun/GI); f_EON returns the BRDF so * NdotL * PI (our sun-scale convention).
    return f_EON(albedo, rough, n, wi, V) * (ndl * 3.14159265f) * l.color.rgb * max(l.color.w, 0.0f)
           * attenuation * LOCAL_LIGHT_SCALE;
}

float local_visibility(LocalLight l, float3 pos, float3 n)
{
    float3 toLight = l.origin.xyz - pos;
    float dist = length(toLight);
    if (dist < 0.01f) return 0.0f;
    RayDesc shadow;
    shadow.Origin = pos + n * (0.5f + dist * 0.0001f);
    shadow.Direction = toLight / dist;
    shadow.TMin = 0.05f;
    shadow.TMax = max(0.05f, dist - 0.1f);
    ShadowPayload sp; sp.vis = 0.0f;
    TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0x01, 0, 0, 1, shadow, sp);
    return sp.vis;
}

float local_visibility_area(LocalLight l, float3 pos, float3 n, inout uint seed)
{
    float3 toCenter = l.origin.xyz - pos;
    float centerDistance = length(toCenter);
    if (centerDistance < 0.01f) return 0.0f;
    float3 axis = toCenter / centerDistance;

    // The game exposes an influence radius but not the physical bulb dimensions.
    // Use a conservative fraction of that radius as the emitter size, clamped to
    // plausible Wolf world-unit sizes. Sampling its projected disk creates the
    // correct blocker/receiver-dependent penumbra instead of an arbitrary blur.
    float influenceRadius = l.invRadius > 1e-7f ? rcp(l.invRadius) : 200.0f;
    float emitterRadius = clamp(influenceRadius * 0.025f, 2.0f, 14.0f);
    float diskR = sqrt(rndf(seed)) * emitterRadius;
    float phi = 6.2831853f * rndf(seed);
    float3 helper = abs(axis.z) < 0.95f ? float3(0,0,1) : float3(0,1,0);
    float3 tangent = normalize(cross(helper, axis));
    float3 bitangent = cross(axis, tangent);
    float3 sampledLight = l.origin.xyz + diskR * (cos(phi)*tangent + sin(phi)*bitangent);

    float3 toLight = sampledLight - pos;
    float dist = length(toLight);
    RayDesc shadow;
    shadow.Origin = pos + n * (0.5f + dist * 0.0001f);
    shadow.Direction = toLight / max(dist, 0.01f);
    shadow.TMin = 0.05f;
    shadow.TMax = max(0.05f, dist - 0.1f);
    ShadowPayload sp; sp.vis = 0.0f;
    TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0x01, 0, 0, 1, shadow, sp);
    return sp.vis;
}

float3 eval_local(LocalLight l, float3 pos, float3 n, float3 albedo, float rough, float3 V)
{
    return eval_local_unshadowed(l, pos, n, albedo, rough, V) * local_visibility(l, pos, n);
}

// Primary direct lighting is deterministic. Randomly selecting one of a handful
// of bright lamps made them visibly blink before Ray Reconstruction converged.
float3 direct_local_all(float3 pos, float3 n, float3 albedo, float rough, float3 V)
{
    uint count = gLightInfo.y;
    if (count == 0) return 0;
    StructuredBuffer<LocalLight> lights = ResourceDescriptorHeap[gLightInfo.x];
    float3 sum = 0;
    for (uint i = 0; i < count; ++i) sum += eval_local(lights[i], pos, n, albedo, rough, V);
    return sum;
}

float2 oct_encode(float3 n)
{
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6f);
    float2 e = n.xy;
    if (n.z < 0.0f) e = (1.0f - abs(e.yx)) * float2(e.x >= 0.0f ? 1.0f : -1.0f,
                                                        e.y >= 0.0f ? 1.0f : -1.0f);
    return e;
}
float3 oct_decode(float2 e)
{
    float3 n = float3(e, 1.0f - abs(e.x) - abs(e.y));
    if (n.z < 0.0f) n.xy = (1.0f - abs(n.yx)) * float2(n.x >= 0.0f ? 1.0f : -1.0f,
                                                        n.y >= 0.0f ? 1.0f : -1.0f);
    return normalize(n);
}
uint pack_normal(float3 n)
{
    uint2 q = (uint2)round(saturate(oct_encode(normalize(n)) * 0.5f + 0.5f) * 65535.0f);
    return q.x | (q.y << 16);
}
float3 unpack_normal(uint p)
{
    float2 e = float2(p & 0xffffu, p >> 16) * (2.0f / 65535.0f) - 1.0f;
    return oct_decode(e);
}
float restir_target(float3 c) { return max(dot(max(c, 0.0f), float3(0.2126f, 0.7152f, 0.0722f)), 0.0f); }

void reservoir_add(inout RestirReservoir r, uint lightIndex, float target, float weight,
                   uint candidateM, inout uint seed)
{
    if (weight <= 0.0f || candidateM == 0u || !isfinite(weight)) return;
    float nextSum = r.weightSum + weight;
    if (rndf(seed) * nextSum < weight) { r.lightIndex = lightIndex; r.target = target; }
    r.weightSum = nextSum;
    r.M += candidateM;
}

bool restir_surface_valid(uint4 packed, float3 pos, float3 n, float maxDistance)
{
    float3 oldPos = asfloat(packed.xyz);
    float3 oldN = unpack_normal(packed.w);
    return all(isfinite(oldPos)) && distance(oldPos, pos) <= maxDistance && dot(oldN, n) >= 0.85f;
}

// ReSTIR DI for the primary local-light term. Fresh uniform candidates are
// combined with the motion-reprojected reservoir and four validated neighboring
// reservoirs from the previous frame. Only the selected sample traces visibility.
float3 direct_local_restir(uint2 px, float2 motion, float3 pos, float3 n,
                           float3 albedo, float rough, float3 V, inout uint seed)
{
    uint count = gLightInfo.y;
    if (count == 0u) return 0.0f;
    if (gRestirInfo.w == 0u) return direct_local_all(pos, n, albedo, rough, V);

    uint width = gRestirInfo.x, height = gRestirInfo.y;
    uint linearIndex = px.y * width + px.x;
    StructuredBuffer<LocalLight> lights = ResourceDescriptorHeap[gLightInfo.x];
    RWStructuredBuffer<RestirReservoir> prevReservoirs = ResourceDescriptorHeap[gRestirSlots.y];
    RWStructuredBuffer<uint4> prevSurfaces = ResourceDescriptorHeap[gRestirSlots.w];
    RWStructuredBuffer<RestirReservoir> curReservoirs = ResourceDescriptorHeap[gRestirSlots.x];
    RWStructuredBuffer<uint4> curSurfaces = ResourceDescriptorHeap[gRestirSlots.z];

    RestirReservoir r; r.lightIndex = 0xffffffffu; r.weightSum = 0.0f; r.target = 0.0f; r.M = 0u;
    [loop] for (uint c = 0u; c < gRestirInfo.w; ++c) {
        uint li = min((uint)(rndf(seed) * count), count - 1u);
        float3 contribution = eval_local_unshadowed(lights[li], pos, n, albedo, rough, V);
        float target = restir_target(contribution);
        reservoir_add(r, li, target, target * count, 1u, seed); // uniform proposal pdf = 1/count
    }

    float validationDistance = max(6.0f, distance(pos, gCamPos.xyz) * 0.015f);
    int2 previousPixel = int2(round(float2(px) + motion));
    int2 offsets[4] = { int2(-4,0), int2(4,0), int2(0,-4), int2(0,4) };
    [unroll] for (uint reuse = 0u; reuse < 5u; ++reuse) {
        int2 q = (reuse == 0u) ? previousPixel : int2(px) + offsets[reuse - 1u];
        if (any(q < 0) || q.x >= (int)width || q.y >= (int)height) continue;
        uint qi = (uint)q.y * width + (uint)q.x;
        RestirReservoir old = prevReservoirs[qi];
        if (old.M == 0u || old.lightIndex >= count || old.target <= 1e-8f) continue;
        if (!restir_surface_valid(prevSurfaces[qi], pos, n, validationDistance)) continue;
        float3 contribution = eval_local_unshadowed(lights[old.lightIndex], pos, n, albedo, rough, V);
        float target = restir_target(contribution);
        uint reuseM = min(old.M, max(gRestirInfo.z, 1u));
        float historyScale = (float)reuseM / max((float)old.M, 1.0f);
        float weight = old.weightSum * historyScale * target / max(old.target, 1e-8f);
        reservoir_add(r, old.lightIndex, target, weight, reuseM, seed);
    }

    if (r.lightIndex == 0xffffffffu || r.M == 0u || r.target <= 1e-8f) {
        curReservoirs[linearIndex] = r;
        curSurfaces[linearIndex] = uint4(asuint(pos), pack_normal(n));
        return 0.0f;
    }
    float normalization = r.weightSum / max((float)r.M * r.target, 1e-8f);
    normalization = min(normalization, 64.0f); // contain disocclusion/firefly outliers
    float3 selected = eval_local_unshadowed(lights[r.lightIndex], pos, n, albedo, rough, V);
    float visibility = local_visibility_area(lights[r.lightIndex], pos, n, seed);
    curReservoirs[linearIndex] = r;
    curSurfaces[linearIndex] = uint4(asuint(pos), pack_normal(n));
    return selected * (visibility * normalization);
}

// Secondary bounces retain one-light sampling to bound their shadow-ray cost.
float3 direct_local_sample(float3 pos, float3 n, float3 albedo, float rough, float3 V, inout uint seed)
{
    uint count = gLightInfo.y;
    if (count == 0) return 0;
    StructuredBuffer<LocalLight> lights = ResourceDescriptorHeap[gLightInfo.x];
    uint index = min((uint)(rndf(seed) * count), count - 1);
    float3 unshadowed = eval_local_unshadowed(lights[index], pos, n, albedo, rough, V);
    return unshadowed * local_visibility_area(lights[index], pos, n, seed) * count;
}

uint load_index(uint elem)
{
    if (gIndexStride == 2) { uint w = Indices.Load((elem*2)&~3u); return (elem&1)?(w>>16):(w&0xffff); }
    return Indices.Load(elem*4);
}
float3 load_pos(uint idx) { return asfloat(Verts.Load3((gBaseVertex + idx) * gStride + gPosOffset)); }
float2 load_uv (uint idx) { return asfloat(Verts.Load2((gBaseVertex + idx) * gStride + gUvOffset)); }
float4 load_particle_color(uint idx)
{
    // BridgePartVert stores RGBA8 as 0xAABBGGRR after pos.xyz + uv.xy.
    uint c = Verts.Load((gBaseVertex + idx) * gStride + 20u);
    return float4(c & 255u, (c >> 8) & 255u, (c >> 16) & 255u, c >> 24) * (1.0f / 255.0f);
}
uint load_particle_tex(uint idx) { return Verts.Load((gBaseVertex + idx) * gStride + 24u); }
uint load_particle_class(uint idx) { return Verts.Load((gBaseVertex + idx) * gStride + 28u); }

// Sparse SHARC update: one temporally rotated pixel in each block.  This pass
// writes no image data; it traces representative diffuse paths and propagates
// the radiance seen at later vertices back through the cache.
[shader("raygeneration")]
void SharcUpdateRayGen()
{
    if (gSharcInfo.y == 0u) return;
    uint block = max(gSharcInfo.w, 1u);
    uint2 cell = DispatchRaysIndex().xy;
    uint phase = gSharcInfo.z;
    uint2 ofs = uint2((phase + cell.y * 3u) % block, (phase * 3u + cell.x * 2u) % block);
    uint2 px = cell * block + ofs;
    uint2 dim = uint2((uint)gCamPos.w, (uint)gSun.w);
    if (any(px >= dim)) return;

    uint seed = (px.x * 1973u + px.y * 9277u + phase * 26699u) | 1u;
    float2 uv = (float2(px) + 0.5f) / float2(dim);
    float2 ndc = uv * 2.0f - 1.0f; ndc.y = -ndc.y;
    float3 vd = float3(ndc.x / gVW0.w, ndc.y / gVW1.w, -1.0f);
    float3 dir = normalize(float3(dot(gVW0.xyz, vd), dot(gVW1.xyz, vd), dot(gVW2.xyz, vd)));
    float3 org = gCamPos.xyz;
    float3 sunDir = normalize(gSun.xyz);
    SharcParameters sharc = make_sharc_parameters();
    SharcState state; SharcInit(state);

    // Include the primary vertex with zero local radiance so its entry stores
    // indirect illumination only; direct lighting remains deterministic.
    for (int bounce = 0; bounce <= IND_BOUNCES; ++bounce) {
        Payload p; p.t = -1.0f;
        RayDesc ray; ray.Origin = org; ray.Direction = dir; ray.TMin = 0.05f; ray.TMax = 1e9f;
        TraceRay(Scene, RAY_FLAG_NONE, 0xFB, 0, 0, 0, ray, p);   // 0xFB: exclude particles (0x04) from SHARC
        if (p.t < 0.0f) { SharcUpdateMiss(sharc, state, sky_radiance(dir, float2(-1,-1))); break; }
        if (p.emissive > 0.5f) { SharcUpdateMiss(sharc, state, p.albedo); break; }

        float3 hit = org + p.t * dir;
        float3 ng = p.geomNormal, n = p.normal;
        if (dot(ng, -dir) < 0.0f) { ng = -ng; n = -n; }
        float eps = 0.5f + p.t * 0.003f;
        float3 directHere = p.emission;
        if (bounce > 0) {
            RayDesc sr; sr.Origin = hit + ng*eps; sr.Direction = sunDir; sr.TMin = 0.1f; sr.TMax = 1e9f;
            ShadowPayload sp; sp.vis = 0.0f;
            TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0x01, 0, 0, 1, sr, sp);
            float3 V = -dir;
            directHere += f_EON(p.albedo, p.rough, n, sunDir, V) * saturate(dot(n, sunDir)) * 3.14159265f * SUN_COL * sp.vis;
            directHere += direct_local_sample(hit, n, p.albedo, p.rough, V, seed);
        } else {
            directHere = 0.0f;
        }
        if (!SharcUpdateHit(sharc, state, make_sharc_hit(hit, ng), directHere, rndf(seed))) break;

        float3 nextDir = cosHemi(n, seed);
        float3 segmentWeight = f_EON(p.albedo, p.rough, n, nextDir, -dir) * 3.14159265f;
        SharcSetThroughput(state, segmentWeight);
        org = hit + ng*eps;
        dir = nextDir;
    }
}

[shader("raygeneration")]
void RayGen()
{
    uint2  px   = DispatchRaysIndex().xy;
    float2 size = float2(gCamPos.w, gSun.w);
    uint   seed = (px.x * 1973u + px.y * 9277u + (uint)gFrame.x * 26699u) | 1u;

    // Default this frame's history record to invalid. Opaque primary hits replace
    // it below; sky/miss/emissive early-outs cannot leave stale reservoirs behind.
    uint restirLinear = px.y * gRestirInfo.x + px.x;
    RWStructuredBuffer<RestirReservoir> restirCurrent = ResourceDescriptorHeap[gRestirSlots.x];
    RWStructuredBuffer<uint4> restirSurfaceCurrent = ResourceDescriptorHeap[gRestirSlots.z];
    RestirReservoir invalidReservoir;
    invalidReservoir.lightIndex = 0xffffffffu; invalidReservoir.weightSum = 0.0f;
    invalidReservoir.target = 0.0f; invalidReservoir.M = 0u;
    restirCurrent[restirLinear] = invalidReservoir;
    restirSurfaceCurrent[restirLinear] = 0u;

    // Deterministic sub-pixel jitter (Halton, host-side). Sample at grid-jit so the
    // sample sits on the same side DLSS assumes for +InJitterOffset (matches how a
    // jittered *projection* shifts the sample opposite to its NDC offset).
    float2 jit = float2(gFrame.z, gFrame.w);
    float2 uv  = (float2(px) + 0.5f - jit) / size;
    float2 ndc = uv * 2.0f - 1.0f; ndc.y = -ndc.y;
    float3 vd  = float3(ndc.x / gVW0.w, ndc.y / gVW1.w, -1.0f);
    float3 eye = gCamPos.xyz;
    float3 dir = normalize(float3(dot(gVW0.xyz, vd), dot(gVW1.xyz, vd), dot(gVW2.xyz, vd)));
    float3 fwd = -normalize(float3(gVW0.z, gVW1.z, gVW2.z));   // view -Z axis, in world

    float3 sunDir = normalize(gSun.xyz);

    // G-buffer values from the PRIMARY hit.
    float  gbDepth = 1e7f;
    float2 gbMV    = 0;
    float3 gbNrm   = float3(0,0,1);
    float3 gbAlb   = 1;
    float3 gbSpec  = float3(0.04f,0.04f,0.04f);
    float  gbRough = 1.0f;
    float3 radiance = 0;

    // Primary ray (traced once; shared by all GI samples). Mask 0x03 = world + sky
    // dome, but NOT particles (0x04): particles are gathered by a separate depth-capped
    // ray below so walls correctly occlude them (any-hit has no traversal ordering).
    Payload p0; p0.t = -1; p0.partAccum = 0;
    RayDesc r0; r0.Origin = eye; r0.Direction = dir; r0.TMin = 0.05f; r0.TMax = 1e9f;
    TraceRay(Scene, RAY_FLAG_NONE, 0x03, 0, 0, 0, r0, p0);

    // Transparent particle billboards: gather them only up to the opaque hit distance
    // (or to infinity if the primary escaped to sky) so geometry occludes them.
    float3 partGather = 0;
    float smokeTrans = 1.0f;
    float3 smokeColor = 0.0f;
    {
        Payload pg; pg.t = -1; pg.partAccum = 0;
        pg.smokeTrans = 1.0f; pg.smokeColorSum = 0.0f; pg.smokeAlphaSum = 0.0f;
        RayDesc rg; rg.Origin = eye; rg.Direction = dir; rg.TMin = 0.05f;
        rg.TMax = (p0.t < 0) ? 1e9f : p0.t - 0.01f;
        // One compact particle geometry; texture/class identity lives per vertex.
        TraceRay(Scene, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0x04, 0, 0, 0, rg, pg);
        partGather = pg.partAccum;
        smokeTrans = saturate(pg.smokeTrans);
        smokeColor = (pg.smokeAlphaSum > 1e-5f) ? pg.smokeColorSum / pg.smokeAlphaSum : 0.0f;
    }

    if (p0.t < 0) {                                    // primary escaped to sky
        gbNrm = -dir; gbAlb = 1; gbDepth = 1.0f;       // far (HW depth [0,1])
        // Reproject in eye-relative space (world coords are huge -> float precision).
        float3 hitRelPrev = dir * 1e6f + gCamDelta.xyz;
        float4 pc = mul(float4(hitRelPrev, 1.0f), gPrevRelWC);
        if (pc.w > 1e-4f) { float2 pn = pc.xy/pc.w; float2 pv = float2(pn.x*0.5f+0.5f, -pn.y*0.5f+0.5f); gbMV = pv*size - (float2(px)+0.5f-jit); }
        radiance = sky_radiance(dir, uv);
    } else {
        float3 hit = eye + p0.t * dir;
        float3 hitRel = p0.t * dir;                    // eye-relative hit (precise, small)
        float3 ng = p0.geomNormal, n = p0.normal;
        if (dot(ng, -dir) < 0) { ng = -ng; n = -n; }
        // Hardware depth z/w (matches the projective viewToClip handed to DLSS):
        // ndc.z = -A + C/linearViewDepth, with A=gVW2.w, C=gCamDelta.w (proj[10]/[14]).
        float linDepth = max(0.01f, dot(hitRel, fwd));
        gbNrm = n; gbAlb = p0.albedo; gbDepth = -gVW2.w + gCamDelta.w / linDepth;
        gbSpec = p0.spec; gbRough = p0.rough;
        if (p0.emissive > 0.5f) {   // real sky dome hit directly: emit texture, no shading
            gbNrm = -dir; gbAlb = p0.albedo;
            float3 hrp = hitRel + gCamDelta.xyz; float4 pcS = mul(float4(hrp,1.0f), gPrevRelWC);
            if (pcS.w > 1e-4f) { float2 pn = pcS.xy/pcS.w; float2 pv = float2(pn.x*0.5f+0.5f, -pn.y*0.5f+0.5f); gbMV = pv*size - (float2(px)+0.5f-jit); }
            Output[px] = float4(p0.albedo * smokeTrans + smokeColor * (1.0f-smokeTrans) + partGather, 1.0f);
            RWTexture2D<float>  dS = ResourceDescriptorHeap[2]; dS[px] = gbDepth;
            RWTexture2D<float2> mS = ResourceDescriptorHeap[3]; mS[px] = (gFrame.y>0.5f)?float2(0,0):gbMV;
            RWTexture2D<float4> nS = ResourceDescriptorHeap[4]; nS[px] = float4(gbNrm,0);
            RWTexture2D<float4> aS = ResourceDescriptorHeap[5]; aS[px] = float4(gbAlb,1);
            RWTexture2D<float4> pS = ResourceDescriptorHeap[6]; pS[px] = float4(0.04f,0.04f,0.04f,1);
            RWTexture2D<float>  rS = ResourceDescriptorHeap[7]; rS[px] = 1.0f;
            return;
        }
        // Motion vector via eye-relative reprojection: shift into the previous frame's
        // eye frame (+camDelta), rotate/project by the previous rotation-only world->clip.
        float3 hitRelPrev = hitRel + gCamDelta.xyz;
        float4 pc = mul(float4(hitRelPrev, 1.0f), gPrevRelWC);
        if (pc.w > 1e-4f) {
            float2 pn = pc.xy/pc.w; float2 pv = float2(pn.x*0.5f+0.5f, -pn.y*0.5f+0.5f);
            // Motion vector = prev screen pos - the JITTERED sample pos this ray used
            // (px+0.5-jit). A static surface reprojects to its own sample -> MV = 0.
            gbMV = pv*size - (float2(px) + 0.5f - jit);
        }
        float eps = 0.5f + p0.t * 0.003f;

        // Direct sun at the primary hit (NEE + ray-traced shadow), computed once.
        RayDesc s; s.Origin = hit + ng*eps; s.Direction = sunDir; s.TMin = 0.1f; s.TMax = 1e9f;
        ShadowPayload sp; sp.vis = 0.0f;
        TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0x01, 0, 0, 1, s, sp);
        float3 Vd = -dir;
        // EON diffuse; f_EON returns the BRDF, so * NdotL * light. *PI keeps our existing
        // (no-1/pi) sun-scale convention.
        float3 direct = f_EON(p0.albedo, p0.rough, n, sunDir, Vd) * saturate(dot(n, sunDir)) * 3.14159265f * SUN_COL * sp.vis;
        direct += direct_local_restir(px, gbMV, hit, n, p0.albedo, p0.rough, Vd, seed);

        // Indirect GI: average NSAMP diffuse paths (lower variance -> RR doesn't boil).
        // GI is EON-consistent: cosine-sampled bounces weighted by the EON BRDF (tp *=
        // f_EON*PI, since the cosine pdf cancels), and NEE sun at each bounce uses EON too.
        float3 indirect = 0;
        for (int smp = 0; smp < NSAMP; ++smp) {
            uint  s2  = seed + smp * 0x9E3779B9u;
            float3 org = hit + ng*eps, d = cosHemi(n, s2), rad = 0;
            float3 tp = f_EON(p0.albedo, p0.rough, n, d, Vd) * 3.14159265f;   // primary surface BRDF weight
            for (int b = 0; b < IND_BOUNCES; ++b) {
                Payload p; p.t = -1;
                RayDesc rb; rb.Origin = org; rb.Direction = d; rb.TMin = 0.02f; rb.TMax = 1e9f;
                TraceRay(Scene, RAY_FLAG_NONE, 0xFB, 0, 0, 0, rb, p);   // 0xFB: GI bounce skips particles
                if (p.t < 0) { rad += tp * sky_radiance(d, float2(-1.0f, -1.0f)); break; }
                if (p.emissive > 0.5f) { rad += tp * p.albedo; break; }   // hit the real sky dome -> emit
                float3 h = org + p.t*d; float3 ngb = p.geomNormal, nb = p.normal;
                if (dot(ngb, -d) < 0) { ngb = -ngb; nb = -nb; }
                // Cache queries are restricted to secondary diffuse hits.  The
                // primary G-buffer/direct light stays exact and responsive.
                if (gSharcInfo.y != 0u) {
                    float3 cached;
                    if (SharcGetCachedRadiance(make_sharc_parameters(), make_sharc_hit(h, ngb), cached, true)) {
                        rad += tp * cached;
                        break;
                    }
                }
                rad += tp * p.emission;
                float e2 = 0.5f + p.t*0.003f; float3 woB = -d;
                RayDesc sr; sr.Origin = h + ngb*e2; sr.Direction = sunDir; sr.TMin = 0.1f; sr.TMax = 1e9f;
                ShadowPayload sp2; sp2.vis = 0.0f;
                TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0x01, 0, 0, 1, sr, sp2);
                rad += tp * f_EON(p.albedo, p.rough, nb, sunDir, woB) * saturate(dot(nb, sunDir)) * 3.14159265f * SUN_COL * sp2.vis;
                rad += tp * direct_local_sample(h, nb, p.albedo, p.rough, woB, s2);
                float3 dn = cosHemi(nb, s2);
                tp *= f_EON(p.albedo, p.rough, nb, dn, woB) * 3.14159265f;   // EON weight for the next bounce
                if (max(tp.x, max(tp.y, tp.z)) < 0.02f) break;
                org = h + ngb*e2; d = dn;
            }
            indirect += min(rad, FIREFLY);
        }
        float3 diffuseCol = direct + indirect / NSAMP;

        // --- Specular lobe: reflection ray (reflects real sky + world) + sun highlight ---
        float3 V = -dir;
        float3 F0s = p0.spec; float roughS = p0.rough;
        float3 spec = 0;
        // Only trace a reflection ray where it's worth it: glossy enough, or bright F0
        // (metal/water/glass). Matte dielectrics reflect ~nothing -> skip (big perf win).
        bool doReflect = (roughS < 0.55f) || (max(F0s.r, max(F0s.g, F0s.b)) > 0.08f);
        {
            uint ss = seed ^ 0x68bc21ebu;
            float3 H = sampleGGX(n, roughS, ss);
            float3 R = reflect(dir, H);
            if (doReflect && dot(R, n) > 0.0f) {
                Payload pr; pr.t = -1;
                RayDesc rr; rr.Origin = hit + ng*eps; rr.Direction = R; rr.TMin = 0.02f; rr.TMax = 1e9f;
                TraceRay(Scene, RAY_FLAG_NONE, 0xFB, 0, 0, 0, rr, pr);   // 0xFB: reflection skips particles
                float3 Lr;
                if (pr.t < 0)                 Lr = sky_radiance(R, float2(-1.0f,-1.0f));
                else if (pr.emissive > 0.5f)  Lr = pr.albedo;                       // real sky dome
                else {                                                              // cheap 1-bounce shade of reflected surface
                    float3 rh = hit + pr.t*R; float3 rng = pr.geomNormal, rn = pr.normal;
                    if (dot(rng,-R) < 0) { rng = -rng; rn = -rn; }
                    float re = 0.5f + pr.t*0.003f;
                    RayDesc rs; rs.Origin = rh + rng*re; rs.Direction = sunDir; rs.TMin = 0.1f; rs.TMax = 1e9f;
                    ShadowPayload rsp; rsp.vis = 0.0f;
                    TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0x01, 0, 0, 1, rs, rsp);
                    Lr = pr.emission + pr.albedo * (SUN_COL * saturate(dot(rn, sunDir)) * rsp.vis + sky_radiance(rn, float2(-1.0f,-1.0f)) * 0.3f);
                    Lr += direct_local_all(rh, rn, pr.albedo, pr.rough, -R);
                }
                float3 F = fresnelS(F0s, saturate(dot(V, H)));
                spec = F * Lr;
            }
            // Direct sun specular highlight (analytic GGX).
            float3 Hs = normalize(V + sunDir);
            float ndl = saturate(dot(n, sunDir)), ndv = saturate(dot(n, V)), ndh = saturate(dot(n, Hs));
            if (ndl > 0.0f && sp.vis > 0.0f) {
                float a = roughS*roughS, a2 = a*a;
                float dd = ndh*ndh*(a2 - 1.0f) + 1.0f;
                float D = a2 / max(PI_ * dd * dd, 1e-6f);
                float G = smithG1(ndv, a) * smithG1(ndl, a) / max(4.0f*ndv*ndl, 1e-4f);
                float3 F = fresnelS(F0s, saturate(dot(V, Hs)));
                spec += min(D * G, 40.0f) * F * SUN_COL * ndl * sp.vis;
            }
        }
        // Energy: metallic/high-F0 surfaces keep less diffuse.
        float kd = 1.0f - saturate(max(F0s.r, max(F0s.g, F0s.b)));
        radiance = p0.emission + diffuseCol * kd + spec;
    }

    // Emit color + G-buffer for Ray Reconstruction (bindless heap slots).
    if (gFrame.y > 0.5f) gbMV = 0;                     // WOLF_MV0 diagnostic: force zero motion
    // Beer-Lambert smoke is order-independent, so the any-hit traversal order
    // cannot make cards pop. Additive fire remains emission on top.
    radiance = radiance * smokeTrans + smokeColor * (1.0f-smokeTrans) + partGather;
    Output[px] = float4(radiance, 1.0f);
    RWTexture2D<float>  gDepth = ResourceDescriptorHeap[2]; gDepth[px] = gbDepth;
    RWTexture2D<float2> gMV    = ResourceDescriptorHeap[3]; gMV[px]    = gbMV;
    RWTexture2D<float4> gNrm   = ResourceDescriptorHeap[4]; gNrm[px]   = float4(gbNrm, 0);
    RWTexture2D<float4> gAlb   = ResourceDescriptorHeap[5]; gAlb[px]   = float4(gbAlb, 1);
    RWTexture2D<float4> gSpec  = ResourceDescriptorHeap[6]; gSpec[px]  = float4(gbSpec, 1);
    RWTexture2D<float>  gRough = ResourceDescriptorHeap[7]; gRough[px] = gbRough;
}

[shader("miss")] void Miss(inout Payload p)             { p.t = -1; }
[shader("miss")] void ShadowMiss(inout ShadowPayload s) { s.vis = 1.0f; }

// Composite the scrolling cloud layers "over" the base sky color, using the base
// dome's UV (all sky layers map the same hemisphere) transformed by each layer's
// own texture affine (which rotates over time -> animated clouds).
float3 composite_sky_layers(float3 baseCol, float2 uv0, float2 ddx, float2 ddy)
{
    float3 c = baseCol;
    if (gSkyLayers.x != 0) {
        float2 uv = float2(dot(gSkyL1x0.xyz, float3(uv0,1)), dot(gSkyL1x1.xyz, float3(uv0,1)));
        Texture2D col = ResourceDescriptorHeap[gSkyLayers.x];
        float4 s = col.SampleGrad(gSamp, uv, ddx, ddy);
        float a = s.a;
        if (gSkyLayers.y != 0) { Texture2D alp = ResourceDescriptorHeap[gSkyLayers.y]; a = alp.SampleGrad(gSamp, uv, ddx, ddy).r; }
        c = lerp(c, s.rgb, saturate(a));
    }
    if (gSkyLayers.z != 0) {
        float2 uv = float2(dot(gSkyL2x0.xyz, float3(uv0,1)), dot(gSkyL2x1.xyz, float3(uv0,1)));
        Texture2D col = ResourceDescriptorHeap[gSkyLayers.z];
        float4 s = col.SampleGrad(gSamp, uv, ddx, ddy);
        float a = s.a;
        if (gSkyLayers.w != 0) { Texture2D alp = ResourceDescriptorHeap[gSkyLayers.w]; a = alp.SampleGrad(gSamp, uv, ddx, ddy).r; }
        c = lerp(c, s.rgb, saturate(a));
    }
    return c;
}

// Primary-camera ray direction for a given pixel (replicates RayGen) -- used to
// build ray differentials so we can hardware-filter textures like the rasterizer does.
float3 pixel_ray_dir(float2 p, float2 dim)
{
    float2 uvp = (p + 0.5f) / dim;
    float2 ndc = uvp * 2.0f - 1.0f; ndc.y = -ndc.y;
    float3 vd = float3(ndc.x / gVW0.w, ndc.y / gVW1.w, -1.0f);
    return normalize(float3(dot(gVW0.xyz, vd), dot(gVW1.xyz, vd), dot(gVW2.xyz, vd)));
}
// Interpolated UV where a ray (eye,dir) meets the triangle's plane (barycentric).
float2 tri_uv(float3 eye, float3 dir, float3 v0, float3 v1, float3 v2, float2 t0, float2 t1, float2 t2)
{
    float3 e1 = v1 - v0, e2 = v2 - v0, nrm = cross(e1, e2);
    float t = dot(nrm, v0 - eye) / dot(nrm, dir);
    float3 vp = eye + t*dir - v0;
    float d00 = dot(e1,e1), d01 = dot(e1,e2), d11 = dot(e2,e2), d20 = dot(vp,e1), d21 = dot(vp,e2);
    float den = d00*d11 - d01*d01;
    float w1 = (d11*d20 - d01*d21) / den, w2 = (d00*d21 - d01*d20) / den;
    return t0*(1.0f - w1 - w2) + t1*w1 + t2*w2;
}

[shader("closesthit")]
void CHit(inout Payload p, BuiltInTriangleIntersectionAttributes attr)
{
    uint tri = gStartIndex + PrimitiveIndex() * 3;
    uint i0 = load_index(tri+0), i1 = load_index(tri+1), i2 = load_index(tri+2);
    float3x4 o2w = ObjectToWorld3x4();
    float3 v0 = mul(o2w, float4(load_pos(i0),1)), v1 = mul(o2w, float4(load_pos(i1),1)), v2 = mul(o2w, float4(load_pos(i2),1));
    float3 n = normalize(cross(v1 - v0, v2 - v0));
    float w1 = attr.barycentrics.x, w2 = attr.barycentrics.y, w0 = 1.0f - w1 - w2;
    float2 uv = load_uv(i0)*w0 + load_uv(i1)*w1 + load_uv(i2)*w2;

    // Reconstruct the material tangent frame once. This drives both Wolf's
    // view-dependent height offset and its tangent-space normal map.
    float3 e1 = v1-v0, e2 = v2-v0;
    float2 duv1 = load_uv(i1)-load_uv(i0), duv2 = load_uv(i2)-load_uv(i0);
    float det = duv1.x*duv2.y - duv1.y*duv2.x;
    float3 T = 0.0f, B = 0.0f;
    bool tangentValid = abs(det) > 1e-8f;
    if (tangentValid) {
        float invDet = rcp(det);
        T = normalize((e1*duv2.y - e2*duv1.y) * invDet);
        T = normalize(T - n*dot(n,T));
        float3 Braw = (e2*duv1.x - e1*duv2.x) * invDet;
        B = normalize(cross(n,T));
        if (dot(B,Braw) < 0.0f) B = -B;
    }

    if (gDepthHeapIndex != 0 && tangentValid && abs(gParallaxParams.x) > 1e-7f) {
        Texture2D depthTex = ResourceDescriptorHeap[gDepthHeapIndex];
        // Data textures currently share the bridge's sRGB SRV, so undo that
        // decode before interpreting the stored grayscale height.
        float height = linear_to_srgb(depthTex.SampleLevel(gSamp, uv, 0).rgb).r;
        float3 toEye = normalize(-WorldRayDirection());
        float3 viewTS = float3(dot(toEye,T), dot(toEye,B), dot(toEye,n));
        float amount = height * gParallaxParams.x + gParallaxParams.y;
        // Bound malformed/legacy constants and grazing angles; parallax should
        // add relief, never pull a surface across half the texture.
        amount = clamp(amount, -0.035f, 0.035f);
        uv += viewTS.xy * (amount / max(abs(viewTS.z), 0.35f));
    }

    // The high base-texture bit tags the atmos sky dome. Surface emission has its
    // own texture slot so an additive material pass can coexist with base albedo.
    bool isSky = (gTexHeapIndex & 0x80000000u) != 0;
    uint texSlot = gTexHeapIndex & 0x7FFFFFFFu;

    float3 albedo;
    if (texSlot != 0) { Texture2D tex = ResourceDescriptorHeap[texSlot]; albedo = tex.SampleLevel(gSamp, uv, 0).rgb; }
    else if ((gEmissiveHeapIndex & 0x7FFFFFFFu) != 0) albedo = 0.0f;
    else { uint id = InstanceID(); albedo = frac(float3(id*0.1234f, id*0.3717f, id*0.6151f)) * 0.6f + 0.2f; }

    if (!isSky && (gNormalHeapIndex & 0x7fffffffu) != 0) {
        uint normalSlot = gNormalHeapIndex & 0x7fffffffu;
        Texture2D<float4> normalTex = ResourceDescriptorHeap[normalSlot];
        float4 encoded = normalTex.SampleLevel(gSamp, uv, 0);
        float3 raw = linear_to_srgb(encoded.rgb);
        // idTech assets may use either RGB tangent normals or DXT5nm/AG encoding.
        // The high descriptor bit is set once per BC3 texture by the host; never
        // guess per pixel, which caused severe streaks where RGB values crossed
        // the old heuristic threshold.
        bool dxt5nm = (gNormalHeapIndex & 0x80000000u) != 0;
        float2 xy = (dxt5nm ? float2(encoded.a, raw.g) : raw.xy) * 2.0f - 1.0f;
        // Wolf's D3D9 normal maps use the opposite tangent-space green-axis
        // convention from the basis reconstructed below.
        xy.y = -xy.y;
        xy *= 0.65f;
        float3 tangentNormal = normalize(float3(xy, sqrt(saturate(1.0f - dot(xy, xy)))));

        if (tangentValid) {
            float3 mapped = normalize(T*tangentNormal.x + B*tangentNormal.y + n*tangentNormal.z);
            // Keep the shading normal in the geometric hemisphere so normal maps
            // cannot create inverted light/shadow leaks at grazing angles.
            if (dot(mapped,n) > 0.05f) n = mapped;
        }
    }

    p.emissive = 0.0f;
    p.emission = 0.0f;
    if (isSky) {
        // FALLBACK sky: emit the clean procedural sky (no dome texture -> no zenith pole
        // artifact). The real per-map dome texture is parked until we re-bake it into a
        // pole-free representation (cubemap). The dome geometry still carries the sky so
        // reflections/GI hit it.
        albedo = analytic_sky(WorldRayDirection(), normalize(gSun.xyz));
        p.emissive = 1.0f;
    } else if ((gEmissiveHeapIndex & 0x3FFFFFFFu) != 0) {
        uint emissiveSlot = gEmissiveHeapIndex & 0x3FFFFFFFu;
        Texture2D et = ResourceDescriptorHeap[emissiveSlot];
        float4 e = et.SampleLevel(gSamp, uv, 0);
        p.emission = e.rgb * (((gEmissiveHeapIndex & 0x80000000u) != 0) ? e.a : 1.0f) * EMISSIVE_SCALE;
        // Expanded particle billboards are pure emitters: no lit diffuse base on the card.
        if (gEmissiveHeapIndex & 0x40000000u) albedo = 0.0f;
    }
    // Wolf's SpecularMap is a Blinn intensity/gloss mask, not metallic RGB F0.
    // Treat ordinary game materials as dielectrics and use the map only to
    // modulate their ~4% reflectance. A conservative roughness floor prevents
    // brick/wood/plaster from turning wet when the legacy Blinn power is high.
    float3 F0 = float3(0.04f, 0.04f, 0.04f);
    float  rough = 0.5f;
    if (!isSky) {
        if (gSpecHeapIndex != 0) {
            Texture2D st = ResourceDescriptorHeap[gSpecHeapIndex];
            float3 legacySpec = st.SampleLevel(gSamp, uv, 0).rgb;
            float mask = dot(legacySpec, float3(0.2126f, 0.7152f, 0.0722f));
            float intensity = (gSpecParams.y > 0.0f) ? gSpecParams.y : 1.0f;
            float dielectricF0 = clamp(0.025f * mask * intensity, 0.005f, 0.04f);
            F0 = dielectricF0.xxx;
        }
        float blinnRough = sqrt(2.0f / (max(gSpecParams.x, 1.0f) + 2.0f));
        rough = clamp(blinnRough * 1.75f, 0.55f, 0.95f);
    }
    p.spec = F0; p.rough = rough;
#if DEBUG_SPEC
    if (!isSky) { albedo = F0; p.emissive = 1.0f; }   // emit raw so the primary pass shows it unshaded
#endif
    p.albedo = albedo; p.normal = n; p.geomNormal = normalize(cross(v1 - v0, v2 - v0)); p.t = RayTCurrent();
}

// Any-hit for expanded world-space particle billboards. Additive and
// premultiplied fire accumulate emission. Smoke contributes Beer-Lambert
// extinction plus softly scene-lit in-scattering; both reductions are
// order-independent, which is important because any-hit traversal is unordered.
[shader("anyhit")]
void PartAnyHit(inout Payload p, BuiltInTriangleIntersectionAttributes attr)
{
    uint tri = gStartIndex + PrimitiveIndex() * 3;
    uint i0 = load_index(tri+0), i1 = load_index(tri+1), i2 = load_index(tri+2);
    uint slot = load_particle_tex(i0);
    if (slot != 0) {
        float w1 = attr.barycentrics.x, w2 = attr.barycentrics.y, w0 = 1.0f - w1 - w2;
        float2 uv = load_uv(i0)*w0 + load_uv(i1)*w1 + load_uv(i2)*w2;
        float4 vc = load_particle_color(i0)*w0 + load_particle_color(i1)*w1 + load_particle_color(i2)*w2;
        Texture2D et = ResourceDescriptorHeap[slot];
        float4 e = et.SampleLevel(gSamp, uv, 0);
        uint cls = load_particle_class(i0);
        float alpha = saturate(e.a * vc.a);
        float3 particleColor = max(0.0f, e.rgb * vc.rgb);

        if (cls == 1u) { // PART_VOLUMETRIC
            // Nearly transparent texels are abundant around every billboard.
            // Reject them before any volumetric math to keep dense effects cheap.
            if (alpha < (1.0f / 255.0f) || p.smokeTrans < 0.01f) {
                IgnoreHit();
                return;
            }
            float opticalDepth = alpha * PARTICLE_SMOKE_DENSITY;
            float opacity = 1.0f - exp(-opticalDepth);

            // Keep any-hit constant-time. Iterating all scene lights here scales
            // as pixels * smoke layers * lights and can trip the Windows GPU
            // watchdog in dense effects. Broad sun/sky fill is stable; a bounded
            // clustered light lookup can be added later outside traversal.
            float3 illumination = 0.045f + max(0.0f, SUN_COL) * 0.055f;
            p.smokeTrans *= (1.0f - opacity);
            p.smokeColorSum += particleColor * illumination * opacity;
            p.smokeAlphaSum += opacity;
        } else {
            // cls 0 is additive fire/sparks; cls 2 is premultiplied soft fire.
            // Replaying the game's VS color restores lifetime fades and tinting.
            p.partAccum += particleColor * alpha * PARTICLE_EMISSIVE_SCALE;
        }
    }
    IgnoreHit();
}
