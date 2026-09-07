// Reinhard tonemap + gamma of the RR-denoised linear-HDR image into the R8 output
// that gets copied to the swapchain. Src is bound as a UAV (read) so no state
// transition is needed between RR (which writes UAV) and this pass.
RWTexture2D<float4> Dst : register(u0);   // heap[8]: R8G8B8A8 output
RWTexture2D<float4> Src : register(u1);   // heap[9]: RGBA16F (RR output, or raw color in fallback)
SamplerState gWrap : register(s0);
SamplerState gClamp : register(s1);

cbuffer VeilParams : register(b0)
{
    uint4  gVeilTex;       // normal0, normal1, mask heap indices; w=active
    float4 gVeilDistance;  // exact Wolf PS $vDistortionDistanceWeightControl
    float4 gVeilRamp;      // exact Wolf PS $vDistortionRampControl
    float4 gVeilXform0;    // exact Wolf VS $vNormalMap0TranslateScale
    float4 gVeilXform1;    // exact Wolf VS $vNormalMap1TranslateScale
    float4 gVeilMisc;      // x=aspect correction
};

float4 sample_src_linear(float2 uv, uint w, uint h)
{
    float2 p=saturate(uv)*float2(w-1,h-1);
    int2 a=int2(floor(p)); int2 b=min(a+1,int2(w-1,h-1)); float2 f=frac(p);
    float4 x0=lerp(Src[int2(a.x,a.y)],Src[int2(b.x,a.y)],f.x);
    float4 x1=lerp(Src[int2(a.x,b.y)],Src[int2(b.x,b.y)],f.x);
    return lerp(x0,x1,f.y);
}

float4 sample_fog_linear(float2 uv)
{
    RWTexture2D<float4> fog=ResourceDescriptorHeap[19];
    uint w,h; fog.GetDimensions(w,h);
    float2 p=saturate(uv)*float2(w-1,h-1);
    int2 a=int2(floor(p)); int2 b=min(a+1,int2(w-1,h-1)); float2 f=frac(p);
    float4 x0=lerp(fog[int2(a.x,a.y)],fog[int2(b.x,a.y)],f.x);
    float4 x1=lerp(fog[int2(a.x,b.y)],fog[int2(b.x,b.y)],f.x);
    return lerp(x0,x1,f.y);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h; Dst.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    float3 c;
    if (gVeilTex.w != 0u && gVeilTex.x != 0u && gVeilTex.y != 0u && gVeilTex.z != 0u) {
        float2 uv=(float2(id.xy)+0.5f)/float2(w,h);
        RWTexture2D<float> depthTex=ResourceDescriptorHeap[2];
        Texture2D<float4> n0=ResourceDescriptorHeap[gVeilTex.x];
        Texture2D<float4> n1=ResourceDescriptorHeap[gVeilTex.y];
        Texture2D<float4> mask=ResourceDescriptorHeap[gVeilTex.z];
        float2 maskUV=float2((uv.x-0.5f)*gVeilMisc.x+0.5f,uv.y);
        uint dw,dh; depthTex.GetDimensions(dw,dh);
        uint2 dp=min(uint2(uv*float2(dw,dh)),uint2(dw-1,dh-1));
        float depth=depthTex[dp];
        float distanceWeight=saturate((depth-gVeilDistance.x)/max(gVeilDistance.y-gVeilDistance.x,1e-6f));
        float ramp=saturate(pow(distanceWeight,gVeilRamp.z)*gVeilRamp.x+gVeilRamp.y);
        float weight=mask.SampleLevel(gClamp,maskUV,0).a*ramp;
        float2 uv0=uv*gVeilXform0.zw+gVeilXform0.xy;
        float2 uv1=uv*gVeilXform1.zw+gVeilXform1.xy;
        float4 a=n0.SampleLevel(gWrap,uv0,0), b=n1.SampleLevel(gWrap,uv1,0);
        float2 deformation=float2(lerp(gVeilDistance.z,a.a,b.a),lerp(gVeilDistance.z,a.g,b.g));
        c=max(0.0f,sample_src_linear(uv+weight*deformation*0.5f,w,h).rgb);
    } else c=max(0.0f,Src[id.xy].rgb);

    float2 finalUV=(float2(id.xy)+0.5f)/float2(w,h);
    float4 fog=sample_fog_linear(finalUV);
    c=c*fog.a+fog.rgb;
    c = c / (c + 1.0f);                    // Reinhard
    c = pow(c, 1.0f / 2.2f);               // linear -> sRGB
    Dst[id.xy] = float4(c, 1.0f);
}
