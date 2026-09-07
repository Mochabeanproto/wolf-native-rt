// Quarter-resolution analytic smoke volumes. The CPU converts each Wolf smoke
// billboard into a textured ellipsoid; this pass integrates extinction and
// in-scattering only up to the opaque scene depth.

struct FogVolume
{
    float4 centerDensity; // xyz center, w density multiplier
    float4 axisU;         // xyz unit axis, w half extent
    float4 axisV;
    float4 axisW;         // billboard normal, w artificial half thickness
    float4 uvRect;        // minU,minV,maxU,maxV in the original atlas
    float4 colorTex;      // rgb vertex tint, w = texture heap index as float bits
};

StructuredBuffer<FogVolume> Volumes : register(t0);
RWTexture2D<float4> FogOut : register(u0); // rgb in-scattering, a transmittance
SamplerState gLinear : register(s0);

cbuffer FogParams : register(b0)
{
    float4 gVW0;         // camera basis row 0; w unused
    float4 gVW1;
    float4 gVW2;
    float4 gEyeCount;    // xyz eye; w volume count as uint bits
    float4 gProjection;  // sx, sy, projection A, projection C
    float4 gSunColor;
    float4 gFogControl;  // x density scale, y ambient, z sun scale
};

bool intersect_ellipsoid(float3 ro, float3 rd, FogVolume v, out float t0, out float t1)
{
    float3 d = ro - v.centerDensity.xyz;
    float3 o = float3(dot(d,v.axisU.xyz)/v.axisU.w,
                      dot(d,v.axisV.xyz)/v.axisV.w,
                      dot(d,v.axisW.xyz)/v.axisW.w);
    float3 r = float3(dot(rd,v.axisU.xyz)/v.axisU.w,
                      dot(rd,v.axisV.xyz)/v.axisV.w,
                      dot(rd,v.axisW.xyz)/v.axisW.w);
    float a = dot(r,r), b = dot(o,r), c = dot(o,o)-1.0f;
    float disc = b*b-a*c;
    if (disc <= 0.0f || a <= 1e-10f) { t0=t1=0; return false; }
    float s = sqrt(disc);
    t0 = (-b-s)/a; t1 = (-b+s)/a;
    return t1 > max(t0,0.0f);
}

[numthreads(8,8,1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint fw,fh; FogOut.GetDimensions(fw,fh);
    if (id.x>=fw || id.y>=fh) return;

    float2 uv=(float2(id.xy)+0.5f)/float2(fw,fh);
    float2 ndc=uv*2.0f-1.0f; ndc.y=-ndc.y;
    float3 vd=float3(ndc.x/gProjection.x,ndc.y/gProjection.y,-1.0f);
    float3 rd=normalize(float3(dot(gVW0.xyz,vd),dot(gVW1.xyz,vd),dot(gVW2.xyz,vd)));
    float3 fwd=-normalize(float3(gVW0.z,gVW1.z,gVW2.z));

    RWTexture2D<float> depthTex=ResourceDescriptorHeap[2];
    uint dw,dh; depthTex.GetDimensions(dw,dh);
    uint2 dp=min(uint2(uv*float2(dw,dh)),uint2(dw-1,dh-1));
    float depth=depthTex[dp];
    float denom=depth+gProjection.z;
    float linearDepth=(abs(denom)>1e-7f)?abs(gProjection.w/denom):1e6f;
    float sceneT=(depth>=0.99999f)?1e6f:linearDepth/max(dot(rd,fwd),0.02f);

    float3 scattering=0.0f;
    float transmittance=1.0f;
    uint count=asuint(gEyeCount.w);
    [loop] for(uint i=0;i<count && transmittance>0.01f;++i) {
        FogVolume v=Volumes[i];
        float enterT,exitT;
        if(!intersect_ellipsoid(gEyeCount.xyz,rd,v,enterT,exitT)) continue;
        enterT=max(enterT,0.05f); exitT=min(exitT,sceneT);
        if(exitT<=enterT) continue;

        float midT=(enterT+exitT)*0.5f;
        float3 q=gEyeCount.xyz+rd*midT-v.centerDensity.xyz;
        float3 local=float3(dot(q,v.axisU.xyz)/v.axisU.w,
                            dot(q,v.axisV.xyz)/v.axisV.w,
                            dot(q,v.axisW.xyz)/v.axisW.w);
        // Sample only the center of the particle's current atlas frame to retain
        // its smoke tone. Using local.xy as texture coordinates stamped the
        // original 2D puff silhouette onto the volume.
        float2 tuv=(v.uvRect.xy+v.uvRect.zw)*0.5f;
        uint texIndex=asuint(v.colorTex.w);
        Texture2D<float4> smokeTex=ResourceDescriptorHeap[texIndex];
        float4 texel=smokeTex.SampleLevel(gLinear,tuv,0);

        // A 3D Gaussian replaces the visible sprite cutout. Density is already
        // negligible at the analytic ellipsoid boundary, avoiding hard rims.
        float profile=exp(-4.0f*dot(local,local));
        float opticalDepth=max(0.0f,max(texel.a,0.25f)*v.centerDensity.w*gFogControl.x*profile)
                          *((exitT-enterT)/max(v.axisW.w*2.0f,1.0f));
        float segmentTrans=exp(-opticalDepth);
        float tone=dot(max(0.0f,texel.rgb),float3(0.2126f,0.7152f,0.0722f));
        float3 smokeColor=lerp(0.10f.xxx,0.38f.xxx,saturate(tone))*max(v.colorTex.rgb,0.25f);
        float3 illumination=gFogControl.y+max(0.0f,gSunColor.rgb)*gFogControl.z;
        scattering+=transmittance*smokeColor*illumination*(1.0f-segmentTrans);
        transmittance*=segmentTrans;
    }
    FogOut[id.xy]=float4(scattering,saturate(transmittance));
}
