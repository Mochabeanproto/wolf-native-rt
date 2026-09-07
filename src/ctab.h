#pragma once
// Parse a D3D9 shader constant table (CTAB) to map the constants we need to
// registers. fxc assigns registers per-shader, so we must read each shader's
// own table rather than assume fixed slots. Results cached by shader pointer.
// Works for both vertex and pixel shaders (same bytecode/CTAB format).
struct IDirect3DVertexShader9;
struct IDirect3DPixelShader9;

struct ShaderConstMap {
    // Camera / transforms (vertex shader).
    int mvp          = -1;  // $mModelViewProjection    (4 regs)
    int proj         = -1;  // $mProjection             (4)
    int modelToWorld = -1;  // $mModelToWorldTransform  (3, float4x3)
    int modelView    = -1;  // $mModelView              (3)
    int viewToWorld  = -1;  // $mViewToWorldTransform   (3)
    int viewOrigin   = -1;  // $vWorldSpaceViewOrigin   (1)
    int skin         = -1;  // $mSkinToViewTransforms   (N)
    // Lights (may be VS or PS depending on the shader). Real Wolf2009 names.
    int sunDir       = -1;  // $vModelSpaceToGlobalLightDirection (1, VS)
    int sunColor     = -1;  // $vGlobalLightColor        (1, VS)
    int lightOrigin  = -1;  // $vLightOrigin             (1, VS)  local light position
    int viewLightPos = -1;  // $vViewLightPosition       (1, PS)  deferred light position
    int lightColor   = -1;  // $vLightDiffuseColor       (1, PS)  local light diffuse
    int invRadius    = -1;  // $fInverseLightRadius      (1, PS)  1/radius
    int skyScale     = -1;  // $vSkyColorScale           (1, PS)  marks the sky/env-map draw
    int envMap       = -1;  // $EnvironmentMap           sampler index
    int envRotation  = -1;  // $vEnvRotation0            (3, VS) environment orientation
    // Material G-buffer pass: specular lobe inputs.
    int specMap      = -1;  // SpecularMap               sampler (s1) specular colour/gloss
    int normalMap    = -1;  // NormalMap                 sampler (s2) tangent-space normal
    int depthMap     = -1;  // DepthMap                  sampler (s3) parallax height
    int specParams   = -1;  // $vSpecularPowerIntensityBiasScale (1) power/intensity/bias/scale
    int parallaxParams = -1; // $vParallaxScaleBias       (1) scale/bias
    // Sky-dome (atmos) layer: fullscreen pass, direction -> sky-texture UV.
    int colorMap     = -1;  // $ColorMap                 sampler (the sky texture)
    int texXform     = -1;  // $mTextureTransform        (2, VS) direction -> UV
    int hposBias     = -1;  // $vHPosBias                (1, VS) far-plane bias (sky/background marker)
    int colorMod     = -1;  // $vColorModulate           (1, VS) tint (scene fog color)
    int colorAdd     = -1;  // $vColorAdd                (1, VS) additive bias
    // Veil vision compositor. These names together form a launch-independent
    // fingerprint; generic particles may use ScreenMap but not this full set.
    int screenMap       = -1;
    int normalMap0      = -1;
    int normalMap1      = -1;
    int maskMap         = -1;
    int gbufferDepth    = -1;
    int distortionDist = -1;
    int distortionRamp = -1;
    int transitionWipe  = -1;
    int noiseMap        = -1;
    int normalXform0    = -1;
    int normalXform1    = -1;
    int aspectCorrection= -1;
    // GPU particle system. The engine expands/animates camera-facing quads in
    // the VS; this fingerprint keeps them distinct from decals and light volumes.
    int particleBounds   = -1; // $vBoundingSphereCenterRadiusSqrd
    int particlePlayback = -1; // $vPlaybackStopCycleTime
    int particleFilmstrip= -1; // $vPivotPointFilmstripScale
    int particleFadeMap  = -1; // $ParticleColorFadeMap sampler
    bool parsed      = false;
};

// Returns a cached parse of the shader's CTAB (parses on first use).
const ShaderConstMap& ctab_get(IDirect3DVertexShader9* vs);
const ShaderConstMap& ctab_get_ps(IDirect3DPixelShader9* ps);

// Diagnostic: log every constant (name, register set + index + count) in a
// shader's CTAB. Used once to discover the exact light/sky constant layout.
void ctab_log_all(const void* code, unsigned bytes, const char* tag);
