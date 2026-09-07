# Build the 64-bit host (x64 -- hardware RT is 64-bit only).
$ErrorActionPreference = 'Stop'
$root   = $PSScriptRoot
$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat'

New-Item -ItemType Directory -Force -Path "$root\build" | Out-Null

# NGX (DLSS Ray Reconstruction) SDK from Remix's packman cache.
$ngx = 'C:\packman-repo\chk\rtx-remix-ngx_sdk_dldn\5'
$ngxInc = "$ngx\include"
$ngxLib = "$ngx\lib\Windows_x86_64\x86_64\nvsdk_ngx_d.lib"
# Newer DLSS runtime DLLs the user supplied (v3.10.7); NGX loads these from the exe dir.
$dllSrc = 'C:\Users\MJBee\Desktop\decom'

# Compile the DXR shader library + tonemap compute to DXIL (SM 6.6 -> dxc).
$dxc  = 'dxc -nologo -enable-16bit-types -T lib_6_6 -Fo build\shader.cso shader.hlsl'
$dxc2 = 'dxc -nologo -T cs_6_6 -E CSMain -Fo build\tonemap.cso tonemap.hlsl'
$dxc3 = 'dxc -nologo -enable-16bit-types -T cs_6_6 -E CSMain -Fo build\sharc_resolve.cso sharc_resolve.hlsl'
$dxc4 = 'dxc -nologo -T cs_6_6 -E CSMain -Fo build\fog.cso fog.hlsl'
# /MD: nvsdk_ngx_d.lib imports the UCRT dynamically (__imp_*), so match with the dynamic CRT.
$cl  = 'cl /nologo /O2 /EHsc /MD /I..\shared /I"' + $ngxInc + '" host.cpp gfx.cpp dlss.cpp /Fobuild\ /Fe:build\wolfrt_host.exe ' +
       '/link user32.lib d3d12.lib dxgi.lib advapi32.lib "' + $ngxLib + '"'
cmd /c "`"$vcvars`" x64 && cd /d `"$root`" && $dxc && $dxc2 && $dxc3 && $dxc4 && $cl"
if ($LASTEXITCODE -ne 0) { throw "host build failed ($LASTEXITCODE)" }

# Stage the DLSS runtime DLLs next to the exe so NGX can load them.
foreach ($d in @('nvngx_dlss.dll','nvngx_dlssd.dll','nvngx_dlssg.dll')) {
    if (Test-Path "$dllSrc\$d") { Copy-Item "$dllSrc\$d" "$root\build\$d" -Force }
}
Write-Host "built: $root\build\wolfrt_host.exe" -ForegroundColor Green
