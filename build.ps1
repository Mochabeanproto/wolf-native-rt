# Build the 32-bit proxy d3d9.dll. Optional -Deploy copies it into the game folder.
#   .\build.ps1            # build only  -> build\d3d9.dll
#   .\build.ps1 -Deploy    # build + copy into "SP - Native Raytracer"
param([switch]$Deploy)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$game = 'D:\SteamLibrary\steamapps\common\Wolfenstein\SP - Native Raytracer'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat'

New-Item -ItemType Directory -Force -Path "$root\build" | Out-Null

# /MT = static CRT (no redist dependency inside the game process); x86 to match Wolf2.exe.
$cl = 'cl /nologo /O2 /EHsc /MT /LD /Isrc /Ishared ' +
      'src\hooks.cpp src\log.cpp src\capture.cpp src\ctab.cpp src\bridge_client.cpp src\vsinterp.cpp /Fobuild\ ' +
      '/link /DEF:d3d9.def /OUT:build\d3d9.dll kernel32.lib user32.lib d3dcompiler.lib'
cmd /c "`"$vcvars`" x86 && cd /d `"$root`" && $cl"
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

$dll = Join-Path $root 'build\d3d9.dll'
Write-Host "built: $dll ($((Get-Item $dll).Length) bytes)" -ForegroundColor Green

if ($Deploy) {
    if (-not (Test-Path $game)) { throw "game folder not found: $game" }
    Copy-Item $dll (Join-Path $game 'd3d9.dll') -Force
    Write-Host "deployed -> $game\d3d9.dll" -ForegroundColor Green
}
