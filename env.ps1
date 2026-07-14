param(
    [ValidateSet("cpu", "cuda")]
    [string]$Profile = "cpu",
    [switch]$WorkbenchStatus,
    [switch]$WorkbenchStop
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot
try {
    if (-not (Get-Command mise -ErrorAction SilentlyContinue)) {
        throw "mise is required. Install it from https://mise.jdx.dev/."
    }
    $Task = if ($WorkbenchStop) {
        "workbench:stop"
    } elseif ($WorkbenchStatus) {
        "workbench:status"
    } else {
        "workbench:start"
    }
    if ($Task -ne "workbench:start") {
        mise run $Task
        if ($LASTEXITCODE -ne 0) { throw "Workbench command failed: $Task" }
        return
    }

    mise run dev --profile $Profile
    if ($LASTEXITCODE -ne 0) { throw "Emperor setup or Workbench startup failed." }

    . (Join-Path $ProjectRoot "torchenv\Scripts\Activate.ps1")
} finally {
    Pop-Location
}
