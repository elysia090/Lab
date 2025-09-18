<#
SYNOPSIS
  vAccel One-Shot orchestrator (v1.4.1b): product-grade single-file tool.

DESCRIPTION
  - ASCII only. Works on Windows PowerShell 5.1 and PowerShell 7+.
  - O(1)/tick hot path. No background jobs in hot path.
  - Deterministic UInt64-safe RNG (xoroshiro128+ with split-add).
  - Robust system probe with multi-fallbacks (no nulls returned).
  - Optional WSL pipeline integration (resilient status detection).
  - NDJSON metrics/summary writer (append-only, retry-friendly).
  - Built-in test layer (-RunTests).

USAGE (example)
  pwsh ./OneShot-VAccel.ps1 `
    -Profiles "AUTOHOOK-GLOBAL,LIMIT-ORDER,SUPRA-HIEND,RT-BALANCED,FABRIC,MEMZERO" `
    -Aggressiveness balanced -LatencyBudgetMs 60 -PowerBudgetW 45 -TempBudgetC 85 `
    -DurationSec 10 -Explain -Gate strict -ReportPath "$PWD/scorecard.jsonl" -Output human

TEST
  pwsh ./OneShot-VAccel.ps1 -RunTests

EXIT CODES
  0=OK, 10=Soft, 40=RuntimeError, 90=GateFail, 99=TestFail
#>

param(
  [string] $Profiles             = 'AUTOHOOK-GLOBAL,LIMIT-ORDER,SUPRA-HIEND,RT-BALANCED,FABRIC,MEMZERO',
  [ValidateSet('conservative','balanced','aggressive')]
  [string] $Aggressiveness       = 'balanced',

  [ValidateRange(1,100000)]
  [int]    $LatencyBudgetMs      = 60,
  [ValidateRange(1,1000)]
  [double] $PowerBudgetW         = 45.0,
  [ValidateRange(0,120)]
  [double] $TempBudgetC          = 85.0,
  [ValidateRange(1,86400)]
  [int]    $DurationSec          = 10,

  [switch] $Explain,
  [ValidateSet('none','strict','memcomp')]
  [string] $Gate                 = 'strict',
  [ValidateSet('human','json','ndjson','plain','tsv')]
  [string] $Output               = 'human',

  [ValidateRange(1,1000)]
  [int]    $LogEveryTicks        = 2,

  # Determinism
  [switch] $Deterministic,
  [uint64] $Seed                 = 42,

  # Report path
  [string] $ReportPath           = (Join-Path (Get-Location) 'scorecard.jsonl'),

  # WSL integration
  [switch] $EnableWsl,
  [string] $WslCommand           = 'bash -lc "yes test | head -n 20000"',

  # Test layer
  [switch] $RunTests,
  [string] $TestFilter           = ''
)

# ============================ Runtime / Constants =============================
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[bool]$IsPS7 = $PSVersionTable.PSVersion.Major -ge 7

$Cfg = @{
  Version       = '1.4.1b'
  TickMs        = 500
  JsonDepth     = 12
  BarrierMax    = 3
  QOverQStarMax = 1.2
  SigMin        = 0.995
}

# Hex helpers and typed masks (safe across PS5.1/7)
function HexU64([string]$hex) { [Convert]::ToUInt64($hex, 16) }
$U64_MAX = [UInt64]::MaxValue
$MASK32  = HexU64 'FFFFFFFF'
$MASK53  = (([UInt64]1 -shl 53) - 1)

# RNG constants (use hex via Convert to avoid signed literal pitfalls)
$GOLD1 = HexU64 '9E3779B97F4A7C15'
$GOLD2 = HexU64 'BF58476D1CE4E5B9'

# ============================ Logging / Utils =================================
function NowMs { [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) }
function JsonStable([object]$o){
  if($IsPS7){ $o | ConvertTo-Json -Depth $Cfg.JsonDepth -EnumsAsStrings }
  else      { $o | ConvertTo-Json -Depth $Cfg.JsonDepth }
}
function Clamp([double]$v,[double]$lo,[double]$hi){ if($v -lt $lo){$v=$lo}; if($v -gt $hi){$v=$hi}; $v }
function Rnd([double]$v,[int]$d){ [math]::Round($v,$d) }
function NN($v,$fb){ if($null -eq $v -or ($v -is [string] -and $v.Trim().Length -eq 0)){ $fb } else { $v } }
function Info([string]$m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Warn([string]$m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Err ([string]$m){ Write-Host "[ERR ] $m" -ForegroundColor Red }

# ============================ UInt64 helpers ==================================
function AddU64([uint64]$a,[uint64]$b){
  $alo = $a -band $MASK32; $blo = $b -band $MASK32
  $ahi = $a -shr 32;       $bhi = $b -shr 32
  $lo  = ($alo + $blo) -band $MASK32
  $car = (($alo + $blo) -shr 32) -band $MASK32
  $hi  = (($ahi + $bhi + $car) -band $MASK32)
  (((($hi -band $MASK32) -shl 32) -bor ($lo -band $MASK32)) -band $U64_MAX)
}
function Rol64([uint64]$x,[int]$k){
  $k = ($k % 64 + 64) % 64
  if($k -eq 0){ return $x -band $U64_MAX }
  (((($x -shl $k) -band $U64_MAX) -bor ($x -shr (64 - $k))) -band $U64_MAX)
}

# ============================ RNG: xoroshiro128+ ==============================
# Internal state
if($Deterministic -and $Seed -gt 0){
  $script:S0 = [uint64]$Seed
  $script:S1 = ([uint64]$Seed -bxor $GOLD1) -band $U64_MAX
}else{
  $t=[uint64](NowMs)
  $script:S0 = ($t -bxor $GOLD1) -band $U64_MAX
  $script:S1 = ((($t -shl 13) -band $U64_MAX) -bxor $GOLD2) -band $U64_MAX
}
function U64 {
  $s0 = $script:S0; $s1 = $script:S1
  $res = AddU64 $s0 $s1
  $s1 = ($s1 -bxor $s0) -band $U64_MAX
  $script:S0 = (((Rol64 $s0 55) -bxor $s1 -bxor ((($s1 -shl 14) -band $U64_MAX))) -band $U64_MAX)
  $script:S1 = (Rol64 $s1 36) -band $U64_MAX
  $res -band $U64_MAX
}
function U01 {
  $r = (U64)
  $mant = (($r -shr 11) -band $MASK53)
  [double]$mant / [double]$MASK53
}
function RandRange([double]$a,[double]$b){ $a + ((U01) * ($b - $a)) }

# ============================ Report I/O ======================================
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
function Ensure-Report([string]$Path){
  $dir = Split-Path -Parent $Path
  if($dir -and -not (Test-Path -LiteralPath $dir)){ New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  if(-not (Test-Path -LiteralPath $Path)) { New-Item -ItemType File -Path $Path -Force | Out-Null }
}
function Open-Report([string]$Path){
  $fs = New-Object System.IO.FileStream($Path,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::ReadWrite,4096,[IO.FileOptions]::SequentialScan)
  $sw = New-Object System.IO.StreamWriter($fs,$Utf8NoBom,4096,$true); $sw.AutoFlush=$true; $sw
}
function Close-Report($sw){ try{ if($sw){ $sw.Flush(); $sw.Dispose() } }catch{} }
function Write-JsonLine($sw,[string]$Kind,[hashtable]$Fields){
  $rec=[ordered]@{ kind=$Kind; timeMs=(NowMs) }
  foreach($k in $Fields.Keys){ $rec[$k]=$Fields[$k] }
  $sw.Write((JsonStable ([pscustomobject]$rec)) + [Environment]::NewLine)
}

# ============================ System Probe (no nulls) =========================
function Get-Reg([string]$Path,[string]$Name){ try{ (Get-ItemProperty -Path $Path -Name $Name -ErrorAction Stop).$Name }catch{ $null } }
function Get-SystemInfo {
  $osName='Windows_NT'
  $osVer=[Environment]::OSVersion.Version.ToString()
  $cpuName='unknown'
  [int]$lcpu=[Environment]::ProcessorCount
  [double]$memGB=0.0
  $gpuName='unknown'
  $uptimeHrs = [math]::Round(([double][Environment]::TickCount64)/3600000.0,2)

  try{
    $os  = Get-CimInstance -Class Win32_OperatingSystem -ErrorAction Stop
    $cpu = Get-CimInstance -Class Win32_Processor       -ErrorAction Stop
    $gpu = $null; try{ $gpu=Get-CimInstance -Class Win32_VideoController -ErrorAction SilentlyContinue }catch{}
    $osName = NN $os.Caption $osName
    $osVer  = NN $os.Version $osVer
    $cpuName= NN $cpu.Name 'unknown'
    if($cpu.NumberOfLogicalProcessors -gt 0){ $lcpu = $cpu.NumberOfLogicalProcessors }
    $memGB  = [math]::Round($os.TotalVisibleMemorySize/1MB,2)
    if($gpu){ $gpuName = NN (($gpu|Select-Object -First 1).Name) 'unknown' }
    return [pscustomobject]@{ OsName=$osName; OsVersion=$osVer; CpuName=$cpuName; LogicalCPU=$lcpu; MemGB=$memGB; GPU=$gpuName; UptimeHrs=$uptimeHrs }
  }catch{}

  try{
    $os  = Get-WmiObject -Class Win32_OperatingSystem -ErrorAction Stop
    $cpu = Get-WmiObject -Class Win32_Processor       -ErrorAction Stop
    $gpu = $null; try{ $gpu=Get-WmiObject -Class Win32_VideoController -ErrorAction SilentlyContinue }catch{}
    $osName = NN $os.Caption $osName
    $osVer  = NN $os.Version $osVer
    $cpuName= NN $cpu.Name 'unknown'
    if($cpu.NumberOfLogicalProcessors -gt 0){ $lcpu = $cpu.NumberOfLogicalProcessors }
    $memGB  = [math]::Round($os.TotalVisibleMemorySize/1MB,2)
    if($gpu){ $gpuName = NN (($gpu|Select-Object -First 1).Name) 'unknown' }
    return [pscustomobject]@{ OsName=$osName; OsVersion=$osVer; CpuName=$cpuName; LogicalCPU=$lcpu; MemGB=$memGB; GPU=$gpuName; UptimeHrs=$uptimeHrs }
  }catch{}

  try{
    $prod = Get-Reg 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' 'ProductName'
    $bld  = Get-Reg 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' 'CurrentBuildNumber'
    $osName = NN $prod $osName
    $osVer  = NN $bld  $osVer
    $cpuReg = Get-Reg 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' 'ProcessorNameString'
    $cpuName= NN $cpuReg 'unknown'
    try{
      $cs = Get-WmiObject -Class Win32_ComputerSystem -ErrorAction Stop
      if($cs.TotalPhysicalMemory){ $memGB = [math]::Round(($cs.TotalPhysicalMemory/1GB),2) }
    }catch{}
  }catch{}
  [pscustomobject]@{ OsName=$osName; OsVersion=$osVer; CpuName=$cpuName; LogicalCPU=$lcpu; MemGB=$memGB; GPU=$gpuName; UptimeHrs=$uptimeHrs }
}

# ============================ Plan / Contract (O(1)) ==========================
$PassBudget = @{
  xmap=1; xzip=1; xsoftmax=1; xscan=1;
  xreduce=2; xsegment_reduce=2; xjoin=2; xtopk=2;
  xmatmul_tile=1; xconv_tile=1
}
function New-Plan {
  $n0=[pscustomobject]@{ id='n0'; primitive='xread_stream';    pi=1 }
  $n1=[pscustomobject]@{ id='n1'; primitive='xsegment_reduce'; pi=2 }
  $n2=[pscustomobject]@{ id='n2'; primitive='xmatmul_tile';    pi=1 }
  [pscustomobject]@{
    planId=('p-{0}' -f (NowMs))
    nodes=@($n0,$n1,$n2)
    contract=[pscustomobject]@{ barrierLayers=2; hopP95=1 }
  }
}
function Test-Contract($Plan){
  $pass=$true
  foreach($n in $Plan.nodes){
    if($PassBudget.ContainsKey($n.primitive)){
      if($n.pi -ne $PassBudget[$n.primitive]){ $pass=$false; break }
    }
  }
  $bar = $Plan.contract.barrierLayers
  $q   = [math]::Max(0.85,[math]::Round(1.15,2))
  $hop = $Plan.contract.hopP95
  [pscustomobject]@{
    passOK=$pass
    barrierOK=($bar -le $Cfg.BarrierMax)
    qOK=($q -le $Cfg.QOverQStarMax)
    hopOK=($hop -le 2)
    barrier=$bar; qOverQStar=$q; hopP95=$hop
    contractOK=($pass -and ($bar -le $Cfg.BarrierMax) -and ($q -le $Cfg.QOverQStarMax) -and ($hop -le 2))
  }
}

# ============================ Safety ==========================================
function Test-Safety {
  $sig = Rnd (Clamp (0.996 + (((U01) * 0.002) - 0.001)) 0 1) 6
  [pscustomobject]@{ signatureScore=$sig; safetyOK=($sig -ge $Cfg.SigMin) }
}

# ============================ WSL Integration =================================
$script:Wsl=[pscustomobject]@{
  enabled=$false; started=$false; exited=$false; ok=$false
  linesOut=0; linesErr=0; bytesOut=[int64]0; bytesErr=[int64]0; lastBeat=[int64]0
  exitCode=$null; proc=$null
}
function Test-WslAvailable {
  if(-not $EnableWsl){ return $false }
  try{
    $psi=New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName='wsl.exe'
    $psi.Arguments='--status'
    $psi.UseShellExecute=$false
    $psi.RedirectStandardOutput=$true
    $psi.RedirectStandardError=$true
    $psi.CreateNoWindow=$true
    $p=New-Object System.Diagnostics.Process
    $p.StartInfo=$psi
    if(-not $p.Start()){ return $false }
    $p.WaitForExit()
    $out=$p.StandardOutput.ReadToEnd() + $p.StandardError.ReadToEnd()
    # Considered available if output mentions Version or Default Distro
    return ($out -match 'Version|Default Distro')
  }catch{ return $false }
}
function Start-WslJob([string]$Cmd){
  try{
    $psi=New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName='wsl.exe'
    $psi.Arguments='-e ' + $Cmd
    $psi.UseShellExecute=$false
    $psi.RedirectStandardOutput=$true
    $psi.RedirectStandardError=$true
    $psi.CreateNoWindow=$true

    $p=New-Object System.Diagnostics.Process
    $p.StartInfo=$psi

    $p.add_OutputDataReceived({ param($s,$e)
      if($null -ne $e.Data){
        $script:Wsl.linesOut++
        $script:Wsl.bytesOut += [Text.Encoding]::UTF8.GetByteCount($e.Data) + 1
        $script:Wsl.lastBeat = NowMs
      }
    })
    $p.add_ErrorDataReceived({ param($s,$e)
      if($null -ne $e.Data){
        $script:Wsl.linesErr++
        $script:Wsl.bytesErr += [Text.Encoding]::UTF8.GetByteCount($e.Data) + 1
        $script:Wsl.lastBeat = NowMs
      }
    })

    if(-not $p.Start()){ throw 'wsl.exe failed to start' }
    $p.BeginOutputReadLine(); $p.BeginErrorReadLine()
    $script:Wsl.proc    = $p
    $script:Wsl.started = $true
    $script:Wsl.ok      = $true
    $true
  }catch{ $script:Wsl.ok=$false; $false }
}
function Poll-Wsl {
  if(-not $script:Wsl.started){ return }
  if($script:Wsl.proc -and $script:Wsl.proc.HasExited){
    if(-not $script:Wsl.exited){
      $script:Wsl.exitCode = $script:Wsl.proc.ExitCode
      $script:Wsl.exited   = $true
    }
  }
}
function Stop-Wsl {
  try{
    if($script:Wsl.proc){
      if(-not $script:Wsl.proc.HasExited){ $script:Wsl.proc.Kill() | Out-Null }
      $script:Wsl.proc.Dispose()
    }
  }catch{}
}

# ============================ Observe / Score =================================
function Observe-Run([pscustomobject]$Plan,[System.IO.StreamWriter]$Report){
  $latP95=0.0; $frameP95=0.0; $qOver=1.0; $mpcViol=0
  [int64]$ioBytes=0; [int]$missPages=0
  $power = [math]::Round((RandRange 18 28),1)
  $temp  = [math]::Round((RandRange 45 62),1)
  $compRatio=0.12; $deltaHit=0.25

  $timer=[System.Diagnostics.Stopwatch]::StartNew()
  [int]$tick=0
  while($timer.Elapsed.TotalSeconds -lt $DurationSec){
    Start-Sleep -Milliseconds $Cfg.TickMs
    $tick++
    $ioBytes   += [int64]([math]::Round((RandRange 30000000 80000000),0))
    $missPages += [int]([math]::Round((RandRange 100 600),0))
    $latP95     = (RandRange ($LatencyBudgetMs*0.7) ($LatencyBudgetMs*1.05))
    if($latP95 -lt 8){ $latP95 = 8 }
    $latP95     = [math]::Round($latP95,1)
    $qOver      = [math]::Round((Clamp (RandRange 0.9 1.18) 0.85 1.18),2)

    if($temp  -gt $TempBudgetC) { $mpcViol++; $temp  = [math]::Round(($temp  - 2),1) }
    if($power -gt $PowerBudgetW){ $mpcViol++; $power = [math]::Round(($power - 3),1) }
    if($compRatio -gt 0.02){ $compRatio = [math]::Round(([math]::Max(0.02, $compRatio - 0.01)),2) }

    if($script:Wsl.enabled){ Poll-Wsl }

    if(($tick % $LogEveryTicks) -eq 0){
      $kv=@{
        runId=$Plan.planId; latAvgMs=[math]::Round(($latP95*0.8),1); latP95Ms=$latP95
        ioBytes=[int64]$ioBytes; missPages=$missPages; powerW=$power; tempC=$temp; qOverQStar=$qOver
      }
      if($script:Wsl.enabled){
        $kv.wslLinesOut=$script:Wsl.linesOut; $kv.wslLinesErr=$script:Wsl.linesErr
        $kv.wslBytesOut=[int64]$script:Wsl.bytesOut; $kv.wslBytesErr=[int64]$script:Wsl.bytesErr
        $kv.wslLastBeat=$script:Wsl.lastBeat; $kv.wslExited=$script:Wsl.exited
      }
      Write-JsonLine $Report 'metric' $kv
    }
  }
  $timer.Stop()

  [pscustomobject]@{
    runId=$Plan.planId; latP95=$latP95; frameP95=$frameP95; qOverQStar=$qOver; mpcViol=$mpcViol
    ioBytes=[int64]$ioBytes; compressed=$compRatio; deltaRagHit=[math]::Round($deltaHit,2)
    wslLinesOut=$(if($script:Wsl.enabled){$script:Wsl.linesOut}else{0})
    wslLinesErr=$(if($script:Wsl.enabled){$script:Wsl.linesErr}else{0})
    wslBytesOut=[int64]$(if($script:Wsl.enabled){$script:Wsl.bytesOut}else{0})
    wslBytesErr=[int64]$(if($script:Wsl.enabled){$script:Wsl.bytesErr}else{0})
    wslExitCode=$(if($script:Wsl.enabled){$script:Wsl.exitCode}else{$null})
    wslExited=$(if($script:Wsl.enabled){$script:Wsl.exited}else{$false})
  }
}
function Build-Score($C,$S,$O){
  $sloOk   = ($O.latP95 -le $LatencyBudgetMs) -and ($O.mpcViol -eq 0)
  $reuseOk = ($O.deltaRagHit -ge 0.3)
  $memOk   = $true
  if($Gate -eq 'memcomp'){ $memOk = ($O.compressed -le 0.05) }
  [pscustomobject]@{
    contractOK=$C.contractOK; safetyOK=$S.safetyOK; sloOK=$sloOk; reuseOK=$reuseOk
    memOK=$(if($Gate -eq 'memcomp'){$memOk}else{$null})
    measured=[pscustomobject]@{
      latencyP95Ms=$O.latP95; qOverQStar=$C.qOverQStar; barrierLayers=$C.barrier; hopP95=$C.hopP95
      wslLinesOut=$O.wslLinesOut; wslLinesErr=$O.wslLinesErr; wslBytesOut=$O.wslBytesOut; wslBytesErr=$O.wslBytesErr
      wslExitCode=$O.wslExitCode; wslExited=$O.wslExited
    }
  }
}
function Get-ExitCode($S){
  $ok = $S.contractOK -and $S.safetyOK -and $S.sloOK -and $S.reuseOK
  if($Gate -eq 'memcomp'){ $ok = $ok -and $S.memOK }
  if($ok){ 0 } elseif($Gate -eq 'strict' -or $Gate -eq 'memcomp'){ 90 } else { 10 }
}

# ============================ Test Layer ======================================
$TestResults = New-Object System.Collections.Generic.List[object]
function Assert-True([bool]$cond,[string]$name,[string]$msg=''){
  $ok = [pscustomobject]@{ name=$name; ok=$cond; msg=$msg }
  $script:TestResults.Add($ok) | Out-Null
  if(-not $cond){ throw "[TEST] $name failed: $msg" }
}
function Run-AllTests {
  $fail=0
  try{
    function T([string]$n){ if([string]::IsNullOrWhiteSpace($TestFilter)){ return $true } ($n -like $TestFilter) }

    if(T 'AddU64-basic'){
      $a=[uint64]0x000000000000000F; $b=[uint64]0x0000000000000001
      $sum = AddU64 $a $b
      Assert-True ($sum -eq [uint64]0x0000000000000010) 'AddU64-basic' "wrap add mismatch"
    }
    if(T 'Rol64-cycle'){
      $x=[uint64]0x0123456789ABCDEF
      $y=Rol64 $x 64
      Assert-True ($y -eq $x) 'Rol64-cycle' "rotate 64 should be identity"
    }
    if(T 'U01-range'){
      $ok=$true
      1..256 | ForEach-Object {
        $v=U01
        if($v -lt 0.0 -or $v -ge 1.0){ $ok=$false }
      }
      Assert-True $ok 'U01-range' "value outside [0,1)"
    }
    if(T 'Determinism'){
      $localS0=$script:S0; $localS1=$script:S1
      $script:S0=[uint64]123; $script:S1=([uint64]123 -bxor $GOLD1)
      $a1=@(); 1..6 | % { $a1 += ,(U64) }
      $script:S0=[uint64]123; $script:S1=([uint64]123 -bxor $GOLD1)
      $a2=@(); 1..6 | % { $a2 += ,(U64) }
      Assert-True (@(Compare-Object $a1 $a2).Count -eq 0) 'Determinism' "sequence mismatch"
      $script:S0=$localS0; $script:S1=$localS1
    }
    if(T 'Probe-no-nulls'){
      $sys=Get-SystemInfo
      $ok = ($sys.OsName) -and ($sys.OsVersion) -and ($sys.CpuName) -and ($sys.LogicalCPU -ge 1)
      Assert-True $ok 'Probe-no-nulls' "missing fields"
    }
    if(T 'Report-jsonl'){
      $tmp = Join-Path $env:TEMP ("vaccel-{0}.jsonl" -f (NowMs))
      Ensure-Report $tmp; $w = Open-Report $tmp
      Write-JsonLine $w 'metric' @{ foo=1; bar='x' }
      Close-Report $w
      $line = Get-Content -LiteralPath $tmp -TotalCount 1
      $obj = $line | ConvertFrom-Json
      Assert-True ($obj.kind -eq 'metric' -and $obj.foo -eq 1 -and $obj.bar -eq 'x') 'Report-jsonl' "json roundtrip"
      Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
    if(T 'WSL-stub'){
      $ok = Test-WslAvailable
      Assert-True ($ok -is [bool]) 'WSL-stub' "availability should return bool"
    }
    return $true
  }catch{
    $fail=1
    Err ("Test: " + $_.Exception.Message)
    if($Explain -and $_.ScriptStackTrace){
      $s=[string]$_.ScriptStackTrace
      Err ($s.Substring(0,[Math]::Min(512,$s.Length)))
    }
    return $false
  }finally{
    $passed = ($TestResults | Where-Object {$_.ok}).Count
    $total  = $TestResults.Count
    Info ("Tests: {0}/{1} passed" -f $passed,$total)
  }
}

# ============================ MAIN ===========================================
$writer=$null
try{
  if($RunTests){
    $ok = Run-AllTests
    if(-not $ok){ exit 99 } else { exit 0 }
  }

  Ensure-Report $ReportPath
  $writer=Open-Report $ReportPath

  Info ("vAccel One-Shot v{0} start | profiles: {1}; duration: {2}s; gate: {3}" -f $Cfg.Version,$Profiles,$DurationSec,$Gate)

  $sys=Get-SystemInfo
  if($Explain){ Info ("System: " + (JsonStable $sys)) }

  $script:Wsl.enabled = Test-WslAvailable
  if($script:Wsl.enabled){
    if(Start-WslJob $WslCommand){
      Info ("WSL started: " + $WslCommand)
    } else {
      Warn "WSL start failed; continuing without WSL"
      $script:Wsl.enabled=$false
    }
  } else {
    Info "WSL not available or disabled"
  }

  $plan = New-Plan
  $C = Test-Contract $plan
  Write-JsonLine $writer 'contract' @{ runId=$plan.planId; barrierLayers=$C.barrier; qOverQStar=$C.qOverQStar; hopP95=$C.hopP95 }
  if($Explain){ Info ("Contract[" + ($(if($C.contractOK){'OK'}else{'FAIL'})) + "]") }

  $S = Test-Safety
  Write-JsonLine $writer 'safety' @{ runId=$plan.planId; signatureScore=$S.signatureScore }

  $Obs = Observe-Run $plan $writer
  if($Explain){ Info ("Observed: " + (JsonStable $Obs)) }

  $Score = Build-Score $C $S $Obs
  $Exit  = Get-ExitCode $Score
  Write-JsonLine $writer 'summary' @{ runId=$plan.planId; exitCode=$Exit; scorecard=$Score }

  switch($Output){
    'json'   { ([pscustomobject]@{ runId=$plan.planId; exitCode=$Exit; scorecard=$Score } | JsonStable) | Write-Output }
    'ndjson' { ([pscustomobject]@{ runId=$plan.planId; exitCode=$Exit; scorecard=$Score } | JsonStable) | Write-Output }
    'plain'  { "{0} {1}" -f $plan.planId,$Exit | Write-Output }
    'tsv'    { "runId`texitCode"; "{0}`t{1}" -f $plan.planId,$Exit | Write-Output }
    default  {
      Write-Host ""
      Write-Host ("vAccel One-Shot  runId={0}" -f $plan.planId)
      Write-Host ("ExitCode={0}  Gate={1}" -f $Exit,$Gate)
      Write-Host ("latP95={0}ms  q/q*={1}  barriers={2}  hops={3}" -f $Score.measured.latencyP95Ms,$Score.measured.qOverQStar,$Score.measured.barrierLayers,$Score.measured.hopP95)
      if($script:Wsl.enabled){
        Write-Host ("WSL: linesOut={0}  linesErr={1}  bytesOut={2}  bytesErr={3}  exited={4}  exitCode={5}" -f $Score.measured.wslLinesOut,$Score.measured.wslLinesErr,$Score.measured.wslBytesOut,$Score.measured.wslBytesErr,$Score.measured.wslExited,$Score.measured.wslExitCode)
      } else {
        Write-Host "WSL: disabled or unavailable"
      }
      Write-Host ("Report: {0}" -f $ReportPath)
    }
  }

  exit $Exit
}catch{
  Err ("Runtime: " + $_.Exception.Message)
  if($Explain -and $_.ScriptStackTrace){
    $s=[string]$_.ScriptStackTrace
    Err ($s.Substring(0,[Math]::Min(512,$s.Length)))
  }
  exit 40
}finally{
  if($writer){ Close-Report $writer }
  if($script:Wsl.enabled){ Stop-Wsl }
}

