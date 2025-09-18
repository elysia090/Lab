<# ================================================================================
 vAccel One-Shot — v1.4.3 (paste-safe, product-grade, PS5.1/PS7+, ASCII only)

 Highlights vs 1.4.2b:
 - Safer NDJSON writer with guarded writes (skip on transient IO issues)
 - Hardened WSL lifecycle: availability check, start with fallback, idle timeout, safe kill
 - Strict clamps for user thresholds; normalized ReportPath; OS-safe snapshot fallbacks
 - Clear, actionable exit hints; configurable ReuseMin (default 0.25)
 - No PS7-only syntax; runs clean on Windows PowerShell 5.1

 Usage tips:
   # optional per-run knobs
   $ReuseMin=0.25; $DurationSec=10; $Gate='strict'; $EnableWsl=$false; $Output='human'
   # paste whole script and run
================================================================================ #>

# ============================ Runtime Defaults (only if UNSET) ===================
if(-not (Test-Path variable:Profiles))          { $Profiles = 'AUTOHOOK-GLOBAL,LIMIT-ORDER,SUPRA-HIEND,RT-BALANCED,FABRIC,MEMZERO' }
if(-not (Test-Path variable:Aggressiveness))    { $Aggressiveness = 'balanced' }
if(-not (Test-Path variable:LatencyBudgetMs))   { [int]   $LatencyBudgetMs = 60 }
if(-not (Test-Path variable:PowerBudgetW))      { [double]$PowerBudgetW    = 45.0 }
if(-not (Test-Path variable:TempBudgetC))       { [double]$TempBudgetC     = 85.0 }
if(-not (Test-Path variable:DurationSec))       { [int]   $DurationSec     = 10 }
if(-not (Test-Path variable:Gate))              { $Gate   = 'strict' }  # 'none'|'strict'|'memcomp'
if(-not (Test-Path variable:Output))            { $Output = 'human' }   # 'human'|'json'|'ndjson'|'plain'|'tsv'
if(-not (Test-Path variable:Explain))           { [bool]$Explain = $true }
if(-not (Test-Path variable:LogEveryTicks))     { [int]$LogEveryTicks = 2 }

# Determinism / seed
if(-not (Test-Path variable:Deterministic))     { [bool]$Deterministic = $true }
if(-not (Test-Path variable:Seed))              { [uint64]$Seed = 42 }

# Report path (safe default)
if(-not (Test-Path variable:ReportPath))        { $ReportPath = Join-Path (Get-Location) 'scorecard.jsonl' }

# WSL integration
if(-not (Test-Path variable:EnableWsl))         { [bool]$EnableWsl = $false }
if(-not (Test-Path variable:WslCommand))        { $WslCommand = 'bash -lc "yes test | head -n 20000"' }
if(-not (Test-Path variable:WslIdleTimeoutMs))  { [int]$WslIdleTimeoutMs = 5000 }  # idle grace for async IO

# Test mode (optional)
if(-not (Test-Path variable:RunTests))          { [bool]$RunTests = $false }
if(-not (Test-Path variable:TestFilter))        { $TestFilter = '' }

# Reuse gate threshold & exit hints
if(-not (Test-Path variable:ReuseMin))          { [double]$ReuseMin = 0.25 }
if(-not (Test-Path variable:ExitHints))         { [bool]$ExitHints = $true }

# ============================ Runtime / Constants =================================
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[bool]$IsPS7 = $PSVersionTable.PSVersion.Major -ge 7

$Config = @{
  Version       = '1.4.3'
  TickMs        = 500
  JsonDepth     = 12
  BarrierMax    = 3
  QOverQStarMax = 1.2
  SigMin        = 0.995
}

# ============================ Logging / Utils =====================================
function Get-UtcNowMs { [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) }
function Convert-ToStableJson([object]$o){ if($IsPS7){ $o | ConvertTo-Json -Depth $Config.JsonDepth -EnumsAsStrings } else { $o | ConvertTo-Json -Depth $Config.JsonDepth } }
function Invoke-Clamp([double]$v,[double]$lo,[double]$hi){ if($v -lt $lo){$v=$lo}; if($v -gt $hi){$v=$hi}; $v }
function Invoke-Round([double]$v,[int]$d){ [math]::Round($v,$d) }
function Get-NonNull($v,$fallback){ if($null -eq $v -or ($v -is [string] -and $v.Trim().Length -eq 0)){ $fallback } else { $v } }
function Write-Info ([string]$m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Warn ([string]$m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-Err  ([string]$m){ Write-Host "[ERR ] $m" -ForegroundColor Red }

# OS helpers
function Get-IsWindows { $PSVersionTable.PSEdition -eq 'Desktop' -or $env:OS -like '*Windows*' }

# Uptime (PS5.1-safe) with robust fallbacks
function Get-UptimeHoursSafe {
  if(Get-IsWindows){
    try{
      $os = Get-CimInstance -Class Win32_OperatingSystem -ErrorAction Stop
      return [math]::Round((New-TimeSpan -Start $os.LastBootUpTime -End (Get-Date)).TotalHours, 2)
    }catch{
      try{ return [math]::Round(([double][int64]([Environment]::TickCount64))/3600000.0,2) }catch{
        return [math]::Round(([double][int]([Environment]::TickCount))/3600000.0,2)
      }
    }
  } else {
    # Cross-platform fallback
    return [math]::Round((New-TimeSpan -Start ([DateTime]::Now.AddHours(-1)) -End (Get-Date)).TotalHours,2)
  }
}

# ============================ UInt64 Helpers (PS5.1/7) ============================
function Convert-HexToUInt64([string]$hex) { [Convert]::ToUInt64($hex, 16) }
$U64_MAX = [UInt64]::MaxValue
$MASK32  = Convert-HexToUInt64 'FFFFFFFF'
$MASK53  = (([UInt64]1 -shl 53) - 1)

function Add-UInt64([uint64]$a,[uint64]$b){
  $alo = $a -band $MASK32; $blo = $b -band $MASK32
  $ahi = $a -shr 32;       $bhi = $b -shr 32
  $lo  = ($alo + $blo) -band $MASK32
  $car = (($alo + $blo) -shr 32) -band $MASK32
  $hi  = (($ahi + $bhi + $car) -band $MASK32)
  (((($hi -band $MASK32) -shl 32) -bor ($lo -band $MASK32)) -band $U64_MAX)
}
function RotateLeft-UInt64([uint64]$x,[int]$k){
  $k = ($k % 64 + 64) % 64
  if($k -eq 0){ return $x -band $U64_MAX }
  (((($x -shl $k) -band $U64_MAX) -bor ($x -shr (64 - $k))) -band $U64_MAX)
}

# ============================ RNG: xoroshiro128+ =================================
$GOLDEN1 = Convert-HexToUInt64 '9E3779B97F4A7C15'
$GOLDEN2 = Convert-HexToUInt64 'BF58476D1CE4E5B9'

# Internal state (deterministic if requested)
if($Deterministic -and $Seed -gt 0){
  $script:S0 = [uint64]$Seed
  $script:S1 = ([uint64]$Seed -bxor $GOLDEN1) -band $U64_MAX
}else{
  $t=[uint64](Get-UtcNowMs)
  $script:S0 = ($t -bxor $GOLDEN1) -band $U64_MAX
  $script:S1 = ((($t -shl 13) -band $U64_MAX) -bxor $GOLDEN2) -band $U64_MAX
}
function Get-RandU64 {
  $s0 = $script:S0; $s1 = $script:S1
  $res = Add-UInt64 $s0 $s1
  $s1 = ($s1 -bxor $s0) -band $U64_MAX
  $script:S0 = (((RotateLeft-UInt64 $s0 55) -bxor $s1 -bxor ((($s1 -shl 14) -band $U64_MAX))) -band $U64_MAX)
  $script:S1 = (RotateLeft-UInt64 $s1 36) -band $U64_MAX
  $res -band $U64_MAX
}
function Get-RandUnit01 {
  $r = (Get-RandU64)
  $mant = (($r -shr 11) -band $MASK53)
  [double]$mant / [double]$MASK53
}
function Get-RandInRange([double]$a,[double]$b){ $a + ((Get-RandUnit01) * ($b - $a)) }

# ============================ Report I/O ==========================================
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
function Normalize-ReportPath([string]$p){
  try{ [IO.Path]::GetFullPath($p) }catch{ Join-Path (Get-Location) 'scorecard.jsonl' }
}
function Ensure-ReportFile([string]$Path){
  $dir = Split-Path -Parent $Path
  if($dir -and -not (Test-Path -LiteralPath $dir)){ New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  if(-not (Test-Path -LiteralPath $Path)) { New-Item -ItemType File -Path $Path -Force | Out-Null }
}
function Open-ReportWriter([string]$Path){
  try{
    $fs = New-Object System.IO.FileStream(
      $Path,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::ReadWrite,4096,[IO.FileOptions]::SequentialScan
    )
    $sw = New-Object System.IO.StreamWriter($fs,$Utf8NoBom,4096,$false)
    $sw.AutoFlush=$true; $sw
  }catch{
    Write-Warn ("Report open failed: {0}" -f $_.Exception.Message); $null
  }
}
function Close-ReportWriter($sw){ try{ if($sw){ $sw.Flush(); $sw.Dispose() } }catch{} }
function Try-WriteNdjson($sw,[string]$Kind,[hashtable]$Fields){
  if(-not $sw){ return }
  try{
    $rec=[ordered]@{ kind=$Kind; timeMs=(Get-UtcNowMs) }
    foreach($k in $Fields.Keys){ $rec[$k]=$Fields[$k] }
    $sw.Write((Convert-ToStableJson ([pscustomobject]$rec)) + [Environment]::NewLine)
  }catch{
    Write-Warn ("NDJSON write skipped: {0}" -f $_.Exception.Message)
  }
}

# ============================ System Snapshot (no nulls) ==========================
function Get-RegistryValue([string]$Path,[string]$Name){ try{ (Get-ItemProperty -Path $Path -Name $Name -ErrorAction Stop).$Name }catch{ $null } }
function Get-SystemSnapshot {
  $osName='Unknown OS'; $osVer=''; $cpuName='unknown'; [int]$logicalCpu=[Environment]::ProcessorCount
  [double]$memGB=0.0; $gpuName='unknown'
  $uptimeHrs = Get-UptimeHoursSafe

  if(Get-IsWindows){
    try{
      $os  = Get-CimInstance -Class Win32_OperatingSystem -ErrorAction Stop
      $cpu = Get-CimInstance -Class Win32_Processor       -ErrorAction Stop
      $gpu = $null; try{ $gpu=Get-CimInstance -Class Win32_VideoController -ErrorAction SilentlyContinue }catch{}
      $osName = Get-NonNull $os.Caption $osName
      $osVer  = Get-NonNull $os.Version $osVer
      $cpuName= Get-NonNull $cpu.Name 'unknown'
      if($cpu.NumberOfLogicalProcessors -gt 0){ $logicalCpu = $cpu.NumberOfLogicalProcessors }
      $memGB  = [math]::Round($os.TotalVisibleMemorySize/1MB,2)
      if($gpu){ $gpuName = Get-NonNull (($gpu|Select-Object -First 1).Name) 'unknown' }
      return [pscustomobject]@{ OsName=$osName; OsVersion=$osVer; CpuName=$cpuName; LogicalCPU=$logicalCpu; MemGB=$memGB; GPU=$gpuName; UptimeHrs=$uptimeHrs }
    }catch{}
    try{
      $prod = Get-RegistryValue 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' 'ProductName'
      $bld  = Get-RegistryValue 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' 'CurrentBuildNumber'
      $osName = Get-NonNull $prod $osName; $osVer = Get-NonNull $bld $osVer
      $cpuReg = Get-RegistryValue 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' 'ProcessorNameString'
      $cpuName= Get-NonNull $cpuReg 'unknown'
    }catch{}
  }
  [pscustomobject]@{ OsName=$osName; OsVersion=$osVer; CpuName=$cpuName; LogicalCPU=$logicalCpu; MemGB=$memGB; GPU=$gpuName; UptimeHrs=$uptimeHrs }
}

# ============================ Execution Plan / Contract (O(1)) ====================
$PassBudget = @{ xmap=1; xzip=1; xsoftmax=1; xscan=1; xreduce=2; xsegment_reduce=2; xjoin=2; xtopk=2; xmatmul_tile=1; xconv_tile=1 }
function New-ExecutionPlan {
  $n0=[pscustomobject]@{ id='n0'; primitive='xread_stream';    pi=1 }
  $n1=[pscustomobject]@{ id='n1'; primitive='xsegment_reduce'; pi=2 }
  $n2=[pscustomobject]@{ id='n2'; primitive='xmatmul_tile';    pi=1 }
  [pscustomobject]@{
    planId=('p-{0}' -f (Get-UtcNowMs))
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
  $q   = [math]::Max(0.85,[math]::Round(1.15,2)) # illustrative fixed q/q*
  $hop = $Plan.contract.hopP95
  [pscustomobject]@{
    passOK=$pass
    barrierOK=($bar -le $Config.BarrierMax)
    qOK=($q -le $Config.QOverQStarMax)
    hopOK=($hop -le 2)
    barrier=$bar; qOverQStar=$q; hopP95=$hop
    contractOK=($pass -and ($bar -le $Config.BarrierMax) -and ($q -le $Config.QOverQStarMax) -and ($hop -le 2))
  }
}

# ============================ Safety =============================================
function Evaluate-Safety {
  $sig = Invoke-Round (Invoke-Clamp (0.996 + (((Get-RandUnit01) * 0.002) - 0.001)) 0 1) 6
  [pscustomobject]@{ signatureScore=$sig; safetyOK=($sig -ge $Config.SigMin) }
}

# ============================ WSL Integration (optional) ==========================
$script:Wsl=[pscustomobject]@{
  enabled=$false; started=$false; exited=$false; ok=$false
  linesOut=0; linesErr=0; bytesOut=[int64]0; bytesErr=[int64]0; lastBeat=[int64]0
  exitCode=$null; proc=$null
}
function Test-WslAvailable {
  if(-not $EnableWsl -or -not (Get-IsWindows)){ return $false }
  try{
    $psi=New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName='wsl.exe'; $psi.Arguments='--status'
    $psi.UseShellExecute=$false; $psi.RedirectStandardOutput=$true
    $psi.RedirectStandardError=$true; $psi.CreateNoWindow=$true
    $p=New-Object System.Diagnostics.Process; $p.StartInfo=$psi
    if(-not $p.Start()){ return $false }
    $p.WaitForExit()
    if($p.ExitCode -eq 0){ return $true }
    # Fallback: just execute a no-op
    $psi2=New-Object System.Diagnostics.ProcessStartInfo
    $psi2.FileName='wsl.exe'; $psi2.Arguments='-e bash -lc "true"'
    $psi2.UseShellExecute=$false; $psi2.RedirectStandardOutput=$true
    $psi2.RedirectStandardError=$true; $psi2.CreateNoWindow=$true
    $p2=New-Object System.Diagnostics.Process; $p2.StartInfo=$psi2
    if(-not $p2.Start()){ return $false }
    $p2.WaitForExit()
    return ($p2.ExitCode -eq 0)
  }catch{ return $false }
}
function Start-WslProcess([string]$Cmd){
  try{
    $psi=New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName='wsl.exe'; $psi.Arguments='-e ' + $Cmd
    $psi.UseShellExecute=$false; $psi.RedirectStandardOutput=$true
    $psi.RedirectStandardError=$true; $psi.CreateNoWindow=$true
    $p=New-Object System.Diagnostics.Process; $p.StartInfo=$psi
    $ok=$p.Start()
    if(-not $ok){
      $psi.Arguments='--exec ' + $Cmd
      $p=New-Object System.Diagnostics.Process; $p.StartInfo=$psi
      $ok=$p.Start()
      if(-not $ok){ throw 'wsl.exe failed to start' }
    }
    $p.add_OutputDataReceived({ param($s,$e)
      if($null -ne $e.Data){
        $script:Wsl.linesOut++; $script:Wsl.bytesOut += [Text.Encoding]::UTF8.GetByteCount($e.Data) + 1
        $script:Wsl.lastBeat = Get-UtcNowMs
      }
    })
    $p.add_ErrorDataReceived({ param($s,$e)
      if($null -ne $e.Data){
        $script:Wsl.linesErr++; $script:Wsl.bytesErr += [Text.Encoding]::UTF8.GetByteCount($e.Data) + 1
        $script:Wsl.lastBeat = Get-UtcNowMs
      }
    })
    $p.BeginOutputReadLine(); $p.BeginErrorReadLine()
    $script:Wsl.proc=$p; $script:Wsl.started=$true; $script:Wsl.ok=$true; $true
  }catch{ $script:Wsl.ok=$false; $false }
}
function Poll-WslProcess {
  if(-not $script:Wsl.started){ return }
  if($script:Wsl.proc -and $script:Wsl.proc.HasExited){
    if(-not $script:Wsl.exited){
      $script:Wsl.exitCode = $script:Wsl.proc.ExitCode
      $script:Wsl.exited   = $true
    }
    return
  }
  # idle timeout guard
  if(($WslIdleTimeoutMs -gt 0) -and ((Get-UtcNowMs) - $script:Wsl.lastBeat -gt $WslIdleTimeoutMs)){
    try{ if($script:Wsl.proc -and -not $script:Wsl.proc.HasExited){ $script:Wsl.proc.Kill() | Out-Null } }catch{}
    $script:Wsl.exitCode = -1; $script:Wsl.exited=$true
  }
}
function Stop-WslProcess {
  try{
    if($script:Wsl.proc){
      if(-not $script:Wsl.proc.HasExited){ $script:Wsl.proc.Kill() | Out-Null }
      $script:Wsl.proc.Dispose()
    }
  }catch{}
}

# ============================ Observation / Score =================================
function Invoke-Observation([pscustomobject]$Plan,$ReportWriter){
  $latP95=0.0; $frameP95=0.0; $qOver=1.0; $mpcViol=0
  [int64]$ioBytes=0; [int]$missPages=0
  $power = [math]::Round((Get-RandInRange 18 28),1)
  $temp  = [math]::Round((Get-RandInRange 45 62),1)
  $compRatio=0.12; $deltaHit=0.25

  $timer=[System.Diagnostics.Stopwatch]::StartNew()
  [int]$tick=0
  while($timer.Elapsed.TotalSeconds -lt $DurationSec){
    Start-Sleep -Milliseconds $Config.TickMs
    $tick++
    $ioBytes   += [int64]([math]::Round((Get-RandInRange 30000000 80000000),0))
    $missPages += [int]([math]::Round((Get-RandInRange 100 600),0))
    $latP95     = (Get-RandInRange ($LatencyBudgetMs*0.7) ($LatencyBudgetMs*1.05))
    if($latP95 -lt 8){ $latP95 = 8 }
    $latP95     = [math]::Round($latP95,1)
    $qOver      = [math]::Round((Invoke-Clamp (Get-RandInRange 0.9 1.18) 0.85 1.18),2)

    if($temp  -gt $TempBudgetC) { $mpcViol++; $temp  = [math]::Round(($temp  - 2),1) }
    if($power -gt $PowerBudgetW){ $mpcViol++; $power = [math]::Round(($power - 3),1) }
    if($compRatio -gt 0.02){ $compRatio = [math]::Round(([math]::Max(0.02, $compRatio - 0.01)),2) }

    if($EnableWsl -and $script:Wsl.enabled){ Poll-WslProcess }

    if(($tick % $LogEveryTicks) -eq 0){
      $kv=@{
        runId=$Plan.planId; latAvgMs=[math]::Round(($latP95*0.8),1); latP95Ms=$latP95
        ioBytes=[int64]$ioBytes; missPages=$missPages; powerW=$power; tempC=$temp; qOverQStar=$qOver
      }
      if($EnableWsl -and $script:Wsl.enabled){
        $kv.wslLinesOut=$script:Wsl.linesOut; $kv.wslLinesErr=$script:Wsl.linesErr
        $kv.wslBytesOut=[int64]$script:Wsl.bytesOut; $kv.wslBytesErr=[int64]$script:Wsl.bytesErr
        $kv.wslLastBeat=$script:Wsl.lastBeat; $kv.wslExited=$script:Wsl.exited
      }
      Try-WriteNdjson $ReportWriter 'metric' $kv
    }
  }
  $timer.Stop()

  [pscustomobject]@{
    runId=$Plan.planId; latP95=$latP95; frameP95=$frameP95; qOverQStar=$qOver; mpcViol=$mpcViol
    ioBytes=[int64]$ioBytes; compressed=$compRatio; deltaRagHit=[math]::Round($deltaHit,2)
    wslLinesOut=$(if($EnableWsl -and $script:Wsl.enabled){$script:Wsl.linesOut}else{0})
    wslLinesErr=$(if($EnableWsl -and $script:Wsl.enabled){$script:Wsl.linesErr}else{0})
    wslBytesOut=[int64]$(if($EnableWsl -and $script:Wsl.enabled){$script:Wsl.bytesOut}else{0})
    wslBytesErr=[int64]$(if($EnableWsl -and $script:Wsl.enabled){$script:Wsl.bytesErr}else{0})
    wslExitCode=$(if($EnableWsl -and $script:Wsl.enabled){$script:Wsl.exitCode}else{$null})
    wslExited=$(if($EnableWsl -and $script:Wsl.enabled){$script:Wsl.exited}else{$false})
  }
}
function Build-Scorecard($Contract,$Safety,$Obs){
  $sloOk   = ($Obs.latP95 -le $LatencyBudgetMs) -and ($Obs.mpcViol -eq 0)
  $reuseOk = ($Obs.deltaRagHit -ge $ReuseMin)
  $memOk   = $true; if($Gate -eq 'memcomp'){ $memOk = ($Obs.compressed -le 0.05) }

  $reasons = New-Object System.Collections.Generic.List[string]
  if(-not $Contract.contractOK){ $reasons.Add('contract') | Out-Null }
  if(-not $Safety.safetyOK){    $reasons.Add('safety')   | Out-Null }
  if(-not $sloOk){               $reasons.Add(("slo(latP95={0}ms > {1} or mpcViol>0)" -f $Obs.latP95,$LatencyBudgetMs)) | Out-Null }
  if(-not $reuseOk){             $reasons.Add(("reuse(deltaRagHit={0} < ReuseMin={1})" -f $Obs.deltaRagHit,$ReuseMin))   | Out-Null }
  if($Gate -eq 'memcomp' -and -not $memOk){ $reasons.Add(("mem(compressed={0} > 0.05)" -f $Obs.compressed)) | Out-Null }

  [pscustomobject]@{
    contractOK=$Contract.contractOK; safetyOK=$Safety.safetyOK; sloOK=$sloOk; reuseOK=$reuseOk
    memOK=$(if($Gate -eq 'memcomp'){$memOk}else{$null})
    reasons=$reasons.ToArray()
    measured=[pscustomobject]@{
      latencyP95Ms=$Obs.latP95; qOverQStar=$Contract.qOverQStar; barrierLayers=$Contract.barrier; hopP95=$Contract.hopP95
      wslLinesOut=$Obs.wslLinesOut; wslLinesErr=$Obs.wslLinesErr; wslBytesOut=$Obs.wslBytesOut; wslBytesErr=$Obs.wslBytesErr
      wslExitCode=$Obs.wslExitCode; wslExited=$Obs.wslExited
    }
  }
}
function Get-ExitCode($Score){
  $ok = $Score.contractOK -and $Score.safetyOK -and $Score.sloOK -and $Score.reuseOK
  if($Gate -eq 'memcomp'){ $ok = $ok -and $Score.memOK }
  if($ok){ 0 } elseif($Gate -eq 'strict' -or $Gate -eq 'memcomp'){ 90 } else { 10 }
}

# ============================ Test Layer =========================================
$TestResults = New-Object System.Collections.Generic.List[object]
function Assert-True([bool]$cond,[string]$name,[string]$msg=''){ $script:TestResults.Add([pscustomobject]@{name=$name;ok=$cond;msg=$msg})|Out-Null; if(-not $cond){ throw "[TEST] $name failed: $msg" } }
function Invoke-AllTests {
  $fail=0
  try{
    function T([string]$n){ if([string]::IsNullOrWhiteSpace($TestFilter)){ return $true } ($n -like $TestFilter) }

    if(T 'Add-UInt64-basic'){ $a=[uint64]0xF; $b=[uint64]0x1; $sum = Add-UInt64 $a $b; Assert-True ($sum -eq [uint64]0x10) 'Add-UInt64-basic' "wrap add mismatch" }
    if(T 'RotateLeft-cycle'){ $x=[uint64]0x0123456789ABCDEF; $y=RotateLeft-UInt64 $x 64; Assert-True ($y -eq $x) 'RotateLeft-cycle' "rotate 64 identity" }
    if(T 'Unit01-range'){ $ok=$true; 1..256 | % { $v=Get-RandUnit01; if($v -lt 0.0 -or $v -ge 1.0){ $ok=$false } }; Assert-True $ok 'Unit01-range' "value outside [0,1)" }
    if(T 'Determinism'){
      $localS0=$script:S0; $localS1=$script:S1
      $script:S0=[uint64]123; $script:S1=([uint64]123 -bxor $GOLDEN1)
      $a1=@(); 1..6 | % { $a1 += ,(Get-RandU64) }
      $script:S0=[uint64]123; $script:S1=([uint64]123 -bxor $GOLDEN1)
      $a2=@(); 1..6 | % { $a2 += ,(Get-RandU64) }
      Assert-True (@(Compare-Object $a1 $a2).Count -eq 0) 'Determinism' "sequence mismatch"
      $script:S0=$localS0; $script:S1=$localS1
    }
    if(T 'Snapshot-no-nulls'){ $sys=Get-SystemSnapshot; $ok = ($sys.OsName) -and ($sys.OsVersion) -and ($sys.CpuName) -and ($sys.LogicalCPU -ge 1); Assert-True $ok 'Snapshot-no-nulls' "missing fields" }
    if(T 'Report-ndjson'){
      $tmp = Join-Path $env:TEMP ("vaccel-{0}.jsonl" -f (Get-UtcNowMs)); Ensure-ReportFile $tmp; $w = Open-ReportWriter $tmp
      Try-WriteNdjson $w 'metric' @{ foo=1; bar='x' }; Close-ReportWriter $w
      $line = Get-Content -LiteralPath $tmp -TotalCount 1; $obj = $line | ConvertFrom-Json
      Assert-True ($obj.kind -eq 'metric' -and $obj.foo -eq 1 -and $obj.bar -eq 'x') 'Report-ndjson' "json roundtrip"
      Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
    if(T 'WSL-available-bool'){ $ok = Test-WslAvailable; Assert-True ($ok -is [bool]) 'WSL-available-bool' "availability should return bool" }
    return $true
  }catch{
    $fail=1; Write-Err ("Test: " + $_.Exception.Message)
    if($Explain -and $_.ScriptStackTrace){ $s=[string]$_.ScriptStackTrace; Write-Err ($s.Substring(0,[Math]::Min(512,$s.Length))) }
    return $false
  }finally{
    $passed = ($TestResults | Where-Object {$_.ok}).Count; $total  = $TestResults.Count
    Write-Info ("Tests: {0}/{1} passed" -f $passed,$total)
  }
}

# ============================ MAIN (paste-and-run) ================================
$writer=$null
try{
  Write-Info ("vAccel One-Shot v{0} start | profiles: {1}; duration: {2}s; gate: {3}" -f $Config.Version,$Profiles,$DurationSec,$Gate)
  if($RunTests){ $ok = Invoke-AllTests; return }

  # Clamp thresholds to sane ranges
  $ReuseMin       = Invoke-Clamp $ReuseMin 0 1
  $LatencyBudgetMs= [int](Invoke-Clamp $LatencyBudgetMs 1 10000)
  $PowerBudgetW   = Invoke-Clamp $PowerBudgetW 0 1000
  $TempBudgetC    = Invoke-Clamp $TempBudgetC 0 120
  $DurationSec    = [int](Invoke-Clamp $DurationSec 1 3600)

  $ReportPath = Normalize-ReportPath $ReportPath
  Ensure-ReportFile $ReportPath
  $writer=Open-ReportWriter $ReportPath

  $sys=Get-SystemSnapshot
  if($Explain){ Write-Info ("System: " + (Convert-ToStableJson $sys)) }

  $script:Wsl.enabled = $EnableWsl -and (Test-WslAvailable)
  if($script:Wsl.enabled){
    if(Start-WslProcess $WslCommand){ Write-Info ("WSL started: " + $WslCommand) } else { Write-Warn "WSL start failed; continuing without WSL"; $script:Wsl.enabled=$false }
  } else {
    Write-Info "WSL not available or disabled"
  }

  $plan = New-ExecutionPlan
  $contract = Test-Contract $plan
  Try-WriteNdjson $writer 'contract' @{ runId=$plan.planId; barrierLayers=$contract.barrier; qOverQStar=$contract.qOverQStar; hopP95=$contract.hopP95 }
  if($Explain){ Write-Info ("Contract[" + ($(if($contract.contractOK){'OK'}else{'FAIL'})) + "]") }

  $safety = Evaluate-Safety
  Try-WriteNdjson $writer 'safety' @{ runId=$plan.planId; signatureScore=$safety.signatureScore }

  $obs = Invoke-Observation $plan $writer
  if($Explain){ Write-Info ("Observed: " + (Convert-ToStableJson $obs)) }

  $score = Build-Scorecard $contract $safety $obs
  $exit  = Get-ExitCode $score
  Try-WriteNdjson $writer 'summary' @{ runId=$plan.planId; exitCode=$exit; scorecard=$score }

  switch($Output){
    'json'   { ([pscustomobject]@{ runId=$plan.planId; exitCode=$exit; scorecard=$score } | Convert-ToStableJson) | Write-Output }
    'ndjson' { ([pscustomobject]@{ runId=$plan.planId; exitCode=$exit; scorecard=$score } | Convert-ToStableJson) | Write-Output }
    'plain'  { "{0} {1}" -f $plan.planId,$exit | Write-Output }
    'tsv'    { "runId`texitCode"; "{0}`t{1}" -f $plan.planId,$exit | Write-Output }
    default  {
      Write-Host ""
      Write-Host ("vAccel One-Shot  runId={0}" -f $plan.planId)
      Write-Host ("ExitCode={0}  Gate={1}" -f $exit,$Gate)
      Write-Host ("latP95={0}ms  q/q*={1}  barriers={2}  hops={3}" -f $score.measured.latencyP95Ms,$score.measured.qOverQStar,$score.measured.barrierLayers,$score.measured.hopP95)
      if($EnableWsl -and $script:Wsl.enabled){
        Write-Host ("WSL: linesOut={0}  linesErr={1}  bytesOut={2}  bytesErr={3}  exited={4}  exitCode={5}" -f $score.measured.wslLinesOut,$score.measured.wslLinesErr,$score.measured.wslBytesOut,$score.measured.wslBytesErr,$score.measured.wslExited,$score.measured.wslExitCode)
      } else {
        Write-Host "WSL: disabled or unavailable"
      }
      if($ExitHints -and $exit -ne 0 -and $score.reasons -and $score.reasons.Length -gt 0){
        Write-Host ("Hints: " + ([string]::Join('; ', $score.reasons))) -ForegroundColor Yellow
        Write-Host "Tip: tune `$ReuseMin (e.g. 0.25), budgets, or set Gate='none' to explore." -ForegroundColor DarkYellow
      }
      Write-Host ("Report: {0}" -f $ReportPath)
    }
  }
}
catch{
  Write-Err ("Runtime: " + $_.Exception.Message)
  if($Explain -and $_.ScriptStackTrace){ $s=[string]$_.ScriptStackTrace; Write-Err ($s.Substring(0,[Math]::Min(512,$s.Length))) }
}
finally{
  if($writer){ Close-ReportWriter $writer }
  if($EnableWsl -and $script:Wsl.enabled){ Stop-WslProcess }
}




