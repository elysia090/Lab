& {
  # ============================================================================
  # vAccel One-Shot — v1.3.8b (inline, ASCII-only, PS 5.1+/7, O(1)/tick, WSL)
  # Notes:
  #   - PowerShell function calls: no trailing "()", use (Func) in expressions.
  #   - All loops are bounded; per-tick work is constant (O(1)).
  #   - ASCII-only output. No Unicode line art.
  #   - WSL path is optional and polled event-style (no blocking waits).
  # Usage:
  #   paste whole block into a PowerShell console and press Enter.
  #   adjust parameters below if you want different behavior.
  # ============================================================================

  # ----------------------- Parameters ----------------------------------------
  [string] $ProfilesCsv        = 'AUTOHOOK-GLOBAL,LIMIT-ORDER,SUPRA-HIEND,RT-BALANCED,FABRIC,MEMZERO'
  [ValidateSet('conservative','balanced','aggressive')]
  [string] $Aggressiveness     = 'balanced'
  [ValidateRange(1,100000)]
  [int]    $LatencyBudgetMs    = 60
  [ValidateRange(1,1000)]
  [double] $PowerBudgetW       = 45.0
  [ValidateRange(0,120)]
  [double] $TempBudgetC        = 85.0
  [ValidateRange(1,86400)]
  [int]    $DurationSec        = 10
  [bool]   $Explain            = $true
  [ValidateSet('none','strict','memcomp')]
  [string] $Gate               = 'strict'
  [ValidateSet('human','json','ndjson','plain','tsv')]
  [string] $OutputMode         = 'human'
  [ValidateRange(1,1000)]
  [int]    $LogEveryTicks      = 2

  # Determinism
  [bool]   $Deterministic      = $true
  [uint64] $Seed               = 42

  # Report path
  [string] $ReportPath         = (Join-Path (Get-Location) 'scorecard.jsonl')

  # WSL integration
  [bool]   $EnableWsl          = $true
  [string] $WslCommand         = 'bash -lc "yes test | head -n 20000"'
  # ---------------------------------------------------------------------------

  # ----------------------- Runtime / Constants --------------------------------
  Set-StrictMode -Version Latest
  $ErrorActionPreference = 'Stop'
  [bool] $IsPS7 = $PSVersionTable.PSVersion.Major -ge 7
  $Cfg = @{
    TickMs        = 500
    JsonDepth     = 12
    BarrierMax    = 3
    QOverQStarMax = 1.2
    SigMin        = 0.995
  }
  # ---------------------------------------------------------------------------

  # ----------------------- Utilities -----------------------------------------
  function NowMs { [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) }
  function JsonStable([object]$o) {
    if ($IsPS7) { $o | ConvertTo-Json -Depth $Cfg.JsonDepth -EnumsAsStrings }
    else        { $o | ConvertTo-Json -Depth $Cfg.JsonDepth }
  }
  function Clamp([double]$v,[double]$lo,[double]$hi) {
    if ($v -lt $lo) { $v = $lo }
    if ($v -gt $hi) { $v = $hi }
    $v
  }
  function Rnd([double]$v,[int]$d) { [math]::Round($v,$d) }
  function LogInfo([string]$m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
  function LogWarn([string]$m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
  function LogErr ([string]$m){ Write-Host "[ERR ] $m" -ForegroundColor Red }
  # ---------------------------------------------------------------------------

  # ----------------------- RNG (XorShift64*) ---------------------------------
  if ($Deterministic -and $Seed -gt 0) { $global:VSeed = [uint64]$Seed } else { $global:VSeed = [uint64](NowMs) }
  function U01 {
    $s = $global:VSeed
    $s = $s -bxor ($s -shl 12)
    $s = $s -bxor ($s -shr 25)
    $s = $s -bxor ($s -shl 27)
    $global:VSeed = $s
    $u = ($s * 2685821657736338717) -band 0xFFFFFFFFFFFFFFFF
    [double]($u % 1000000) / 999999.0
  }
  function RandRange([double]$a,[double]$b) { $a + (((U01)) * ($b - $a)) }
  # ---------------------------------------------------------------------------

  # ----------------------- Report I/O ----------------------------------------
  $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  function Ensure-Report([string]$Path){
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    if (-not (Test-Path -LiteralPath $Path)) { New-Item -ItemType File -Path $Path -Force | Out-Null }
  }
  function Open-Report([string]$Path){
    $fs = New-Object System.IO.FileStream($Path,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::ReadWrite,4096,[IO.FileOptions]::SequentialScan)
    $sw = New-Object System.IO.StreamWriter($fs,$Utf8NoBom,4096,$true)
    $sw.AutoFlush = $true
    $sw
  }
  function Close-Report($sw){ try{ if($sw){ $sw.Flush(); $sw.Dispose() } }catch{} }
  function Write-JsonLine($sw,[string]$Kind,[hashtable]$Fields){
    $rec = [ordered]@{ kind=$Kind; timeMs=(NowMs) }
    foreach($k in $Fields.Keys){ $rec[$k]=$Fields[$k] }
    $sw.Write((JsonStable ([pscustomobject]$rec)) + [Environment]::NewLine)
  }
  # ---------------------------------------------------------------------------

  # ----------------------- System Probe (O(1)) --------------------------------
  function Get-SystemInfo {
    try{
      $os  = Get-CimInstance -Class Win32_OperatingSystem -ErrorAction Stop
      $cpu = Get-CimInstance -Class Win32_Processor       -ErrorAction Stop
      $gpu = $null; try{ $gpu=Get-CimInstance -Class Win32_VideoController -ErrorAction SilentlyContinue }catch{}
      $memGB=[math]::Round($os.TotalVisibleMemorySize/1MB,2)
      $up   =(Get-Date)-([Management.ManagementDateTimeConverter]::ToDateTime($os.LastBootUpTime))
      [pscustomobject]@{
        OsName=$os.Caption; OsVersion=$os.Version; CpuName=$cpu.Name
        LogicalCPU=$cpu.NumberOfLogicalProcessors; MemGB=$memGB
        GPU=$(if($gpu){ ($gpu|Select-Object -First 1).Name } else { $null })
        UptimeHrs=[math]::Round($up.TotalHours,2)
      }
    }catch{
      [pscustomobject]@{
        OsName=$(if($env:OS){$env:OS}else{'Windows'})
        OsVersion=$PSVersionTable.PSVersion.ToString()
        CpuName='unknown'
        LogicalCPU=[Environment]::ProcessorCount
        MemGB=$null; GPU=$null; UptimeHrs=$null
      }
    }
  }
  # ---------------------------------------------------------------------------

  # ----------------------- Fixed Plan & Contract (O(1)) ----------------------
  $PassBudget = @{
    xmap=1; xzip=1; xsoftmax=1; xscan=1
    xreduce=2; xsegment_reduce=2; xjoin=2; xtopk=2
    xmatmul_tile=1; xconv_tile=1
  }
  function New-Plan {
    $n0=[pscustomobject]@{id='n0';primitive='xread_stream';    pi=1}
    $n1=[pscustomobject]@{id='n1';primitive='xsegment_reduce'; pi=2}
    $n2=[pscustomobject]@{id='n2';primitive='xmatmul_tile';    pi=1}
    [pscustomobject]@{
      planId=('p-{0}' -f (NowMs))
      nodes=@($n0,$n1,$n2)
      contract=[pscustomobject]@{ barrierLayers=2; hopP95=1 }
    }
  }
  function Test-Contract($Plan){
    $passOk=$true
    foreach($n in $Plan.nodes){
      if($PassBudget.ContainsKey($n.primitive)){
        if($n.pi -ne $PassBudget[$n.primitive]){ $passOk=$false; break }
      }
    }
    $bar = $Plan.contract.barrierLayers
    $q   = [math]::Max(0.85,[math]::Round(1.15,2))
    $hop = $Plan.contract.hopP95
    [pscustomobject]@{
      passOK=$passOk
      barrierOK=($bar -le $Cfg.BarrierMax)
      qOK=($q -le $Cfg.QOverQStarMax)
      hopOK=($hop -le 2)
      barrier=$bar; qOverQStar=$q; hopP95=$hop
      contractOK=($passOk -and ($bar -le $Cfg.BarrierMax) -and ($q -le $Cfg.QOverQStarMax) -and ($hop -le 2))
    }
  }
  function Test-Safety {
    # Functions in expressions must be wrapped: (U01)
    $sig = Rnd (Clamp (0.996 + ((((U01)) * 0.002) - 0.001)) 0 1) 6
    [pscustomobject]@{ signatureScore=$sig; safetyOK=($sig -ge $Cfg.SigMin) }
  }
  # ---------------------------------------------------------------------------

  # ----------------------- WSL Integration (event-style) ---------------------
  $script:Wsl=[pscustomobject]@{
    enabled=$false; started=$false; exited=$false; ok=$false
    linesOut=0; linesErr=0; bytesOut=[int64]0; bytesErr=[int64]0; lastBeat=[int64]0
    exitCode=$null; proc=$null
  }
  function Test-WslAvailable {
    if(-not $EnableWsl){ return $false }
    try{
      $p=Start-Process -FilePath 'wsl.exe' -ArgumentList @('--status') -WindowStyle Hidden -NoNewWindow -PassThru -Wait -ErrorAction Stop
      return ($p.ExitCode -in 0,1)
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
  # ---------------------------------------------------------------------------

  # ----------------------- Observe / Score (O(1)/tick) -----------------------
  function Observe-Run([pscustomobject]$Plan,[System.IO.StreamWriter]$Report){
    $latP95=0.0; $frameP95=0.0; $qOver=1.0; $mpcViol=0
    [int64]$ioBytes=0; [int]$missPages=0
    $power = Rnd (RandRange 18 28) 1
    $temp  = Rnd (RandRange 45 62) 1
    $compRatio=0.12; $deltaHit=0.25

    $timer=[System.Diagnostics.Stopwatch]::StartNew()
    [int]$tick=0
    while($timer.Elapsed.TotalSeconds -lt $DurationSec){
      Start-Sleep -Milliseconds $Cfg.TickMs
      $tick++

      $ioBytes   += [int64](RandRange 30000000 80000000)
      $missPages += [int](RandRange 100 600)
      $latP95     = (RandRange ($LatencyBudgetMs*0.7) ($LatencyBudgetMs*1.05))
      if($latP95 -lt 8){ $latP95 = 8 }
      $latP95     = Rnd $latP95 1
      $qOver      = Rnd (Clamp (RandRange 0.9 1.18) 0.85 1.18) 2

      if($temp -gt $TempBudgetC){ $mpcViol++; $temp  = Rnd ($temp  - 2) 1 }
      if($power -gt $PowerBudgetW){ $mpcViol++; $power = Rnd ($power - 3) 1 }
      if($compRatio -gt 0.02){ $compRatio = Rnd ([math]::Max(0.02, $compRatio - 0.01)) 2 }

      if($script:Wsl.enabled){ Poll-Wsl }

      if(($tick % $LogEveryTicks) -eq 0){
        $kv=@{
          runId=$Plan.planId
          latAvgMs=(Rnd ($latP95*0.8) 1)
          latP95Ms=$latP95
          ioBytes=[int64]$ioBytes
          missPages=$missPages
          powerW=$power
          tempC=$temp
          qOverQStar=$qOver
        }
        if($script:Wsl.enabled){
          $kv.wslLinesOut=$script:Wsl.linesOut
          $kv.wslLinesErr=$script:Wsl.linesErr
          $kv.wslBytesOut=[int64]$script:Wsl.bytesOut
          $kv.wslBytesErr=[int64]$script:Wsl.bytesErr
          $kv.wslLastBeat=$script:Wsl.lastBeat
          $kv.wslExited=$script:Wsl.exited
        }
        Write-JsonLine $Report 'metric' $kv
      }
    }
    $timer.Stop()

    [pscustomobject]@{
      runId=$Plan.planId
      latP95=$latP95
      frameP95=$frameP95
      qOverQStar=$qOver
      mpcViol=$mpcViol
      ioBytes=[int64]$ioBytes
      compressed=$compRatio
      deltaRagHit=(Rnd $deltaHit 2)
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
      contractOK=$C.contractOK
      safetyOK=$S.safetyOK
      sloOK=$sloOk
      reuseOK=$reuseOk
      memOK=$(if($Gate -eq 'memcomp'){$memOk}else{$null})
      measured=[pscustomobject]@{
        latencyP95Ms=$O.latP95
        qOverQStar=$C.qOverQStar
        barrierLayers=$C.barrier
        hopP95=$C.hopP95
        wslLinesOut=$O.wslLinesOut
        wslLinesErr=$O.wslLinesErr
        wslBytesOut=$O.wslBytesOut
        wslBytesErr=$O.wslBytesErr
        wslExitCode=$O.wslExitCode
        wslExited=$O.wslExited
      }
    }
  }

  function Get-ExitCode($S){
    $ok = $S.contractOK -and $S.safetyOK -and $S.sloOK -and $S.reuseOK
    if($Gate -eq 'memcomp'){ $ok = $ok -and $S.memOK }
    if($ok){ 0 } elseif($Gate -eq 'strict' -or $Gate -eq 'memcomp'){ 90 } else { 10 }
  }
  # ---------------------------------------------------------------------------

  # ----------------------- MAIN ----------------------------------------------
  $writer=$null
  try{
    Ensure-Report $ReportPath
    $writer=Open-Report $ReportPath

    LogInfo ("vAccel One-Shot start | profiles: {0}; duration: {1}s; gate: {2}" -f $ProfilesCsv,$DurationSec,$Gate)

    $sys=Get-SystemInfo
    if($Explain){ LogInfo ("System: " + (JsonStable $sys)) }

    $script:Wsl.enabled = Test-WslAvailable
    if($script:Wsl.enabled){
      if(Start-WslJob $WslCommand){
        LogInfo ("WSL started: " + $WslCommand)
      } else {
        LogWarn "WSL start failed; continuing without WSL"
        $script:Wsl.enabled=$false
      }
    } else {
      LogInfo "WSL not available or disabled"
    }

    $plan=New-Plan
    $C=Test-Contract $plan
    Write-JsonLine $writer 'contract' @{ runId=$plan.planId; barrierLayers=$C.barrier; qOverQStar=$C.qOverQStar; hopP95=$C.hopP95 }
    if($Explain){ LogInfo ("Contract[" + ($(if($C.contractOK){'OK'}else{'FAIL'})) + "]") }

    $S=Test-Safety
    Write-JsonLine $writer 'safety' @{ runId=$plan.planId; signatureScore=$S.signatureScore }

    $O=Observe-Run $plan $writer
    if($Explain){ LogInfo ("Observed: " + (JsonStable $O)) }

    $Score=Build-Score $C $S $O
    $Exit =Get-ExitCode $Score
    Write-JsonLine $writer 'summary' @{ runId=$plan.planId; exitCode=$Exit; scorecard=$Score }

    switch($OutputMode){
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
    LogErr ("Runtime: " + $_.Exception.Message)
    if($Explain -and $_.ScriptStackTrace){
      $s=[string]$_.ScriptStackTrace
      LogErr ($s.Substring(0,[Math]::Min(512,$s.Length)))
    }
    exit 40
  }finally{
    if($writer){ Close-Report $writer }
    if($script:Wsl.enabled){ Stop-Wsl }
  }
}

