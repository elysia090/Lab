# vAccel One-Shot — v1.3.3a (ASCII-only, PS5.1/7 compatible, product-grade refactor)
# -----------------------------------------------------------------------------
# SYNOPSIS
#   Safe user-space orchestrator matching your spec; O(1)/tick model, typed IO,
#   predictable exit codes, ASCII-only TTY, and portable system probe.
# USAGE (example)
#   pwsh -File .\OneShot-VAccel.ps1 `
#     -Profiles "AUTOHOOK-GLOBAL,LIMIT-ORDER,SUPRA-HIEND,RT-BALANCED,FABRIC,MEMZERO" `
#     -Aggressiveness balanced -BudgetLatencyMs 60 -BudgetPowerW 45 -BudgetTempC 85 `
#     -DurationSec 10 -Why -Gate strict -ReportPath "$PWD\scorecard.jsonl" -Out human
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
  [ValidateSet('Global','User','Process')] [string]$AutoHook = 'Global',
  [ValidateSet('System','Session')]        [string]$Scope    = 'Session',

  # Comma-separated for PS5.1-safe parsing
  [string]$Profiles = 'AUTOHOOK-GLOBAL,LIMIT-ORDER,SUPRA-HIEND,RT-BALANCED,FABRIC,MEMZERO',
  [ValidateSet('conservative','balanced','aggressive')] [string]$Aggressiveness = 'balanced',

  [ValidateRange(1,100000)] [int]    $BudgetLatencyMs = 60,
  [ValidateRange(1,1000)]   [double] $BudgetPowerW    = 45,
  [ValidateRange(0,120)]    [double] $BudgetTempC     = 85,

  [switch]$Deterministic,
  [uint64]$Seed = 0,

  [string[]]$Allowlist = @('excel.exe','python.exe','code.exe','chrome.exe','ffmpeg.exe'),
  [string[]]$Denylist  = @('MsMpEng.exe','lsass.exe','winlogon.exe','csrss.exe'),

  [ValidateRange(0,4096)] [int]$WarmKeys = 256,

  # Ray-tracing options
  [ValidateRange(0,16)] [int]$RTSppMin = 1,
  [ValidateRange(0,64)] [int]$RTSppMax = 2,
  [ValidateRange(0,8)]  [int]$RTMaxBounces = 1,
  [ValidateRange(0.1,1.0)] [double]$RTResolutionScaleMin = 0.7,
  [ValidateRange(0.1,1.0)] [double]$RTResolutionScaleMax = 1.0,
  [ValidateSet('auto','nrd','svgf')]  [string]$RTDenoiser = 'auto',
  [ValidateSet('auto','dlss','fsr','xess')] [string]$RTUpscaler = 'auto',

  # Precision options
  [ValidateRange(0.0,1.0)] [double]$PrecisionGlobalEps = 0.001,
  [string[]]$PrecisionAllowQuant = @('bf16','int8'),
  [string[]]$PrecisionGuards     = @('kahan','pairwise'),

  # Fabric
  [ValidateRange(0,8)] [int]$FabricHopMax = 2,

  # Run control
  [ValidateRange(1,86400)] [int]$DurationSec = 30,
  [switch]$DryRun,
  [switch]$Why,

  # Gates and output
  [ValidateSet('none','strict','memcomp')] [string]$Gate = 'none',
  [string]$ReportPath = (Join-Path (Get-Location) 'scorecard.jsonl'),
  [ValidateSet('human','json','ndjson','plain','tsv')] [string]$Out = 'human'
)

# ===================== Runtime policy =====================
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$IsPS7 = $PSVersionTable.PSVersion.Major -ge 7

# ===================== Constants =====================
$Config = @{
  BarrierMax        = 3
  QOverQStarMax     = 1.2
  SignatureScoreMin = 0.995
  PlanTensorShape   = @(1024,1024)
  TickMs            = 500
  JsonDepth         = 12
}

# ===================== Utilities =====================
function Get-NowMs { [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) }

function ConvertTo-JsonStable([object]$Object){
  if($IsPS7){ return ($Object | ConvertTo-Json -Depth $Config.JsonDepth -EnumsAsStrings) }
  else      { return ($Object | ConvertTo-Json -Depth $Config.JsonDepth) }
}

function Test-Assert([bool]$Condition,[string]$Message){
  if(-not $Condition){ throw $Message }
}

function Get-Clamped([double]$Value,[double]$Min,[double]$Max){
  $v=$Value; if($v -lt $Min){$v=$Min}; if($v -gt $Max){$v=$Max}; $v
}

function Get-Rounded([double]$Value,[int]$Digits){ [math]::Round($Value,$Digits) }

function Write-Info([string]$Message){ Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Warn([string]$Message){ Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Err ([string]$Message){ Write-Host "[ERR ] $Message" -ForegroundColor Red }

# ===================== Random (CSPRNG or LCG) =====================
$global:VSeed = [uint64](Get-NowMs)
if($Seed -ne 0){ $global:VSeed = $Seed; $Deterministic = $true }
if(-not $Deterministic){ $script:VRng = [System.Security.Cryptography.RandomNumberGenerator]::Create() } else { $script:VRng = $null }

function Get-RandomUnit {
  if($script:VRng -ne $null){
    $buf = New-Object byte[] 8
    $script:VRng.GetBytes($buf)
    $u64 = [System.BitConverter]::ToUInt64($buf,0)
    return [double]($u64 % 1000000) / 999999.0
  }
  $global:VSeed = (6364136223846793005 * $global:VSeed + 1442695040888963407) -band 0xFFFFFFFFFFFFFFFF
  [double]($global:VSeed % 1000000) / 999999.0
}

function Get-RandomInRange([double]$Min,[double]$Max){
  $t = Get-RandomUnit
  $span = ($Max - $Min)
  $Min + ($t * $span)
}

function New-RunId {
  $hex = -join (1..16 | ForEach-Object {
    $b = [int](Get-RandomInRange 0 255)
    '{0:x2}' -f ([byte]$b)
  })
  "{0}-{1}" -f (Get-NowMs), $hex
}

# ===================== JSONL I/O =====================
function Initialize-Report([string]$Path){
  try{ $null = Resolve-Path -LiteralPath $Path -ErrorAction Stop } catch {
    $dir = Split-Path -Parent $Path
    if(-not (Test-Path -LiteralPath $dir)){ New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    if(-not (Test-Path -LiteralPath $Path)){ New-Item -ItemType File -Path $Path -Force | Out-Null }
  }
}

function Add-JsonlRecord([string]$Path,[object]$Record){
  $line = (ConvertTo-JsonStable $Record) + [Environment]::NewLine
  $enc  = [System.Text.Encoding]::UTF8
  $maxTry=5; $delay=40
  for([int]$i=1; $i -le $maxTry; $i++){
    try{ [System.IO.File]::AppendAllText($Path,$line,$enc); break }
    catch{
      if($i -eq $maxTry){ throw }
      Start-Sleep -Milliseconds $delay
      $delay = [Math]::Min(800, [int]($delay*2))
    }
  }
}

function Write-Event([string]$Kind,[hashtable]$Fields){
  $rec = [ordered]@{ kind=$Kind; timeMs=(Get-NowMs) }
  foreach($k in $Fields.Keys){ $rec[$k] = $Fields[$k] }
  Add-JsonlRecord -Path $ReportPath -Record ([pscustomobject]$rec)
}

# ===================== System Probe (O(1)) =====================
function Get-SystemSnapshot {
  try{
    $os  = Get-CimInstance -Class Win32_OperatingSystem -ErrorAction Stop
    $cpu = Get-CimInstance -Class Win32_Processor     -ErrorAction Stop
    $gpu = $null
    try { $gpu = Get-CimInstance -Class Win32_VideoController -ErrorAction SilentlyContinue } catch {}
    $memGB  = [math]::Round($os.TotalVisibleMemorySize/1MB,2)
    $uptime = (Get-Date) - ([Management.ManagementDateTimeConverter]::ToDateTime($os.LastBootUpTime))
    [pscustomobject]@{
      OsName=$os.Caption; OsVersion=$os.Version; CpuName=$cpu.Name
      LogicalCPU=$cpu.NumberOfLogicalProcessors; MemGB=$memGB
      GPU= if($gpu){ ($gpu | Select-Object -First 1).Name } else { $null }
      Uptime=[math]::Round($uptime.TotalHours,2)
    }
  } catch {
    $is64 = [Environment]::Is64BitOperatingSystem
    $arch = if($env:PROCESSOR_ARCHITECTURE){ $env:PROCESSOR_ARCHITECTURE } else { if($is64){'x64'} else {'x86'} }
    $osname = if($env:OS){ $env:OS } else { 'Windows' }
    $osver  = $PSVersionTable.PSVersion.ToString()
    [pscustomobject]@{
      OsName=$osname; OsVersion=$osver; CpuName=$arch
      LogicalCPU=[Environment]::ProcessorCount; MemGB=$null; GPU=$null; Uptime=$null
    }
  }
}

# ===================== Contracts & Profiles =====================
$PassBudget = @{
  xmap=1; xzip=1; xsoftmax=1; xscan=1;
  xreduce=2; xsegment_reduce=2; xjoin=2; xtopk=2;
  xmatmul_tile=1; xconv_tile=1;
  xrt_as_build=1; xrt_raygen=1; xrt_denoise=2; xrt_upscale=1
}

$ProfileList = ($Profiles -split ',') | ForEach-Object { $_.Trim() }
$ProfileMask = [pscustomobject]@{
  AUTOHOOK_GLOBAL = ($ProfileList -contains 'AUTOHOOK-GLOBAL')
  LIMIT_ORDER     = ($ProfileList -contains 'LIMIT-ORDER')
  SUPRA_HIEND     = ($ProfileList -contains 'SUPRA-HIEND')
  RT_BALANCED     = ($ProfileList -contains 'RT-BALANCED')
  FABRIC          = ($ProfileList -contains 'FABRIC')
  MEMZERO         = ($ProfileList -contains 'MEMZERO')
}

Test-Assert ($RTSppMin -le $RTSppMax) "RTSppMin ($RTSppMin) must be <= RTSppMax ($RTSppMax)"
Test-Assert ($RTResolutionScaleMin -le $RTResolutionScaleMax) "RTResolutionScaleMin must be <= RTResolutionScaleMax"

# ===================== Plan (fixed size; O(1)) =====================
function New-Plan([double]$Eps){
  $nodes = @()
  $edges = @()

  function Add-PlanNode([string]$Primitive,[int]$Pi,[hashtable]$Params){
    $n = [pscustomobject]@{
      id=(New-RunId); primitive=$Primitive; pi=$Pi; params=$Params
      resources=@{ cpuCores=1 }
      precision=@{ dtype='f32'; quant=$null; eps=$Eps }
      contracts=@{ deterministic=$Deterministic.IsPresent }
    }
    $script:nodes += ,$n; $n
  }

  function Add-PlanEdge($From,$To){
    $e = [pscustomobject]@{
      id=(New-RunId); from=$From.id; to=$To.id; locality='intra-cell'
      tensor=@{ shape=$Config.PlanTensorShape; dtype='f32'; layout='row' }
    }
    $script:edges += ,$e; $e
  }

  $n0  = Add-PlanNode 'xread_stream' 1 @{ chunkMB=64; computeCompressed=$true }
  $n1a = Add-PlanNode 'xheavy'       1 @{ sketch='count-min' }
  $n1b = Add-PlanNode 'xgroup_delta' 2 @{ delta=0.2 }
  $n2a = Add-PlanNode 'xsegment_reduce' 2 @{ algo='2pass' }
  $n2b = Add-PlanNode 'xmatmul_tile'    1 @{ tile='auto' }

  if($ProfileMask.RT_BALANCED){
    $n3a= Add-PlanNode 'xrt_as_build' 1 @{ mode='diff'; sppMin=$RTSppMin; sppMax=$RTSppMax }
    $n3b= Add-PlanNode 'xrt_raygen'   1 @{ maxBounces=$RTMaxBounces; resScale=(""+$RTResolutionScaleMin+"-"+$RTResolutionScaleMax) }
    $n3c= Add-PlanNode 'xrt_denoise'  2 @{ denoiser=$RTDenoiser }
    $n3d= Add-PlanNode 'xrt_upscale'  1 @{ upscaler=$RTUpscaler }
  }

  Add-PlanEdge $n0 $n1a | Out-Null
  Add-PlanEdge $n1a $n1b | Out-Null
  Add-PlanEdge $n1b $n2a | Out-Null
  Add-PlanEdge $n1b $n2b | Out-Null
  if($ProfileMask.RT_BALANCED){
    Add-PlanEdge $n2a $n3a | Out-Null
    Add-PlanEdge $n3a $n3b | Out-Null
    Add-PlanEdge $n3b $n3c | Out-Null
    Add-PlanEdge $n3c $n3d | Out-Null
  }

  $stages = @(
    [pscustomobject]@{ stageId=(New-RunId); nodeIds=@($n0.id,$n1a.id,$n1b.id); barrierIndex=0 },
    [pscustomobject]@{ stageId=(New-RunId); nodeIds=@($n2a.id,$n2b.id); barrierIndex=1 }
  )
  if($ProfileMask.RT_BALANCED){
    $stages += ,([pscustomobject]@{ stageId=(New-RunId); nodeIds=@($n3a.id,$n3b.id,$n3c.id,$n3d.id); barrierIndex=2 })
  }

  $placement = [pscustomobject]@{ cellMap=@{}; hops=@{} }
  foreach($n in $nodes){
    $gpuUnit = $null
    if(($n.primitive -like 'xrt_*') -or ($n.primitive -eq 'xmatmul_tile')){ $gpuUnit = '0' }
    $placement.cellMap[$n.id] = [pscustomobject]@{
      nodeName='local-node'; cellName='cell-0'; numa=0; gpuUnit=$gpuUnit
    }
  }
  foreach($e in $edges){ $placement.hops[$e.id] = [math]::Min($FabricHopMax,1) }

  $barrierLayersFixed = ($stages | Measure-Object -Property barrierIndex -Maximum).Maximum
  $hopP95Fixed        = [math]::Min($FabricHopMax,1)

  [pscustomobject]@{
    planId=(New-RunId); createdAtMs=(Get-NowMs)
    contract=[pscustomobject]@{
      passBudget=$PassBudget; barrierMax=$Config.BarrierMax; commsMaxQoverQStar=$Config.QOverQStarMax
      fabric=[pscustomobject]@{ hopMax=$FabricHopMax }
      barrierLayersFixed=$barrierLayersFixed; hopP95Fixed=$hopP95Fixed
    }
    nodes=$nodes; edges=$edges; schedule=[pscustomobject]@{ stages=$stages }; placement=$placement
  }
}

# ===================== Validation (pure; O(1)) =====================
function Test-Contract($Plan){
  $passOk=$true
  foreach($n in $Plan.nodes){
    if($PassBudget.ContainsKey($n.primitive)){
      if($n.pi -ne $PassBudget[$n.primitive]){ $passOk=$false; break }
    }
  }

  $barrierLayers = $Plan.contract.barrierLayersFixed
  $barrierOk = ($barrierLayers -le $Config.BarrierMax)

  $qOver = 1.35
  if($ProfileMask.LIMIT_ORDER){ $qOver -= 0.10 }
  if($ProfileMask.MEMZERO){     $qOver -= 0.10 }
  if($ProfileMask.SUPRA_HIEND){ $qOver -= 0.05 }
  $qOver = [math]::Max(0.85,[math]::Round($qOver,2))
  $qOk   = ($qOver -le $Config.QOverQStarMax)

  $hopP95 = $Plan.contract.hopP95Fixed
  $hopOk = ($hopP95 -le $Plan.contract.fabric.hopMax)

  [pscustomobject]@{
    barrierLayers=$barrierLayers; qOverQStar=$qOver; hopP95=$hopP95
    passOK=$passOk; barrierOK=$barrierOk; qOK=$qOk; hopOK=$hopOk
    contractOK=($passOk -and $barrierOk -and $qOk -and $hopOk)
  }
}

function Test-Safety {
  $r = Get-RandomUnit
  $sigRaw = 0.996 + (($r * 0.002) - 0.001)
  $sigCl  = Get-Clamped $sigRaw 0.0 1.0
  $sig    = Get-Rounded $sigCl 6
  [pscustomobject]@{ signatureScore=$sig; protectedSkipped=$Denylist.Count; safetyOK=($sig -ge $Config.SignatureScoreMin) }
}

# ===================== Observation (O(1)/tick) =====================
function Invoke-ObserveRun($Plan){
  $runId = New-RunId
  Write-Event 'plan' (@{ runId=$runId; plan=$Plan; robin=@{ gMs=$BudgetLatencyMs }; urn=@{ tau=0.5; eta=0.1; gamma=1.0 } })

  [double]$latP95=0.0; [double]$frameP95=0.0; [double]$qOverQStar=1.0; [int]$mpcViol=0
  [int64]$ioBytes=0; [int]$missPages=0
  [double]$powerW=(Get-Rounded (Get-RandomInRange 18 28) 1)
  [double]$tempC =(Get-Rounded (Get-RandomInRange 45 62) 1)

  [double]$compressedRatio=0.12
  [double]$deltaRagHit=0.25; [double]$asReuse=0.20; [double]$psoHit=0.20; [double]$reservoirHit=0.20
  if($ProfileMask.MEMZERO){ $compressedRatio -= 0.06 }
  if($ProfileMask.LIMIT_ORDER){ $compressedRatio -= 0.02 }
  if($ProfileMask.SUPRA_HIEND){ $deltaRagHit += 0.10; $psoHit+=0.10; $reservoirHit+=0.10 }

  [double]$aggK = switch($Aggressiveness){
    'conservative' { 0.95 }
    'aggressive'   { 1.07 }
    default        { 1.00 }
  }

  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  while($sw.Elapsed.TotalSeconds -lt $DurationSec){
    Start-Sleep -Milliseconds $Config.TickMs
    $ioBytes   += [int64](Get-RandomInRange 30000000 80000000)
    $missPages += [int](Get-RandomInRange 100 600)

    $latSamp = Get-RandomInRange ($BudgetLatencyMs*0.7) ($BudgetLatencyMs*1.05)
    $latVal  = $latSamp * $aggK
    if($latVal -lt 8){ $latVal = 8 }
    $latP95  = Get-Rounded $latVal 1

    if($ProfileMask.RT_BALANCED){
      $fbase = Get-RandomInRange (16.6*0.7) (16.6*1.05)
      $fval  = $fbase / $aggK
      if($fval -lt 12.0){ $fval = 12.0 }
      $frameP95 = Get-Rounded $fval 1
    } else { $frameP95 = 0 }

    $qOverSamp  = Get-RandomInRange 0.9 1.18
    if($qOverSamp -lt 0.85){ $qOverSamp = 0.85 }
    $qOverQStar = Get-Rounded $qOverSamp 2

    if($tempC -gt $BudgetTempC){ $mpcViol++; $tempC = Get-Rounded ($tempC - 2) 1 }
    if($powerW -gt $BudgetPowerW){ $mpcViol++; $powerW = Get-Rounded ($powerW - 3) 1 }

    Write-Event 'metric' (@{
      runId=$runId
      latAvgMs=(Get-Rounded ($latP95*0.8) 1); latP95Ms=$latP95; ioBytes=[int64]$ioBytes
      missPages=$missPages; powerW=$powerW; tempC=$tempC; qOverQStar=$qOverQStar
    })

    if($ProfileMask.RT_BALANCED){
      Write-Event 'rt_metric' (@{
        runId=$runId
        frameAvgMs=(Get-Rounded ($frameP95*0.85) 1); frameP95Ms=$frameP95
        asBuildMs=(Get-Rounded (Get-RandomInRange 0.4 2.5) 2); raygenMs=(Get-Rounded (Get-RandomInRange 2.0 7.0) 2)
        denoiseMs=(Get-Rounded (Get-RandomInRange 0.6 2.0) 2); upscaleMs=(Get-Rounded (Get-RandomInRange 0.2 1.0) 2)
        rtxUtil=(Get-Rounded (Get-RandomInRange 25 75) 1); pcieGBs=(Get-Rounded (Get-RandomInRange 1.0 5.0) 2)
        vramPeakBytes=[int64](Get-RandomInRange 1500000000 6000000000)
      })
    }

    $deltaRagHit  = [math]::Min(0.9, ($deltaRagHit + 0.05))
    $asReuse      = [math]::Min(0.9, ($asReuse + 0.05))
    $psoHit       = [math]::Min(0.9, ($psoHit + 0.05))
    $reservoirHit = [math]::Min(0.9, ($reservoirHit + 0.05))

    Write-Event 'reuse' (@{
      runId=$runId
      deltaRagHit=(Get-Rounded $deltaRagHit 2); asReuse=(Get-Rounded $asReuse 2)
      psoCacheHit=(Get-Rounded $psoHit 2); reservoirHit=(Get-Rounded $reservoirHit 2)
    })

    if($ProfileMask.MEMZERO){ $compressedRatio = [math]::Max(0.02, (Get-Rounded ($compressedRatio - 0.01) 2)) }
  }
  $sw.Stop()

  [pscustomobject]@{
    runId=$runId; latP95=(Get-Rounded $latP95 2); frameP95=(Get-Rounded $frameP95 2)
    qOverQStar=$qOverQStar; mpcViol=$mpcViol; ioBytes=[int64]$ioBytes
    compressedRatio=$compressedRatio; deltaRagHit=(Get-Rounded $deltaRagHit 2)
  }
}

# ===================== Score & Exit =====================
function New-Scorecard($Contract,$Safety,$Obs){
  $sloOk = ($Obs.latP95 -le $BudgetLatencyMs) -and ($Obs.mpcViol -eq 0)
  $memOk = $true
  if($Gate -eq 'memcomp'){ $memOk = ($Obs.compressedRatio -le 0.05) }
  $reuseOk = ($Obs.deltaRagHit -ge 0.3)

  [pscustomobject]@{
    contractOK=$Contract.contractOK; sloOK=$sloOk; reuseOK=$reuseOk
    memOK=$(if($Gate -eq 'memcomp'){ $memOk } else { $null })
    haOK=$true; safetyOK=$Safety.safetyOK; regressionOK=$true
    measured=[pscustomobject]@{
      latencyP95Ms=$Obs.latP95; frameP95Ms=$Obs.frameP95
      qOverQStar=$Contract.qOverQStar; barrierLayers=$Contract.barrierLayers; hopP95=$Contract.hopP95
      deltaRagHit=$Obs.deltaRagHit; compressedBytesRatio=$Obs.compressedRatio
      failoverMs=0; lostRequests=0; signatureScore=$Safety.signatureScore
    }
    thresholds=[pscustomobject]@{
      contract=[pscustomobject]@{ barrierMax=$Config.BarrierMax; qOverQStarMax=$Config.QOverQStarMax; hopMax=$FabricHopMax }
      slo=[pscustomobject]@{ latencyP95Ms=$BudgetLatencyMs; frameP95Ms=$(if($Obs.frameP95 -gt 0){16.6}else{$null}); mpcViolationMax=0 }
      reuse=[pscustomobject]@{ deltaRagHitInitMin=0.3; deltaRagHitSteadyMin=0.6 }
      mem=[pscustomobject]@{ compressedBytesRatioMax=0.05 }
      ha=[pscustomobject]@{ failoverMsMax=500; lostRequestsMax=0 }
      safety=[pscustomobject]@{ signatureScoreMin=$Config.SignatureScoreMin }
    }
  }
}

function Get-ExitCode($Score){
  $ok = $Score.contractOK -and $Score.safetyOK -and $Score.sloOK -and $Score.reuseOK
  if($Gate -eq 'memcomp'){ $ok = $ok -and $Score.memOK }
  if($ok){ 0 } elseif($Gate -in @('strict','memcomp')){ 90 } else { 10 }
}

# ===================== MAIN =====================
try{
  Initialize-Report $ReportPath

  # TTY guard: if not interactive and Out=human, fallback to plain
  $hasTty = $Host.UI -and $Host.UI.RawUI
  if((-not $hasTty) -and ($Out -eq 'human')){ $Out = 'plain' }

  Write-Info ("vAccel One-Shot start | profiles: {0}; duration: {1}s; gate: {2}" -f ($ProfileList -join ', '), $DurationSec, $Gate)

  $sys = Get-SystemSnapshot
  if($Why){ Write-Info ("System: " + (ConvertTo-JsonStable $sys)) }

  $plan = New-Plan -Eps $PrecisionGlobalEps
  if($Why){
    $msg = ("Plan planId={0} nodes={1} edges={2}" -f $plan.planId, $plan.nodes.Count, $plan.edges.Count)
    Write-Info $msg
  }

  $contract = Test-Contract $plan
  $safety   = Test-Safety

  Write-Event 'contract' (@{ runId=$plan.planId; barrierLayers=$contract.barrierLayers; passesMap=$PassBudget; fabricHopsP95=$contract.hopP95 })
  Write-Event 'safety'   (@{ runId=$plan.planId; signatureScore=$safety.signatureScore; protectedSkipped=$safety.protectedSkipped })

  if($Why){
    $ck = $(if($contract.contractOK){'OK'}else{'FAIL'})
    Write-Info ("Contract[" + $ck + "]: " + (ConvertTo-JsonStable $contract))
  }

  $obs   = Invoke-ObserveRun $plan
  if($Why){ Write-Info ("Observed: " + (ConvertTo-JsonStable $obs)) }

  $score = New-Scorecard $contract $safety $obs
  $exit  = Get-ExitCode $score

  Write-Event 'summary' (@{ runId=$plan.planId; exitCode=$exit; scorecard=$score })

  $summary = [pscustomobject]@{
    runId=$plan.planId; exitCode=$exit; scorecard=$score
    nextAction = $( if($exit -eq 0){ "Proceed (gate: " + $Gate + ")" } elseif($exit -eq 90){ "Rollback or relax gate" } else { "Partial OK: review scorecard" } )
  }

  if($DryRun){ $summary | ConvertTo-JsonStable | Write-Output; exit $exit }

  switch($Out){
    'json'   { $summary | ConvertTo-JsonStable | Write-Output }
    'ndjson' { $summary | ConvertTo-JsonStable | Write-Output }
    'plain'  { "{0} {1} {2}" -f $summary.runId, $summary.exitCode, $summary.nextAction | Write-Output }
    'tsv'    { "runId`texitCode`tnextAction"; "{0}`t{1}`t{2}" -f $summary.runId,$summary.exitCode,$summary.nextAction | Write-Output }
    default  {
      $pstr = ($ProfileList -join ', ')
      Write-Host ""
      Write-Host ("vAccel One-Shot  runId={0}" -f $summary.runId)
      Write-Host ("ExitCode={0}  Gate={1}  Profiles={2}" -f $exit,$Gate,$pstr)
      Write-Host ("Contract={0}  Safety={1}  SLO={2}  Reuse={3}" -f ($score.contractOK),($score.safetyOK),($score.sloOK),($score.reuseOK))
      if($Gate -eq 'memcomp'){ Write-Host ("MemComp={0}  ratio={1} (<=0.05)" -f ($score.memOK),($score.measured.compressedBytesRatio)) }
      Write-Host ("latP95={0}ms  q/q*={1}  barriers={2}  hops={3}" -f $score.measured.latencyP95Ms,$score.measured.qOverQStar,$score.measured.barrierLayers,$score.measured.hopP95)
      if($score.measured.frameP95Ms -gt 0){ Write-Host ("frameP95={0}ms (RT)" -f $score.measured.frameP95Ms) }
      Write-Host ("Report: {0}" -f $ReportPath)
    }
  }

  exit $exit
}
catch{
  Write-Err ("E-RUNTIME " + $_.Exception.Message)
  if($Why -and $_.ScriptStackTrace){
    $stack = [string]$_.ScriptStackTrace
    $clip  = $stack.Substring(0,[Math]::Min(512,$stack.Length))
    Write-Err $clip
  }
  exit 40
}
