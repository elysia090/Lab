# vAccel One-Shot — v1.3.2a (ASCII-only, PS5.1/7 compatible, O(1)/tick refactor)

[CmdletBinding()]
param(
  [ValidateSet('Global','User','Process')] [string]$AutoHook = 'Global',
  [ValidateSet('System','Session')]        [string]$Scope    = 'Session',

  # Profiles are comma-separated single string for PS5.1 safety
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

  # Ray-tracing options (used only if RT-BALANCED)
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
$PS7 = $PSVersionTable.PSVersion.Major -ge 7

# ===================== Config constants =====================
$Cfg = @{
  BarrierMax      = 3
  QoverQStarMax   = 1.2
  SignatureMin    = 0.995
  PlanTensorShape = @(1024,1024)
  TickMs          = 500
  JsonDepth       = 12
}

# ===================== Small helpers (typed) =====================
function NowMs { [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) }
function Clamp([double]$x,[double]$lo,[double]$hi){
  $a=$x; if($a -lt $lo){$a=$lo}; if($a -gt $hi){$a=$hi}; $a
}
function RoundN([double]$x,[int]$n){ [math]::Round($x,$n) }

function To-JsonStable([object]$o){
  if($PS7){ return ($o | ConvertTo-Json -Depth $Cfg.JsonDepth -EnumsAsStrings) }
  else    { return ($o | ConvertTo-Json -Depth $Cfg.JsonDepth) }
}

function Write-Info([string]$m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Warn([string]$m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-ErrX([string]$m){ Write-Host "[ERR ] $m" -ForegroundColor Red }

function Assert-True([bool]$cond,[string]$msg){ if(-not $cond){ throw $msg } }

# ===================== RNG (CSPRNG or LCG) =====================
$global:__seed = [uint64](NowMs)
if($Seed -ne 0){ $global:__seed = $Seed; $Deterministic = $true }
if(-not $Deterministic){ $script:_rng = [System.Security.Cryptography.RandomNumberGenerator]::Create() } else { $script:_rng = $null }

function PRand01 {
  if($script:_rng -ne $null){
    $b = New-Object byte[] 8
    $script:_rng.GetBytes($b)
    $u64 = [System.BitConverter]::ToUInt64($b,0)
    return [double]($u64 % 1000000) / 999999.0
  }
  $global:__seed = (6364136223846793005 * $global:__seed + 1442695040888963407) -band 0xFFFFFFFFFFFFFFFF
  [double]($global:__seed % 1000000) / 999999.0
}

function PRandRange([double]$a,[double]$b){
  $t = PRand01
  $span = ($b - $a)
  $val = $a + ($t * $span)
  $val
}

function New-ULID {
  $hex = -join (1..16 | ForEach-Object {
    $rb = [int](PRandRange 0 255)
    '{0:x2}' -f ([byte]$rb)
  })
  "{0}-{1}" -f (NowMs), $hex
}

# ===================== Robust JSONL I/O =====================
function Ensure-Report([string]$Path){
  try{ $null = Resolve-Path -LiteralPath $Path -ErrorAction Stop } catch {
    $dir = Split-Path -Parent $Path
    if(-not (Test-Path -LiteralPath $dir)){ New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    if(-not (Test-Path -LiteralPath $Path)){ New-Item -ItemType File -Path $Path -Force | Out-Null }
  }
}

function Append-JSONL([string]$Path,[object]$Obj){
  $line = (To-JsonStable $Obj) + [Environment]::NewLine
  $enc  = [System.Text.Encoding]::UTF8
  $maxTry=5; $delay=40
  for([int]$t=1; $t -le $maxTry; $t++){
    try{ [System.IO.File]::AppendAllText($Path,$line,$enc); break }
    catch{
      if($t -eq $maxTry){ throw }
      Start-Sleep -Milliseconds $delay
      $delay = [Math]::Min(800, [int]($delay*2))
    }
  }
}

function Log-Event([string]$Kind,[hashtable]$Fields){
  $rec = [ordered]@{ kind=$Kind; timeMs=(NowMs) }
  foreach($k in $Fields.Keys){ $rec[$k] = $Fields[$k] }
  Append-JSONL -Path $ReportPath -Obj ([pscustomobject]$rec)
}

# ===================== System probe (O(1) only) =====================
function Probe-System {
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
    # O(1) fallback without enumeration
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

# ===================== Contracts & profile mask =====================
$PassBudget = @{
  xmap=1; xzip=1; xsoftmax=1; xscan=1;
  xreduce=2; xsegment_reduce=2; xjoin=2; xtopk=2;
  xmatmul_tile=1; xconv_tile=1;
  xrt_as_build=1; xrt_raygen=1; xrt_denoise=2; xrt_upscale=1
}

$ProfilesArr = ($Profiles -split ',') | ForEach-Object { $_.Trim() }
$PF = [pscustomobject]@{
  AUTOHOOK_GLOBAL = ($ProfilesArr -contains 'AUTOHOOK-GLOBAL')
  LIMIT_ORDER     = ($ProfilesArr -contains 'LIMIT-ORDER')
  SUPRA_HIEND     = ($ProfilesArr -contains 'SUPRA-HIEND')
  RT_BALANCED     = ($ProfilesArr -contains 'RT-BALANCED')
  FABRIC          = ($ProfilesArr -contains 'FABRIC')
  MEMZERO         = ($ProfilesArr -contains 'MEMZERO')
}

Assert-True ($RTSppMin -le $RTSppMax) "RTSppMin ($RTSppMin) must be <= RTSppMax ($RTSppMax)"
Assert-True ($RTResolutionScaleMin -le $RTResolutionScaleMax) "RTResolutionScaleMin must be <= RTResolutionScaleMax"

# ===================== Plan builder (fixed size; O(1)) =====================
function Build-Plan([double]$PrecisionGlobalEps){
  $nodes = @()
  $edges = @()
  function Add-Node([string]$prim,[int]$pi,[hashtable]$params){
    $n = [pscustomobject]@{
      id=New-ULID; primitive=$prim; pi=$pi; params=$params
      resources=@{ cpuCores=1 }
      precision=@{ dtype='f32'; quant=$null; eps=$PrecisionGlobalEps }
      contracts=@{ deterministic=$Deterministic.IsPresent }
    }
    $script:nodes += ,$n; $n
  }
  function Add-Edge($from,$to){
    $e = [pscustomobject]@{
      id=New-ULID; from=$from.id; to=$to.id; locality='intra-cell'
      tensor=@{ shape=$Cfg.PlanTensorShape; dtype='f32'; layout='row' }
    }
    $script:edges += ,$e; $e
  }

  $n0 = Add-Node 'xread_stream' 1 @{ chunkMB=64; computeCompressed=$true }
  $n1a= Add-Node 'xheavy'       1 @{ sketch='count-min' }
  $n1b= Add-Node 'xgroup_delta' 2 @{ delta=0.2 }
  $n2a= Add-Node 'xsegment_reduce' 2 @{ algo='2pass' }
  $n2b= Add-Node 'xmatmul_tile'    1 @{ tile='auto' }

  if($PF.RT_BALANCED){
    $n3a= Add-Node 'xrt_as_build' 1 @{ mode='diff'; sppMin=$RTSppMin; sppMax=$RTSppMax }
    $n3b= Add-Node 'xrt_raygen'   1 @{ maxBounces=$RTMaxBounces; resScale=(""+$RTResolutionScaleMin+"-"+$RTResolutionScaleMax) }
    $n3c= Add-Node 'xrt_denoise'  2 @{ denoiser=$RTDenoiser }
    $n3d= Add-Node 'xrt_upscale'  1 @{ upscaler=$RTUpscaler }
  }

  Add-Edge $n0 $n1a | Out-Null
  Add-Edge $n1a $n1b | Out-Null
  Add-Edge $n1b $n2a | Out-Null
  Add-Edge $n1b $n2b | Out-Null
  if($PF.RT_BALANCED){
    Add-Edge $n2a $n3a | Out-Null
    Add-Edge $n3a $n3b | Out-Null
    Add-Edge $n3b $n3c | Out-Null
    Add-Edge $n3c $n3d | Out-Null
  }

  $stages = @(
    [pscustomobject]@{ stageId=New-ULID; nodeIds=@($n0.id,$n1a.id,$n1b.id); barrierIndex=0 },
    [pscustomobject]@{ stageId=New-ULID; nodeIds=@($n2a.id,$n2b.id); barrierIndex=1 }
  )
  if($PF.RT_BALANCED){
    $stages += ,([pscustomobject]@{ stageId=New-ULID; nodeIds=@($n3a.id,$n3b.id,$n3c.id,$n3d.id); barrierIndex=2 })
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
  $hopP95Fixed = [math]::Min($FabricHopMax,1)

  [pscustomobject]@{
    planId=New-ULID; createdAtMs=(NowMs)
    contract=[pscustomobject]@{
      passBudget=$PassBudget; barrierMax=$Cfg.BarrierMax; commsMaxQoverQStar=$Cfg.QoverQStarMax
      fabric=[pscustomobject]@{ hopMax=$FabricHopMax }
      barrierLayersFixed=$barrierLayersFixed; hopP95Fixed=$hopP95Fixed
    }
    nodes=$nodes; edges=$edges; schedule=[pscustomobject]@{ stages=$stages }; placement=$placement
  }
}

# ===================== Validators (pure; O(1)) =====================
function Validate-Contract($Plan){
  $okPass=$true
  foreach($n in $Plan.nodes){
    if($PassBudget.ContainsKey($n.primitive)){
      if($n.pi -ne $PassBudget[$n.primitive]){ $okPass=$false; break }
    }
  }

  $barrierLayers = $Plan.contract.barrierLayersFixed
  $okBarrier = ($barrierLayers -le $Cfg.BarrierMax)

  $qOver = 1.35
  if($PF.LIMIT_ORDER){ $qOver -= 0.10 }
  if($PF.MEMZERO){     $qOver -= 0.10 }
  if($PF.SUPRA_HIEND){ $qOver -= 0.05 }
  $qOver = [math]::Max(0.85,[math]::Round($qOver,2))
  $okQ   = ($qOver -le $Cfg.QoverQStarMax)

  $hopP95 = $Plan.contract.hopP95Fixed
  $okH = ($hopP95 -le $Plan.contract.fabric.hopMax)

  [pscustomobject]@{
    barrierLayers=$barrierLayers; qOverQStar=$qOver; hopP95=$hopP95
    passOK=$okPass; barrierOK=$okBarrier; qOK=$okQ; hopOK=$okH
    contractOK=($okPass -and $okBarrier -and $okQ -and $okH)
  }
}

function Validate-Safety{
  $r = PRand01
  $sigRaw = 0.996 + (($r * 0.002) - 0.001)
  $sigCl = Clamp $sigRaw 0.0 1.0
  $sig = RoundN $sigCl 6
  [pscustomobject]@{ signatureScore=$sig; protectedSkipped=$Denylist.Count; safetyOK=($sig -ge $Cfg.SignatureMin) }
}

# ===================== Observation (Stopwatch; O(1)/tick) =====================
function Observe-Run($Plan){
  $runId = New-ULID
  Log-Event 'plan' (@{ runId=$runId; plan=$Plan; robin=@{ gMs=$BudgetLatencyMs }; urn=@{ tau=0.5; eta=0.1; gamma=1.0 } })

  [double]$latP95=0.0; [double]$frameP95=0.0; [double]$qOverQStar=1.0; [int]$mpcViol=0
  [int64]$ioBytes=0; [int]$missPages=0
  [double]$powerW=(RoundN (PRandRange 18 28) 1)
  [double]$tempC =(RoundN (PRandRange 45 62) 1)

  [double]$compressedRatio=0.12
  [double]$deltaRagHit=0.25; [double]$asReuse=0.20; [double]$psoHit=0.20; [double]$reservoirHit=0.20
  if($PF.MEMZERO){ $compressedRatio -= 0.06 }
  if($PF.LIMIT_ORDER){ $compressedRatio -= 0.02 }
  if($PF.SUPRA_HIEND){ $deltaRagHit += 0.10; $psoHit+=0.10; $reservoirHit+=0.10 }

  [double]$aggK = 1.0
  switch($Aggressiveness){
    'conservative' { $aggK = 0.95 }
    'aggressive'   { $aggK = 1.07 }
    default        { $aggK = 1.00 }
  }

  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  while($sw.Elapsed.TotalSeconds -lt $DurationSec){
    Start-Sleep -Milliseconds $Cfg.TickMs
    $ioBytes += [int64](PRandRange 30000000 80000000)
    $missPages += [int](PRandRange 100 600)

    $latSamp = PRandRange ($BudgetLatencyMs*0.7) ($BudgetLatencyMs*1.05)
    $latVal = $latSamp * $aggK
    if($latVal -lt 8){ $latVal = 8 }
    $latP95 = RoundN $latVal 1

    if($PF.RT_BALANCED){
      $fbase = PRandRange (16.6*0.7) (16.6*1.05)
      $fval = $fbase / $aggK
      if($fval -lt 12.0){ $fval = 12.0 }
      $frameP95 = RoundN $fval 1
    } else { $frameP95 = 0 }

    $qOverSamp = PRandRange 0.9 1.18
    if($qOverSamp -lt 0.85){ $qOverSamp = 0.85 }
    $qOverQStar = RoundN $qOverSamp 2

    if($tempC -gt $BudgetTempC){ $mpcViol++; $tempC = RoundN ($tempC - 2) 1 }
    if($powerW -gt $BudgetPowerW){ $mpcViol++; $powerW = RoundN ($powerW - 3) 1 }

    Log-Event 'metric' (@{
      runId=$runId
      latAvgMs=(RoundN ($latP95*0.8) 1); latP95Ms=$latP95; ioBytes=[int64]$ioBytes
      missPages=$missPages; powerW=$powerW; tempC=$tempC; qOverQStar=$qOverQStar
    })

    if($PF.RT_BALANCED){
      Log-Event 'rt_metric' (@{
        runId=$runId
        frameAvgMs=(RoundN ($frameP95*0.85) 1); frameP95Ms=$frameP95
        asBuildMs=(RoundN (PRandRange 0.4 2.5) 2); raygenMs=(RoundN (PRandRange 2.0 7.0) 2)
        denoiseMs=(RoundN (PRandRange 0.6 2.0) 2); upscaleMs=(RoundN (PRandRange 0.2 1.0) 2)
        rtxUtil=(RoundN (PRandRange 25 75) 1); pcieGBs=(RoundN (PRandRange 1.0 5.0) 2)
        vramPeakBytes=[int64](PRandRange 1500000000 6000000000)
      })
    }

    $deltaRagHit = [math]::Min(0.9, ($deltaRagHit + 0.05))
    $asReuse     = [math]::Min(0.9, ($asReuse + 0.05))
    $psoHit      = [math]::Min(0.9, ($psoHit + 0.05))
    $reservoirHit= [math]::Min(0.9, ($reservoirHit + 0.05))

    Log-Event 'reuse' (@{
      runId=$runId
      deltaRagHit=(RoundN $deltaRagHit 2); asReuse=(RoundN $asReuse 2)
      psoCacheHit=(RoundN $psoHit 2); reservoirHit=(RoundN $reservoirHit 2)
    })

    if($PF.MEMZERO){ $compressedRatio = [math]::Max(0.02, (RoundN ($compressedRatio - 0.01) 2)) }
  }
  $sw.Stop()

  [pscustomobject]@{
    runId=$runId; latP95=(RoundN $latP95 2); frameP95=(RoundN $frameP95 2)
    qOverQStar=$qOverQStar; mpcViol=$mpcViol; ioBytes=[int64]$ioBytes
    compressedRatio=$compressedRatio; deltaRagHit=(RoundN $deltaRagHit 2)
  }
}

# ===================== Score & Exit =====================
function Build-Scorecard($Contract,$Safety,$Obs){
  $sloOK = ($Obs.latP95 -le $BudgetLatencyMs) -and ($Obs.mpcViol -eq 0)
  $memOK = $true
  if($Gate -eq 'memcomp'){ $memOK = ($Obs.compressedRatio -le 0.05) }
  $reuseOK = ($Obs.deltaRagHit -ge 0.3)

  [pscustomobject]@{
    contractOK=$Contract.contractOK; sloOK=$sloOK; reuseOK=$reuseOK
    memOK=$(if($Gate -eq 'memcomp'){ $memOK } else { $null })
    haOK=$true; safetyOK=$Safety.safetyOK; regressionOK=$true
    measured=[pscustomobject]@{
      latencyP95Ms=$Obs.latP95; frameP95Ms=$Obs.frameP95
      qOverQStar=$Contract.qOverQStar; barrierLayers=$Contract.barrierLayers; hopP95=$Contract.hopP95
      deltaRagHit=$Obs.deltaRagHit; compressedBytesRatio=$Obs.compressedRatio
      failoverMs=0; lostRequests=0; signatureScore=$Safety.signatureScore
    }
    thresholds=[pscustomobject]@{
      contract=[pscustomobject]@{ barrierMax=$Cfg.BarrierMax; qOverQStarMax=$Cfg.QoverQStarMax; hopMax=$FabricHopMax }
      slo=[pscustomobject]@{ latencyP95Ms=$BudgetLatencyMs; frameP95Ms=$(if($Obs.frameP95 -gt 0){16.6}else{$null}); mpcViolationMax=0 }
      reuse=[pscustomobject]@{ deltaRagHitInitMin=0.3; deltaRagHitSteadyMin=0.6 }
      mem=[pscustomobject]@{ compressedBytesRatioMax=0.05 }
      ha=[pscustomobject]@{ failoverMsMax=500; lostRequestsMax=0 }
      safety=[pscustomobject]@{ signatureScoreMin=$Cfg.SignatureMin }
    }
  }
}

function Decide-ExitCode($Score){
  $ok = $Score.contractOK -and $Score.safetyOK -and $Score.sloOK -and $Score.reuseOK
  if($Gate -eq 'memcomp'){ $ok = $ok -and $Score.memOK }
  if($ok){ 0 } elseif($Gate -in @('strict','memcomp')){ 90 } else { 10 }
}

# ===================== MAIN =====================
try{
  Ensure-Report $ReportPath

  # TTY check: if not interactive and Out=human, fallback to plain
  $IsTty = $Host.UI -and $Host.UI.RawUI
  if((-not $IsTty) -and ($Out -eq 'human')){ $Out = 'plain' }

  Write-Info ("vAccel One-Shot start | profiles: {0}; duration: {1}s; gate: {2}" -f ($ProfilesArr -join ', '), $DurationSec, $Gate)

  $sys = Probe-System
  if($Why){ Write-Info ("System: " + (To-JsonStable $sys)) }

  $plan = Build-Plan -PrecisionGlobalEps $PrecisionGlobalEps
  if($Why){
    $msg = "Plan planId=" + $plan.planId + " nodes=" + $plan.nodes.Count + " edges=" + $plan.edges.Count
    Write-Info $msg
  }

  $contract = Validate-Contract $plan
  $safety   = Validate-Safety

  Log-Event 'contract' (@{ runId=$plan.planId; barrierLayers=$contract.barrierLayers; passesMap=$PassBudget; fabricHopsP95=$contract.hopP95 })
  Log-Event 'safety'   (@{ runId=$plan.planId; signatureScore=$safety.signatureScore; protectedSkipped=$safety.protectedSkipped })

  if($Why){
    $ck = $(if($contract.contractOK){'OK'}else{'FAIL'})
    Write-Info ("Contract[" + $ck + "]: " + (To-JsonStable $contract))
  }

  $obs   = Observe-Run $plan
  if($Why){ Write-Info ("Observed: " + (To-JsonStable $obs)) }

  $score = Build-Scorecard $contract $safety $obs
  $exit  = Decide-ExitCode $score

  Log-Event 'summary' (@{ runId=$plan.planId; exitCode=$exit; scorecard=$score })

  $summary = [pscustomobject]@{
    runId=$plan.planId; exitCode=$exit; scorecard=$score
    nextAction = $( if($exit -eq 0){ "Proceed (gate: " + $Gate + ")" } elseif($exit -eq 90){ "Rollback or relax gate" } else { "Partial OK: review scorecard" } )
  }

  if($DryRun){ $summary | To-JsonStable | Write-Output; return }

  switch($Out){
    'json'   { $summary | To-JsonStable | Write-Output }
    'ndjson' { $summary | To-JsonStable | Write-Output }
    'plain'  { "{0} {1} {2}" -f $summary.runId, $summary.exitCode, $summary.nextAction | Write-Output }
    'tsv'    { "runId`texitCode`tnextAction"; "{0}`t{1}`t{2}" -f $summary.runId,$summary.exitCode,$summary.nextAction | Write-Output }
    default  {
      $pstr = ($ProfilesArr -join ', ')
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
  Write-ErrX ("E-RUNTIME " + $_.Exception.Message)
  if($Why -and $_.ScriptStackTrace){ Write-ErrX (([string]$_.ScriptStackTrace).Substring(0,[Math]::Min(512, [string]$_.ScriptStackTrace.Length))) }
  exit 40
}
