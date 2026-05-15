#!/usr/bin/env pwsh

param(
    [string]$InputJsonl      = "data/jsonls/temporal_graph.jsonl",
    [string]$ProcessedDir    = "data/jsonls/processed",
    [string]$GraphOutputDir  = "data/jsonls/temporal_graph_output",
    [string]$GraphToolsDir   = "graph_tools",
    [string]$EmbedUrl        = "http://127.0.0.1:8001",
    [string]$EmbedModelName  = "Qwen/Qwen3-Embedding-0.6B",
    [int]   $BatchSize       = 64,
    [int]   $Parallelism     = 2,
    [switch]$SkipProcess,
    [switch]$SkipBuild,
    [switch]$SkipEmbed,
    [switch]$CoreOnly,
    [int]   $AppendFromLine  = 0,
    [switch]$AutoAppendFromOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

function Invoke-Checked([string[]]$cmd) {
    Write-Host ("> " + ($cmd -join " ")) -ForegroundColor DarkGray
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($cmd -join ' ')"
    }
}

function Get-LastJsonlObject([string]$path) {
    if (-not (Test-Path $path)) {
        return $null
    }
    $last = $null
    Get-Content -Path $path | ForEach-Object {
        $text = ($_ | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            return
        }
        try {
            $last = $text | ConvertFrom-Json
        } catch {
        }
    }
    return $last
}

function Resolve-AutoAppendFromLine([string]$inputJsonlPath, [string]$graphOutputDirPath) {
    $processedGraphPath = Join-Path $graphOutputDirPath "processed_graph.jsonl"
    if (-not (Test-Path $processedGraphPath)) {
        Write-Host "[AutoAppendFromOutput] processed_graph.jsonl not found; defaulting append start line to 1." -ForegroundColor Yellow
        return 1
    }

    $lastObj = Get-LastJsonlObject $processedGraphPath
    if ($null -eq $lastObj) {
        Write-Host "[AutoAppendFromOutput] processed_graph.jsonl is empty; defaulting append start line to 1." -ForegroundColor Yellow
        return 1
    }

    $lastSourceId = "$($lastObj.source_id)"
    if ([string]::IsNullOrWhiteSpace($lastSourceId)) {
        Write-Host "[AutoAppendFromOutput] last source_id missing; defaulting append start line to 1." -ForegroundColor Yellow
        return 1
    }

    $lineNo = 0
    $matchLine = 0
    Get-Content -Path $inputJsonlPath | ForEach-Object {
        $lineNo += 1
        $line = $_
        if ($line -match '"id"\s*:\s*"([^"]+)"') {
            $idVal = $Matches[1]
            if ($idVal -eq $lastSourceId) {
                $matchLine = $lineNo
            }
        }
    }

    if ($matchLine -le 0) {
        throw "[AutoAppendFromOutput] Could not find last source_id '$lastSourceId' in input JSONL '$inputJsonlPath'."
    }

    return ($matchLine + 1)
}

$StartTime = Get-Date
$Root = $PSScriptRoot | Split-Path -Parent

Push-Location $Root
try {
    if (-not (Test-Path $InputJsonl)) {
        throw "Input JSONL not found: $InputJsonl"
    }

    $EffectiveAppendFromLine = $AppendFromLine
    if ($AppendFromLine -gt 0 -and $AutoAppendFromOutput) {
        Write-Host "[AutoAppendFromOutput] Ignored because explicit -AppendFromLine was provided." -ForegroundColor Yellow
    } elseif ($AutoAppendFromOutput) {
        $EffectiveAppendFromLine = Resolve-AutoAppendFromLine -inputJsonlPath $InputJsonl -graphOutputDirPath $GraphOutputDir
        Write-Host "[AutoAppendFromOutput] Resolved append start line: $EffectiveAppendFromLine" -ForegroundColor Cyan
    }

    if (-not $SkipProcess) {
        if ($EffectiveAppendFromLine -gt 0) {
            Write-Host "[AppendFromLine] Stage 1 skipped for incremental tail processing." -ForegroundColor Yellow
            $FilteredJsonl = $InputJsonl
        } else {
            Write-Step "Stage 1 / 3  - QA-check and error-filter temporal_graph.jsonl"

            $null = New-Item -ItemType Directory -Force -Path $ProcessedDir
            $FilteredJsonl = "$ProcessedDir/temporal_graph_error_filtered.jsonl"

            $processArgs = @(
                "python",
                (Join-Path $GraphToolsDir "process_temporal_graph.py"),
                "--input", $InputJsonl,
                "--output-dir", $ProcessedDir,
                "--remove-error-lines",
                "--error-filtered-output", $FilteredJsonl
            )
            Invoke-Checked $processArgs
            Write-Host "Filtered JSONL: $FilteredJsonl" -ForegroundColor Green
        }
    } else {
        Write-Host "[SkipProcess] Stage 1 skipped." -ForegroundColor Yellow
        $FilteredJsonl = "$ProcessedDir/temporal_graph_error_filtered.jsonl"
    }

    if (-not $SkipBuild) {
        Write-Step "Stage 2 / 3  - Build nodes / edges / tags artifacts"

        if (-not (Test-Path $FilteredJsonl)) {
            Write-Host "[WARN] Filtered JSONL not found at '$FilteredJsonl'; falling back to '$InputJsonl'." -ForegroundColor Yellow
            $FilteredJsonl = $InputJsonl
        }

        $null = New-Item -ItemType Directory -Force -Path $GraphOutputDir

        if ($EffectiveAppendFromLine -gt 0) {
            $buildArgs = @(
                "python",
                (Join-Path $GraphToolsDir "append_temporal_graph_output.py"),
                "--input", $InputJsonl,
                "--output-dir", $GraphOutputDir,
                "--start-line", "$EffectiveAppendFromLine"
            )
        } else {
            $buildArgs = @(
                "python",
                (Join-Path $GraphToolsDir "build_temporal_graph_output.py"),
                "--input", $FilteredJsonl,
                "--output-dir", $GraphOutputDir
            )
        }

        Invoke-Checked $buildArgs
        Write-Host "Graph artifacts written to: $GraphOutputDir" -ForegroundColor Green
    } else {
        Write-Host "[SkipBuild] Stage 2 skipped." -ForegroundColor Yellow
    }

    if (-not $SkipEmbed) {
        Write-Step "Stage 3 / 3  - Recompute embedding indexes"

        Write-Host "Probing embed server at $EmbedUrl ..." -ForegroundColor DarkGray
        try {
            $probeBody = @{ model = $EmbedModelName; input = @("probe") } | ConvertTo-Json
            $null = Invoke-RestMethod -Uri "$EmbedUrl/v1/embeddings" -Method Post -ContentType "application/json" -Body $probeBody -TimeoutSec 10
            Write-Host "Embed server OK (v1/embeddings)." -ForegroundColor Green
        } catch {
            try {
                $probeBody = @{ texts = @("probe"); is_query = $false } | ConvertTo-Json
                $null = Invoke-RestMethod -Uri "$EmbedUrl/embed" -Method Post -ContentType "application/json" -Body $probeBody -TimeoutSec 10
                Write-Host "Embed server OK (/embed)." -ForegroundColor Green
            } catch {
                throw "Embed server at '$EmbedUrl' is unreachable. Start it first or use -SkipEmbed."
            }
        }

        $EmbedDir = "$GraphOutputDir/embeddings"
        if ($EffectiveAppendFromLine -le 0 -and (Test-Path $EmbedDir)) {
            Write-Host "Removing stale .npy / .uids.json / .meta.jsonl files from $EmbedDir ..." -ForegroundColor DarkGray
            Get-ChildItem -Path $EmbedDir -Filter "*.npy" | Remove-Item -Force
            Get-ChildItem -Path $EmbedDir -Filter "*.uids.json" | Remove-Item -Force
            Get-ChildItem -Path $EmbedDir -Filter "*.meta.jsonl" | Remove-Item -Force
            Write-Host "Stale caches removed." -ForegroundColor Green
        }

        $embedArgs = @(
            "python",
            "scripts/precompute_graph_embeddings.py",
            "--graph-output-dir", $GraphOutputDir,
            "--qwen-server-url", $EmbedUrl,
            "--server-embedding-model-name", $EmbedModelName,
            "--batch-size", "$BatchSize",
            "--parallelism", "$Parallelism"
        )
        if ($CoreOnly) {
            $embedArgs += "--core-only"
            Write-Host "[CoreOnly] Skipping per-edge retrieval embeddings (faster)." -ForegroundColor Yellow
        }
        if ($EffectiveAppendFromLine -gt 0) {
            $embedArgs += "--append-only"
            Write-Host "[AppendFromLine] Embedding append mode enabled." -ForegroundColor Yellow
        }

        Invoke-Checked $embedArgs
        Write-Host "Embeddings written to: $EmbedDir" -ForegroundColor Green
    } else {
        Write-Host "[SkipEmbed] Stage 3 skipped." -ForegroundColor Yellow
    }

    $Elapsed = (Get-Date) - $StartTime
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Green
    Write-Host ("  Graph rebuild complete in " + [math]::Round($Elapsed.TotalSeconds, 1) + "s") -ForegroundColor Green
    Write-Host ("=" * 72) -ForegroundColor Green

} finally {
    Pop-Location
}
