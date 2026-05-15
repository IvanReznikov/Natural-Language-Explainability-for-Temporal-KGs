param(
    [string]$Milestone = "all"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $RootDir "..\output\run_all_examples\$Timestamp"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Run-Example {
    param(
        [string]$Path,
        [string]$Name
    )

    Write-Host "`nRunning: $Name" -ForegroundColor Cyan
    $LogFile = Join-Path $OutputDir "$Name.log"

    try {
        python $Path *>&1 | Tee-Object -FilePath $LogFile
        Write-Host "Success: $Name" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "Failed: $Name" -ForegroundColor Red
        return $false
    }
}

$Examples = @()

if ($Milestone -eq "all" -or $Milestone -eq "1") {
    $Examples += @(
        @{ Path = "examples/milestone1/m1e1_delta_encoding_example.py"; Name = "m1_delta_encoding" },
        @{ Path = "examples/milestone1/m1e1_changelog_example.py"; Name = "m1_changelog" },
        @{ Path = "examples/milestone1/m1e1_graph_indexing_example.py"; Name = "m1_graph_indexing" },
        @{ Path = "examples/milestone1/m1e1_visualization_example.py"; Name = "m1_visualization" }
    )
}

if ($Milestone -eq "all" -or $Milestone -eq "2") {
    $Examples += @(
        @{ Path = "examples/milestone2/m2e2_intent_example.py"; Name = "m2_intent" },
        @{ Path = "examples/milestone2/m2e4_taxonomy_example.py"; Name = "m2_taxonomy" },
        @{ Path = "examples/milestone2/m2e5_trace_meta_query_example.py"; Name = "m2_trace_meta" },
        @{ Path = "examples/milestone2/m2e6_trigger_chain_example.py"; Name = "m2_trigger_chain" },
        @{ Path = "examples/milestone2/m2e7_harness_example.py"; Name = "m2_harness" }
    )
}

if ($Milestone -eq "all" -or $Milestone -eq "3") {
    $Examples += @(
        @{ Path = "examples/milestone3/m3e1_dataset_overview_example.py"; Name = "m3_dataset_overview" },
        @{ Path = "examples/milestone3/m3e2_fidelity_summary_example.py"; Name = "m3_fidelity_summary" },
        @{ Path = "examples/milestone3/m3e3_human_eval_overview_example.py"; Name = "m3_human_eval_overview" },
        @{ Path = "examples/milestone3/m3e4_quality_summary_example.py"; Name = "m3_quality_summary" },
        @{ Path = "examples/milestone3/m3e5_matrix_overview_example.py"; Name = "m3_matrix_overview" },
        @{ Path = "examples/milestone3/m3e5_lcel_query_example.py"; Name = "m3_lcel_query" }
    )
}

$SuccessCount = 0
$TotalCount = $Examples.Count

foreach ($Example in $Examples) {
    if (Run-Example -Path $Example.Path -Name $Example.Name) {
        $SuccessCount++
    }
}

Write-Host "`nSummary: $SuccessCount/$TotalCount examples succeeded" -ForegroundColor Yellow
Write-Host "Logs: $OutputDir" -ForegroundColor Yellow
