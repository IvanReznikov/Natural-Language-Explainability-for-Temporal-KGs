#!/usr/bin/env pwsh
Set-StrictMode -Version Latest

Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Pulling prebuilt qwen-llm image and building qwen-embed..."
$env:DOCKER_BUILDKIT = "1"
docker compose pull qwen-llm
docker compose --progress plain build qwen-embed

Write-Host "Starting Qwen LLM and Embedding containers (detached)..."
docker compose up -d qwen-llm qwen-embed

Write-Host "Done. LLM -> http://localhost:8000/v1  Embeddings -> http://localhost:8001/v1"
Write-Host "Use: docker compose ps ; docker compose logs -f qwen-llm"
