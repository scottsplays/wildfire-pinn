# Turn the Vertex AI endpoints on or off (billing runs only while a model is deployed).
#   powershell -ExecutionPolicy Bypass -File cloud.ps1 status
#   powershell -ExecutionPolicy Bypass -File cloud.ps1 off            # PINN only
#   powershell -ExecutionPolicy Bypass -File cloud.ps1 on             # PINN only (~15-20 min)
#   powershell -ExecutionPolicy Bypass -File cloud.ps1 off -All       # PINN + U-Net (ask your teammate first)
param([Parameter(Mandatory = $true)][ValidateSet("on", "off", "status")][string]$Action, [switch]$All)

$Region = "us-central1"
$Project = "wildfirepinn"
$Services = @(
  @{ Name = "PINN v0.3.1"; Endpoint = "2947478962520981504"; Model = "6701330407004045312"; Display = "pinn-v031"; Machine = "n1-standard-2" }
)
if ($All) {
  $Services += @{ Name = "U-Net"; Endpoint = "1266228926628233216"; Model = "735186800644980736"; Display = "wildfire-unet"; Machine = "n1-standard-2" }
}

gcloud config set project $Project 2>$null | Out-Null
foreach ($s in $Services) {
  $ids = @(gcloud ai endpoints describe $s.Endpoint --region=$Region --format="value(deployedModels[].id)" 2>$null) -split "[;,\s]+" | Where-Object { $_ }
  switch ($Action) {
    "status" {
      if ($ids) { Write-Host "$($s.Name): ON  (billing)  deployed model $($ids -join ', ')" -ForegroundColor Yellow }
      else      { Write-Host "$($s.Name): OFF (no charge)" -ForegroundColor Green }
    }
    "off" {
      if (-not $ids) { Write-Host "$($s.Name): already off"; continue }
      foreach ($id in $ids) {
        Write-Host "$($s.Name): turning off ($id) ..."
        gcloud ai endpoints undeploy-model $s.Endpoint --region=$Region --deployed-model-id=$id
      }
    }
    "on" {
      if ($ids) { Write-Host "$($s.Name): already on"; continue }
      Write-Host "$($s.Name): turning on - takes 15-20 minutes, leave this window open ..."
      gcloud ai endpoints deploy-model $s.Endpoint --region=$Region --model=$($s.Model) --display-name=$($s.Display) `
        --machine-type=$($s.Machine) --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100
    }
  }
}
