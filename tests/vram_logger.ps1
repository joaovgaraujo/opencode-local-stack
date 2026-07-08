param([string]$Out = "D:\Downloads\qwen36_test\llm\tests\vram.log", [int]$Seconds = 600)
"ts_s,card_used_mib,llama_ws_mib" | Set-Content $Out
$i = 0
while ($i -lt $Seconds) {
    $card = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join ""
    $p = Get-Process llama-server -ErrorAction SilentlyContinue
    $ws = if ($p) { [math]::Round($p.WorkingSet64/1MB) } else { 0 }
    "$i,$card,$ws" | Add-Content $Out
    Start-Sleep -Seconds 1
    $i++
}
