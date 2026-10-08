$ErrorActionPreference='Stop'
Set-Location 'E:\工作\RAG-Agent'
$r='E:\工作\RAG-Agent\.test-tmp\win-audit-deploy-20261008'
$before=(Get-NetTCPConnection -LocalPort 8501 -State Listen | Select-Object -First 1).OwningProcess
$process=Get-CimInstance Win32_Process -Filter "ProcessId=$before"
if($process.CommandLine -notmatch 'streamlit run src/frontend/app.py'){throw '8501进程不属于本轮Streamlit，停止重启'}
Stop-ScheduledTask -TaskName 'RAG-Audit-Streamlit-20261008'
if(Get-Process -Id $before -ErrorAction SilentlyContinue){Stop-Process -Id $before -Force}
$action=New-ScheduledTaskAction -Execute 'E:\工作\RAG-Agent\.venv\Scripts\python.exe' -Argument '"E:\工作\RAG-Agent\.test-tmp\win-audit-deploy-20261008\rag_start_streamlit.py" native-restart' -WorkingDirectory 'E:\工作\RAG-Agent'
Set-ScheduledTask -TaskName 'RAG-Audit-Streamlit-20261008' -Action $action | Out-Null
Start-ScheduledTask -TaskName 'RAG-Audit-Streamlit-20261008'
$health=$null
for($try=0;$try -lt 15;$try++){
    try{$health=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2; break}catch{Start-Sleep -Seconds 1}
}
$after=(Get-NetTCPConnection -LocalPort 8501 -State Listen | Select-Object -First 1).OwningProcess
$result=[ordered]@{before_pid=$before;after_pid=$after;new_process=($before -ne $after);health_status=$health.StatusCode}
$result | ConvertTo-Json | Set-Content "$r\native-restart.json" -Encoding UTF8
$result | ConvertTo-Json
if(!$result.new_process -or $health.StatusCode -ne 200){exit 1}
