"""只改变本次报告日志路径，三个恢复用例均执行真实Qwen与工具。"""
from copy import deepcopy
from pathlib import Path
import runpy
import sys
root=Path(sys.argv[1]); output=Path(sys.argv[2])
sys.path.insert(0,str(root))
import src.utils.config as cfg
config=deepcopy(cfg.load_config())
config['paths']['logs']=str(output.parent/'recovery-logs')
cfg.load_config=lambda:deepcopy(config)
script=root/'reports/5_3_3 Agent决策优化/verify_error_recovery.py'
sys.argv=[str(script),'--output',str(output)]
runpy.run_path(str(script),run_name='__main__')
