import subprocess, sys, os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["PYTHONUNBUFFERED"]="1"
proc = subprocess.run([sys.executable, "Source/validate_nested_semantic_calibration.py", "--quick"], capture_output=True, text=True, timeout=600)
open("Source/experiments/semantic_calibration_phase314/run_quick_stdout.txt","w",encoding="utf-8").write(proc.stdout or "")
open("Source/experiments/semantic_calibration_phase314/run_quick_stderr.txt","w",encoding="utf-8").write(proc.stderr or "")
open("Source/experiments/semantic_calibration_phase314/run_quick_exit.txt","w",encoding="utf-8").write(str(proc.returncode))
print("stdout:", proc.stdout[:5000] if proc.stdout else "<empty>")
print("stderr:", proc.stderr[:5000] if proc.stderr else "<empty>")
print("exit:", proc.returncode)