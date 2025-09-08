import subprocess
import time
from pathlib import Path

script = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/V1/modular/get_data_V1.py")
log_dir = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/log")
executer = Path(r"/home/keivan/pythons/advanced_python/bin/python3")
log_dir.mkdir(parents=True, exist_ok=True)

i_values = [16, 1, 5, 11, 7, 15, 14, 12, 9, 23, 4, 20, 18, 10, 8, 3, 2, 19, 6, 22, 21]
max_concurrent = 5
active_processes = []

for val in i_values:
    # Wait for a free slot
    while len(active_processes) >= max_concurrent:
        # Remove finished processes
        active_processes = [p for p in active_processes if p.poll() is None]
        time.sleep(1)
    print(f"Starting process for i={val}")
    # Start child process with unbuffered stdout (-u)
    p = subprocess.Popen([str(executer), "-u", str(script), str(val)],)
    active_processes.append(p)
    time.sleep(30)  # stagger start

# Wait for all processes to finish
for p in active_processes:
    p.wait()
