import shutil
import os

outputs_path = "outputs"
runs_path = "runs"
outputs_name = os.listdir(outputs_path)
for name in os.listdir(runs_path):
    if name not in outputs_name:
        shutil.rmtree(os.path.join(runs_path, name))

