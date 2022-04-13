import os
import subprocess
import sys

result = subprocess.run(["./experiments/language_model/mlm.sh", "deberta-base"], check=True)    
