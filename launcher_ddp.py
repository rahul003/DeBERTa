import os
#print(os.listdir('/opt/ml/input/data/train/'))
#os.putenv('PYTHONPATH', '/fsx/huilgolr/DeBERTa/')
os.system("./experiments/language_model/mlm.sh deberta-base")
