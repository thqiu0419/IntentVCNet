import numpy as np
import shutil
import os
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    source_path = "/data/mm2025/IntentVC/"
    for bbox in tqdm(glob("/data/mm2025/IntentVC/*/*/*.npy")):
        # print(bbox)
        target_path = bbox.replace('/data', '/data3')
        target_path = os.path.dirname(target_path)
        os.makedirs(target_path, exist_ok=True)
        shutil.copy2(bbox, target_path)