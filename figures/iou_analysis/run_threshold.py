
import subprocess
import sys
from tqdm import tqdm
from pathlib import Path


script = Path(__file__).with_name("explore_iou_threshold.py")
for threshold in tqdm(range(1, 10), desc='Iterating threshold from 0.1 to 0.9...'):
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--folder_path",
            "/scr/yren/IoUs/",
            "--meta_path",
            "/scr/data/annotated_mn_datasets/metadata.csv",
            "--iou_threshold",
            f"{threshold / 10:.1f}",
        ],
        check=True,
    )
