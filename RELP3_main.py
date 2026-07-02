import os
import sys
import argparse

# Path loading
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
detectron_path = os.path.join(current_dir, 'detectron_all')
sam2_path = os.path.join(current_dir, 'weights', 'sam2')

for p in [utils_path, detectron_path, sam2_path, current_dir]:
    if p not in sys.path:
        sys.path.append(p)
# -------------------------------------------------------------

from utils.config.config_manager import ConfigManager

# Enable MPS fallback for ops not implemented on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from utils import utils_general
from utils.core.pipeline import DefectDetectionPipeline

def main():
    print("Initializing RELP3 System...")
    
    # 全局配置路径
    CONFIG_PATH = os.path.join(current_dir, "RELP_Configuration.xlsx")

    # 初始化并加载全量配置
    ConfigManager().load_excel(CONFIG_PATH)

    print("Entering Main Defect Detection Pipeline...")
    pipeline = DefectDetectionPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
