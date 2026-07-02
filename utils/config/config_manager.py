import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

# --- 强类型 Dataclasses 定义 ---

@dataclass
class FailureModeConfig:
    product: str
    generation: str
    failure_mode: str
    sam_mode: Optional[str] = None
    pic_download_mode: Optional[str] = None
    pic_path: Optional[str] = None
    radar: Optional[str] = None
    radar_time_window: Optional[str] = None
    output_path: Optional[str] = None

@dataclass
class FlowConfig:
    failure_mode: str
    debug_mode: Optional[str] = None
    sn_mapping: Optional[str] = None
    defect_detection_debug: Optional[str] = None
    defect_detection_setting: Optional[str] = None
    config_group: Optional[str] = None
    defect_identification: Optional[str] = None
    upsampling: Optional[str] = None
    upsampling_definition: Optional[str] = None
    post_processing: Optional[str] = None
    reference: Optional[str] = None
    input_dim: Optional[Any] = None
    dino_input_dim: Optional[Any] = None
    crop_output_dim: Optional[Any] = None
    pca_alignment: Optional[str] = None
    fine_tuning: Optional[str] = None
    large_dut: Optional[str] = None
    large_dut_height_ratio: Optional[float] = None

@dataclass
class OutputConfig:
    product: str
    generation: str
    failure_mode: str
    defect_output_format: Optional[str] = None
    mask_color: Optional[str] = None
    parametrics: Optional[str] = None
    curved_line_measurement: Optional[str] = None
    color_space_conversion: Optional[str] = None
    color_space_invertion: Optional[str] = None
    color_space_binning: Optional[str] = None
    scoring: Optional[str] = None
    scoring_setting: Optional[str] = None
    advanced_scoring: Optional[str] = None

# --- 单例配置管理器 ---

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.is_initialized = False
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if self.is_initialized:
            return
            
        self.excel_path = None
        
        # Raw DataFrames Cache
        self.sheets_cache: Dict[str, pd.DataFrame] = {}
        
        # Typed Objects Cache
        self.failure_modes: List[FailureModeConfig] = []
        self.flow_configs: Dict[str, FlowConfig] = {}
        self.output_configs: Dict[str, OutputConfig] = {}
        
        self.is_initialized = True

    def load_excel(self, file_path: str):
        """一次性加载所有需要的表单进入内存"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found at {file_path}")
        
        self.excel_path = file_path
        print(f"ConfigManager: Loading configuration from {file_path} ...")
        
        xl = pd.ExcelFile(file_path)
        required_sheets = [
            'Failure Mode', 'Flow', 'Weights', 'General FM', 
            'Filtering', 'Output', 'Adaptive Gaussian', 'KNN', 
            'Dino_TH', 'Scaling', 'Fine_Tuning', 'Reference'
        ]
        
        for sheet in required_sheets:
            if sheet in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                print(f"Loaded {sheet}: {df.shape}")
                # 将全空白或者全NaN的行清掉，避免造成干扰
                df = df.dropna(how='all') 
                self.sheets_cache[sheet] = df
            else:
                print(f"Warning: ConfigManager could not find sheet '{sheet}'")
                self.sheets_cache[sheet] = pd.DataFrame()
                
        self._parse_typed_configs()
        print("ConfigManager: Loading completed.")

    def _parse_typed_configs(self):
        """将部分高频调用的DataFrame转化为强类型Dataclass"""
        
        # 1. Parse Failure Mode
        fm_df = self.sheets_cache.get('Failure Mode', pd.DataFrame())
        if not fm_df.empty:
            for _, row in fm_df.iterrows():
                # Replace nan with None
                r = row.replace({np.nan: None}).to_dict()
                if not r.get('Failure Mode'): continue
                
                self.failure_modes.append(
                    FailureModeConfig(
                        product=str(r.get('Product', '')),
                        generation=str(r.get('Generation', '')),
                        failure_mode=str(r.get('Failure Mode')),
                        sam_mode=r.get('SAM Mode'),
                        pic_download_mode=r.get('Pic_Download_Mode'),
                        pic_path=r.get('Pic_Path'),
                        radar=r.get('Radar'),
                        radar_time_window=r.get('Radar_Time_Window'),
                        output_path=r.get('Output_Path')
                    )
                )

        # 2. Parse Flow Config
        flow_df = self.sheets_cache.get('Flow', pd.DataFrame())
        if not flow_df.empty:
            for _, row in flow_df.iterrows():
                r = row.replace({np.nan: None}).to_dict()
                fm = r.get('Failure Mode')
                if not fm: continue
                
                self.flow_configs[str(fm)] = FlowConfig(
                    failure_mode=str(fm),
                    debug_mode=r.get('Debug Mode'),
                    sn_mapping=r.get('SN Mapping'),
                    defect_detection_debug=r.get('Defect Detection Debug'),
                    defect_detection_setting=r.get('Defect Detection Setting'),
                    config_group=r.get('Config_Group'),
                    defect_identification=r.get('Defect Identification'),
                    upsampling=r.get('Upsampling'),
                    upsampling_definition=r.get('Upsampling Definition'),
                    post_processing=r.get('Post Processing'),
                    reference=r.get('Reference'),
                    input_dim=r.get('Input_Dim'),
                    dino_input_dim=r.get('Dino_Input_Dim'),
                    crop_output_dim=r.get('Crop_Output_Dim'),
                    pca_alignment=r.get('PCA_Alignment'),
                    fine_tuning=r.get('Fine Tuning'),
                    large_dut=r.get('Large_DUT'),
                    large_dut_height_ratio=float(r['Large_DUT_Height_Ratio']) if r.get('Large_DUT_Height_Ratio') is not None else None
                )

    def get_sheet(self, sheet_name: str) -> pd.DataFrame:
        """为尚未完成类重构的代码提供兜底，直接返回缓存的 DataFrame"""
        return self.sheets_cache.get(sheet_name, pd.DataFrame())

    def get_flow(self, failure_mode: str) -> Optional[FlowConfig]:
        """获取特定 Failure Mode 的 Flow Config"""
        return self.flow_configs.get(str(failure_mode))

    def get_all_failure_modes(self) -> List[FailureModeConfig]:
        """获取所有激活的 Failure Mode"""
        return self.failure_modes

