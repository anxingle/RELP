import os
import pandas as pd
from typing import Dict, Any

from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.general_fm_ops import image_matches_mapping
from utils.base_utils import _norm

class GeneralFmDispatcher(AnalysisStrategy):
    """
    基于文件名的多模式调度器 (Dispatcher)。
    替代了之前 9000 行大循环中“强行改写变量跳转”的面条逻辑。
    
    职责:
    1. 根据 Flow 的 Defect Detection Setting 触发。
    2. 读取 General FM 里的 Mappings (如 B1, B2)。
    3. 扫描文件夹，对图片按照文件名进行分类打包。
    4. 将打包好的图片和对应的 General FM Rules 发送给真正的业务 Pipeline (如 K11p dent)。
    """

    def __init__(self, context: Dict[str, Any]):
        super().__init__(context)
        self.cm = ConfigManager()
        self.flow_df = self.cm.get_sheet('Flow')
        self.gfm_df = self.cm.get_sheet('General FM')

    def execute(self) -> None:
        fm = self.context.get('fm')
        print(f"\n🚀 [DISPATCHER] Starting General FM Dispatcher for '{fm}' 🚀")
        
        # Resolve product and generation if missing
        if not self.product or not self.generation or not self.download_path:
            fm_df = self.cm.get_sheet('Failure Mode')
            fm_matches = fm_df[fm_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm)]
            if not fm_matches.empty:
                self.product = str(fm_matches.iloc[0]['Product']).strip()
                self.generation = str(fm_matches.iloc[0]['Generation']).strip()
                self.context['product'] = self.product
                self.context['generation'] = self.generation
                
                # Fetch Pic_Path if it wasn't provided by the main script yet
                if not self.download_path:
                     raw_pic_path = str(fm_matches.iloc[0]['Pic_Path']).strip()
                     if raw_pic_path and raw_pic_path.lower() != 'nan':
                         self.download_path = raw_pic_path
                         self.context['download_path'] = self.download_path
                         
                print(f">>> [Dispatcher] Resolved Product: {self.product}, Generation: {self.generation}")

        # Also resolve General Defect Key from Flow
        flow_matches = self.flow_df[self.flow_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm)]
        general_defect_key = None
        if not flow_matches.empty:
            dds_cols = [c for c in self.flow_df.columns if _norm(c) in ('defectdetectionsetting',)]
            if dds_cols:
                val = str(flow_matches.iloc[0][dds_cols[0]]).strip()
                if val and val.lower() != 'nan':
                    val_str = str(val) if val is not None else ''
                    general_defect_key = val_str.replace('[Save]', '').replace('[save]', '').strip()
                    self.context['general_defect_key'] = general_defect_key
                    print(f">>> [Dispatcher] Resolved General Defect Key: {general_defect_key}")

        # 1. 拿 General FM 里的配置行
        gfm_matches = self.gfm_df[
            self.gfm_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm)
        ]
        
        if gfm_matches.empty:
            print(f">>> [Dispatcher] No routing rules found in General FM sheet for '{fm}'.")
            return
            
        print(f">>> [Dispatcher] Loaded {len(gfm_matches)} grouping rules from General FM.")
        
        # 2. 准备配置组字典
        # Key: 唯一的 Task Identifier (dest_fm, config_group)
        # Value: {'files': [], 'rules': []}
        task_batches = {}
        
        # 提取各个规则
        rules_list = []
        for _, row in gfm_matches.iterrows():
            rules_list.append({
                'Mapping': row.get('Mapping', ''),
                'Config_Group': row.get('Config_Group', 'Remaining'),
                'Scope': row.get('Scope', ''),
                'Defect Class': row.get('Defect Class', ''),
                'RELP Failure Mode': row.get('RELP Failure Mode', fm) # Default to original FM if empty
            })
            
        # 3. 扫描目录并打包文件
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        
        download_path = self.download_path if self.download_path is not None else ""
        if not os.path.isabs(download_path):
            download_path = os.path.join(project_root, download_path)
            
        if not os.path.exists(download_path):
            print(f">>> [Dispatcher] Target directory does not exist: {download_path}")
            return
            
        print(f">>> [Dispatcher] Scanning '{download_path}' for dynamic grouping...")
        
        # Find all valid images
        all_images = []
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file is not None and str(file).lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.heic')) and not str(file).startswith('.'):
                    all_images.append(os.path.join(root, file))
                    
        # Grouping Logic
        for image_path in all_images:
            matched = False
            
            # Priority 1: Check Mappings
            for rule in rules_list:
                mapping = rule['Mapping']
                if mapping and str(mapping).lower() != 'nan':
                    if image_matches_mapping(image_path, mapping):
                        self._add_to_batch(task_batches, rule, image_path)
                        matched = True
                        break
            
            # Priority 2: Fallback to Remaining (no Mapping matches)
            if not matched:
                for rule in rules_list:
                    if _norm(rule['Config_Group']) == 'remaining':
                        self._add_to_batch(task_batches, rule, image_path)
                        matched = True
                        break
                        
            # Priority 3: First rule if no remaining exists
            if not matched and rules_list:
                 self._add_to_batch(task_batches, rules_list[0], image_path)
                 
        # 4. Run Stage 1 (First Net - AI 本能提取 & 第四重网 - Scope 裁切)
        current_flow_matches = self.flow_df[self.flow_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm)]
        method_col = next((c for c in self.flow_df.columns if _norm(c) in ('defectidentification', 'defect_identification')), None)
        current_method = str(current_flow_matches.iloc[0][method_col]) if not current_flow_matches.empty and method_col else ''
        
        inherited_data = {}
        if current_method and current_method.lower() != 'nan':
            print(f"\n>>> [DISPATCHER] Phase 1: Executing base method '{current_method}' for '{fm}'...")
            from utils.core.strategies.strategy_factory import StrategyFactory
            stage1_context = self.context.copy()
            stage1_context['return_masks_only'] = True
            stage1_context['file_list'] = all_images
            stage1_context['general_fm_rules'] = rules_list # Pass all rules so Stage 1 can apply Scope!
            
            stage1_strategy = StrategyFactory.get_strategy(current_method, fm, stage1_context)
            if stage1_strategy:
                inherited_data = stage1_strategy.execute() or {}
                print(f">>> [DISPATCHER] Phase 1 completed. Inherited {len(inherited_data)} masks.")
            else:
                print(f">>> [DISPATCHER] Warning: Could not create Stage 1 strategy for '{current_method}'.")

        # 5. 执行分发 (Phase 2)
        print(f"\n>>> [Dispatcher] Grouping complete. Formed {len(task_batches)} distinct sub-tasks.")
        from utils.core.strategies.strategy_factory import StrategyFactory
        
        for task_key, batch_data in task_batches.items():
            dest_fm, config_group = task_key
            files_to_process = batch_data['files']
            rules = batch_data['rules']
            
            if not files_to_process:
                continue
                
            print(f"\n--- [DISPATCHING TASK] ---")
            print(f"  -> Target FM: {dest_fm}")
            print(f"  -> Target Config Group: {config_group}")
            print(f"  -> Images to process: {len(files_to_process)}")
            print(f"  -> Embedded Rules: {rules}")
            
            # 找到目标 Flow 的 Defect Identification
            flow_matches = self.flow_df[
                (self.flow_df['Failure Mode'].astype(str).apply(_norm) == _norm(dest_fm)) &
                (self.flow_df['Config_Group'].astype(str).apply(_norm) == _norm(config_group))
            ]
            
            if flow_matches.empty:
                # 尝试不带 Config_Group 找兜底
                flow_matches = self.flow_df[
                    self.flow_df['Failure Mode'].astype(str).apply(_norm) == _norm(dest_fm)
                ]
                
            if flow_matches.empty:
                print(f"Warning: Target Failure Mode '{dest_fm}' not found in Flow sheet. Skipping.")
                continue
                
            target_flow = flow_matches.iloc[0]
            
            # Get identification method
            method = str(target_flow[method_col]) if method_col else 'Unknown'
            
            # 构造子任务的 Context
            sub_context = self.context.copy()
            sub_context['fm'] = dest_fm
            sub_context['config_group_override'] = config_group
            sub_context['file_list'] = files_to_process
            sub_context['general_fm_rules'] = rules # Inject the Scope & Defect Class rules!
            sub_context['inherited_data'] = {f: inherited_data[f] for f in files_to_process if f in inherited_data}
            
            strategy = StrategyFactory.get_strategy(method, dest_fm, sub_context)
            if strategy:
                strategy.execute()
            else:
                print(f"Warning: Could not create strategy for '{method}' -> '{dest_fm}'.")
                
        print(f"\n✅ [DISPATCHER] All sub-tasks completed for '{fm}'.")

    def _add_to_batch(self, batches, rule, image_path):
        dest_fm = str(rule['RELP Failure Mode'])
        if dest_fm.lower() == 'nan':
             dest_fm = self.context.get('fm')
        
        config_group = str(rule['Config_Group'])
        if config_group.lower() == 'nan':
            config_group = ''
            
        task_key = (dest_fm, config_group)
        if task_key not in batches:
            batches[task_key] = {
                'files': [],
                'rules': [rule] # Store the rule for Scope filtering later
            }
        batches[task_key]['files'].append(image_path)
