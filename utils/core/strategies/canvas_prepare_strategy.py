import os
import cv2
import numpy as np
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
import utils.utils_general as ug
from utils import cv_ops, crop_ops, filtering_ops, output_ops

class CanvasPrepareStrategy(AnalysisStrategy):
    """
    Canvas Prepare 策略：
    代表了最优美的纯粹视觉管线：
    1. 原图找本体并用 SAM 抠图
    2. 切黑边 (Tight Crop) 并变成带 Padding 的规整方形画布 (Canvas)
    3. 在画布上做一切后续检测，绝不拉伸/回退坐标
    """
    def execute(self) -> None:
        print(">>> [Canvas Prepare Strategy] Executing...")
        cfg_mgr = ConfigManager()
        fm = self.fm
        
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        fm_row = fm_df[fm_df['Failure Mode'] == fm].iloc[0] if not fm_df.empty else None
        
        self.product = str(fm_row['Product']) if fm_row is not None else self.product
        self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
        self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
        
        # Load defect filter config (simulating Mac_Classes logic)
        defect_filter_config, class_names = ug.load_defect_config("RELP_Configuration.xlsx", sheet_name="Mac_Classes")
        
        # Resolve models (Tape detector + Defect detector)
        # In a real environment, we'd pull these dynamically. For now we use the ones that K11p used.
        weights_df = cfg_mgr.get_sheet('Weights')
        tape_detector = ug.load_detectron2_model(
            "weights/macbook/K11p/DUT/OD_CFG.pickle",
            "weights/macbook/K11p/DUT/model_final.pth"
        )
        defect_detector = ug.load_detectron2_model(
            "weights/General_FM/K11p/OD_CFG.pickle",
            "weights/General_FM/K11p/model_final.pth"
        )
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from pathlib import Path
        
        sam2_model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(Path.cwd().parent / "sam2/checkpoints/sam2.1_hiera_tiny.pt"),
            device="cpu"
        )
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        print(">>> [Canvas Prepare Strategy] Models loaded.")

        input_path = Path(self.download_path)
        output_base_str = str(fm_row['Output_Path']).strip() if fm_row is not None and pd.notna(fm_row.get('Output_Path')) else "Result"
        if output_base_str.lower() == 'nan': output_base_str = "Result"
        output_base = Path(output_base_str)
        
        output_folder_name = f"{self.product}_{self.generation}_{fm}_Result"
        output_dir_base = output_base / output_folder_name
        output_dir_defects = output_dir_base / "defects"
        output_dir_parametric = output_dir_base
        
        output_dir_defects.mkdir(parents=True, exist_ok=True)
        output_dir_parametric.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            print(f"❌ Input path does not exist: {input_path}")
            return
            
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        
        all_parametric_results = []
        parametric_cols_order = []
        
        for img_path in image_paths:
            print(f"\nProcessing image: {img_path.name}")
            
            # THE "CANVAS PREPARE" CORE
            centered_image, dut_contour = ug.process_image(
                img_path, sam2_predictor, tape_detector
            )
            
            if centered_image is None:
                print(" failed to prepare canvas. Skipping.")
                continue
                
            # Perform detection on the PERFECT CANVAS
            outputs = defect_detector(centered_image)
            instances = outputs["instances"]
            overlay_image = centered_image.copy()
            final_combined_mask = np.zeros(centered_image.shape[:2], dtype=np.uint8)
            found_specific_defect = False
            
            if len(instances) > 0:
                pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                
                for i, (box, class_id) in enumerate(zip(pred_boxes, pred_classes)):
                    class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
                    
                    if class_name in defect_filter_config and defect_filter_config[class_name]["filter_chain"]:
                        found_specific_defect = True
                        x1, y1, x2, y2 = map(int, box)
                        
                        cropped_image = centered_image[y1:y2, x1:x2]
                        if cropped_image.size == 0: continue
                        
                        filter_config = defect_filter_config[class_name]
                        precise_mask = ug.apply_filter_chain(cropped_image, filter_config.get("filter_chain", []))
                        
                        if precise_mask is not None:
                            full_mask = np.zeros(centered_image.shape[:2], dtype=np.uint8)
                            full_mask[y1:y2, x1:x2] = precise_mask
                            overlay_image[full_mask > 0] = (255, 0, 255) # Magenta highlight
                            final_combined_mask = cv2.bitwise_or(final_combined_mask, full_mask)
            
            if found_specific_defect:
                output_path = output_dir_defects / f"{img_path.stem}_overlay.jpg"
                cv2.imwrite(str(output_path), overlay_image)
                print(f"  ✅ Saved perfect overlay to {output_path}")
                
                parametric_row, p_cols = ug.generate_parametric_row(
                    img_path.name, centered_image, final_combined_mask, dut_contour
                )
                all_parametric_results.append(parametric_row)
                if not parametric_cols_order:
                    parametric_cols_order = ['Filename', 'Defect Pct'] + p_cols + ['Weighted Score']
        
        if all_parametric_results:
            df_out = pd.DataFrame(all_parametric_results)
            if parametric_cols_order:
                existing_cols = [c for c in parametric_cols_order if c in df_out.columns]
                df_out = df_out[existing_cols]
            
            out_excel = output_dir_parametric / "Parametric_Output.xlsx"
            df_out.to_excel(out_excel, index=False)
            print(f"\n✅ Pipeline Complete! Output saved to: {out_excel}")
        else:
            print("\n⚠️ No defects found, Parametric Output not generated.")

