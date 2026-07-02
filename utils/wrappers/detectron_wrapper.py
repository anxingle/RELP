import pickle
import os

def load_detectron_model(cfg_path, weights_path, device='cpu'):
    """
    Helper to load Detectron2 model from config and weights.
    """
    try:
        from detectron2.engine import DefaultPredictor
    except ImportError:
        DefaultPredictor = None
        print("Warning: detectron2 is not installed.")

    try:
        if not cfg_path or not weights_path:
            print("Error: cfg_path or weights_path is missing.")
            return None
            
        with open(cfg_path, 'rb') as f:
            cfg = pickle.load(f)
            
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        
        if DefaultPredictor:
            predictor = DefaultPredictor(cfg)
            print(f"Loaded Detectron2 model from {weights_path}")
            
            # --- Class Name Registration from classes.rtf (Ported from detectron_ops) ---
            import os
            import re
            try:
                from detectron2.data import MetadataCatalog
                dataset_name = None
                if hasattr(cfg, 'DATASETS'):
                    if len(cfg.DATASETS.TEST) > 0:
                        dataset_name = cfg.DATASETS.TEST[0]
                    elif len(cfg.DATASETS.TRAIN) > 0:
                        dataset_name = cfg.DATASETS.TRAIN[0]
                
                if dataset_name:
                    search_dirs = [os.path.dirname(cfg_path), os.path.dirname(weights_path)]
                    print(f"DEBUG: Looking for classes.rtf in: {search_dirs}")
                    rtf_path = None
                    for d in search_dirs:
                        if d and os.path.isdir(d):
                            p = os.path.join(d, 'classes.rtf')
                            if os.path.exists(p):
                                rtf_path = p
                                break
                    
                    if rtf_path:
                        print(f"Loading class names from {rtf_path}")
                        with open(rtf_path, 'r', errors='ignore') as f_rtf:
                            content_rtf = f_rtf.read()
                        
                        # Use robust RTF extraction to avoid escape characters splitting the text incorrectly
                        # E.g. '0: Case\n1: Cam_Rim'
                        text_part = content_rtf.split(r'\cf0')[-1].replace('}', '').replace('\\\n', '\n')
                        matches = re.findall(r'(\d+):\s*([a-zA-Z0-9_]+)', text_part)
                        if not matches:
                             matches = re.findall(r'(\d+):\s*([a-zA-Z0-9_]+)', content_rtf)

                        if matches:
                            matches.sort(key=lambda x: int(x[0]))
                            max_id = int(matches[-1][0])
                            class_names = ["Unknown"] * (max_id + 1)
                            for match in matches:
                                class_id = int(match[0])
                                class_name = match[1]
                                if class_id < len(class_names):
                                    class_names[class_id] = class_name
                            
                            MetadataCatalog.get(dataset_name).set(thing_classes=class_names)
                            print(f"Registered class names for '{dataset_name}': {class_names}")
                        else:
                            print("Warning: classes.rtf found but no 'ID: Name' pattern matched.")
                    else:
                        print("Warning: classes.rtf not found.")
            except Exception as e:
                print(f"Error registering class names: {e}")
            
            return predictor
        return None
    except Exception as e:
        print(f"Error loading Detectron2 model: {e}")
        return None
