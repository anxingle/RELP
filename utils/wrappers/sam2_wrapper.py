import os
import cv2
import numpy as np

class SAM2OnnxPredictor:
    def __init__(self, model_dir, device='cpu'):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is not installed. Please install it to use ONNX models.")

        self.device = device
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  
        
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == 'mps':
            providers = ['CPUExecutionProvider'] # Force CPU even if MPS requested due to SAM2 limitations

        files = os.listdir(model_dir)
        enc_file = next((f for f in files if 'encoder' in f and f.endswith('.onnx')), None)
        dec_file = next((f for f in files if 'decoder' in f and f.endswith('.onnx')), None)

        if not enc_file or not dec_file:
            raise FileNotFoundError(f"Could not find encoder/decoder ONNX files in {model_dir}")

        print(f"Loading ONNX Encoder: {enc_file}")
        self.encoder = ort.InferenceSession(os.path.join(model_dir, enc_file), sess_options, providers=providers)
        
        print(f"Loading ONNX Decoder: {dec_file}")
        self.decoder = ort.InferenceSession(os.path.join(model_dir, dec_file), sess_options, providers=providers)
        
        self.input_size = (1024, 1024)
        self.original_size = None
        self.features = None
        
        self.enc_input_name = self.encoder.get_inputs()[0].name
        
        enc_outputs = self.encoder.get_outputs()
        self.enc_output_indices = {}
        for i, out in enumerate(enc_outputs):
            if 'image_embed' in out.name: self.enc_output_indices['image_embeddings'] = i
            elif 'high_res_feats_0' in out.name: self.enc_output_indices['high_res_feats_0'] = i
            elif 'high_res_feats_1' in out.name: self.enc_output_indices['high_res_feats_1'] = i
        
        dec_inputs = self.decoder.get_inputs()
        self.dec_input_names = {}
        for inp in dec_inputs:
            name = inp.name
            if 'image_embed' in name: self.dec_input_names['image_embeddings'] = name
            elif 'high_res_feats_0' in name: self.dec_input_names['high_res_feats_0'] = name
            elif 'high_res_feats_1' in name: self.dec_input_names['high_res_feats_1'] = name
            elif 'point_coord' in name: self.dec_input_names['point_coords'] = name
            elif 'point_label' in name: self.dec_input_names['point_labels'] = name
            elif 'has_mask_input' in name: self.dec_input_names['has_mask_input'] = name
            elif 'mask_input' in name: self.dec_input_names['mask_input'] = name
            elif 'orig_im_size' in name: self.dec_input_names['orig_im_size'] = name

    def _preprocess_image(self, image):
        target_size = 1024
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(image, (new_w, new_h))
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - mean) / std
        
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        img_padded = np.pad(img_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        img_chw = img_padded.transpose(2, 0, 1)
        img_batch = img_chw[None, ...]
        return img_batch.astype(np.float32), scale

    def set_image(self, image):
        self.original_size = image.shape[:2]
        img_input, self.scale = self._preprocess_image(image)
        
        outputs = self.encoder.run(None, {self.enc_input_name: img_input})
        
        self.features = {}
        if self.enc_output_indices:
            self.features['image_embeddings'] = outputs[self.enc_output_indices['image_embeddings']]
            self.features['high_res_feats_0'] = outputs[self.enc_output_indices['high_res_feats_0']]
            self.features['high_res_feats_1'] = outputs[self.enc_output_indices['high_res_feats_1']]
        else:
            self.features['high_res_feats_0'] = outputs[0]
            self.features['high_res_feats_1'] = outputs[1]
            self.features['image_embeddings'] = outputs[2]

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        if self.features is None:
            raise RuntimeError("No image set. Call set_image() first.")

        coords = []
        labels = []
        
        if box is not None:
            box = np.array(box).reshape(-1, 4)
            for b in box:
                coords.append([b[0], b[1]])
                coords.append([b[2], b[3]])
                labels.append(2) 
                labels.append(3) 
        
        if point_coords is not None:
            point_coords = np.array(point_coords).reshape(-1, 2)
            point_labels = np.array(point_labels).reshape(-1)
            for i, p in enumerate(point_coords):
                coords.append(p)
                labels.append(point_labels[i])
        
        if not coords:
             return np.array([]), np.array([]), np.array([])

        coords_np = np.array(coords, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.float32)
        coords_np *= self.scale
        
        coords_batch = coords_np[None, :, :]
        labels_batch = labels_np[None, :]
        
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        orig_im_size = np.array(self.original_size, dtype=np.int32)[None, :]
        
        inputs = {
            self.dec_input_names.get('image_embeddings'): self.features['image_embeddings'],
            self.dec_input_names.get('high_res_feats_0'): self.features['high_res_feats_0'],
            self.dec_input_names.get('high_res_feats_1'): self.features['high_res_feats_1'],
            self.dec_input_names.get('point_coords'): coords_batch,
            self.dec_input_names.get('point_labels'): labels_batch,
            self.dec_input_names.get('mask_input'): mask_input,
            self.dec_input_names.get('has_mask_input'): has_mask_input,
            self.dec_input_names.get('orig_im_size'): orig_im_size
        }
        
        inputs = {k: v for k, v in inputs.items() if k is not None}
        
        outputs = self.decoder.run(None, inputs)
        masks = outputs[0] 
        scores = outputs[1] 
        
        if not multimask_output:
            best_idx = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best_idx, :, :][:, None, :, :]
            scores = scores[np.arange(scores.shape[0]), best_idx][:, None]
        
        orig_h, orig_w = self.original_size
        new_h = int(orig_h * self.scale)
        new_w = int(orig_w * self.scale)
        
        valid_h = max(1, int(new_h * 0.25))
        valid_w = max(1, int(new_w * 0.25))
        
        final_masks = []
        for m in masks[0]:
            m_crop = m[:valid_h, :valid_w]
            if m_crop.dtype != np.float32:
                m_crop = m_crop.astype(np.float32)
            
            # The original implementation in utils_zee used INTER_LINEAR without thresholding first!
            # The thresholding was handled later in generate_sam2_mask.
            m_orig = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            final_masks.append(m_orig)
            
        return np.array(final_masks), scores[0], None

def init_sam2(config_path, checkpoint_path, device='cpu'):
    try:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
             print(f"Warning: SAM2 checkpoint not found at {checkpoint_path}")
             return None
        
        if os.path.isdir(checkpoint_path) or (checkpoint_path.endswith('.onnx')):
            if os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.dirname(checkpoint_path)
                
            print(f"Initializing SAM2 ONNX from: {checkpoint_path}")
            try:
                sam2_model = SAM2OnnxPredictor(checkpoint_path, device)
                print("SAM2 ONNX Predictor initialized.")
                return sam2_model
            except Exception as e:
                print(f"Failed to init ONNX predictor: {e}")
                print("Falling back to standard init...")

        if not config_path or not os.path.exists(config_path):
             print(f"Warning: SAM2 config not found at {config_path}")
             return None
        
        original_cwd = os.getcwd()
        
        if 'configs' in config_path:
             idx = config_path.rfind('configs')
             config_dir = config_path[:idx]
             config_name = config_path[idx:]
        else:
             config_dir = os.path.dirname(config_path)
             config_name = os.path.basename(config_path)
        
        try:
            os.chdir(config_dir)
            from sam2.build_sam import build_sam2
            sam2_model = build_sam2(config_name, checkpoint_path, device='cpu')
            print("SAM2 model initialized.")
            return sam2_model
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"Warning: Could not initialize SAM2 model: {e}")
        return None

def get_sam2_predictor(model):
    """
    Helper to get a SAM2 predictor.
    """
    if model is None:
        return None
    
    if hasattr(model, 'predict') and hasattr(model, 'set_image'):
        return model
        
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(model)
    except ImportError:
        print("Warning: sam2 module not found.")
        return None
