"""SAM3 本地图像分割封装（官方 sam3 包，不需要 SAM2 的 YAML 配置）。

资源目录（以下路径均相对 RELP 仓库根目录）：
    RELP/
    ├─ utils/wrappers/sam3_wrapper.py
    ├─ docs/dut_and_center_sam3.py
    └─ weights/sam3/
       ├─ checkpoints/sam3.pt                  # 官方 SAM3 图像模型权重
       └─ assets/bpe_simple_vocab_16e6.txt.gz  # 文本提示分词用 BPE 词表

配置说明：这个官方图像模型不需要 SAM2 风格的 YAML 或单独 config_path。
模型结构由已安装 sam3 包中的 sam3/model_builder.py 定义；本 wrapper 不
读取 Hugging Face 模型库的 config.json，也不需要将它另存到 weights/。
安装 sam3 包仍是必要条件，单独放置上述两个资源文件不能替代 Python 包。

常规初始化（推荐）：
    from utils.wrappers.sam3_wrapper import init_sam3
    predictor = init_sam3(device="cuda")

默认 checkpoint/BPE 通过本文件位置定位到仓库根目录，不依赖当前工作目录。
若要显式指定资源，可从 RELP 根目录使用以下相对路径：
    predictor = init_sam3(
        checkpoint_path="weights/sam3/checkpoints/sam3.pt",
        bpe_path="weights/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device="cuda",
    )
注意：显式传入的相对路径按调用进程的当前工作目录解析；从其他目录调用时，
优先省略这两个参数使用默认值，或自行以仓库根目录拼接路径。

两个文件必须事先存在。builder 始终传 load_from_HF=False；本模块不下载
任何文件，也不改变进程的 HF 环境变量，以免影响 Grounding DINO 等其他模型。
本地 BPE 显式传入，避免包升级后悄悄换词表；不做包内词表的隐式回退。
API 核查基于 facebookresearch/sam3 commit：
0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc。

颜色约定：set_image / segment_text 只接受 RGB uint8 HWC ndarray 或 RGB PIL。
OpenCV 读出的 BGR 必须在调用处转一次 RGB；wrapper 不交换通道：

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor = get_sam3_predictor(init_sam3(device="cuda"))
    predictor.set_image(image_rgb)
    masks, iou_scores, low_res_logits = predictor.predict(
        box=np.array([x1, y1, x2, y2]), multimask_output=False,
    )
    # masks 是 0/1；保存灰度 mask 时转为 uint8 并乘 255。
    cv2.imwrite(mask_path, masks[0].astype(np.uint8) * 255)
    # 保存彩色裁剪仍从 image_bgr 裁剪，不把 RGB 数组直接交给 cv2.imwrite。
    cv2.imwrite(crop_path, image_bgr[y1:y2, x1:x2])

    result = predictor.predict_text("ipad textile case")
    # result: masks[N,H,W] bool, boxes[N,4] XYXY 原图像素, scores[N] 文本置信度。
    # 多个实例全部保留；调用方按 DUT 规则选择，不默认合并、腐蚀或填洞。
    predictor.reset_image()  # 大批处理每张图结束后释放图像特征。

安装/兼容性注意：需安装官方 sam3 包及其推理依赖。SAM3 的上游依赖声明含
numpy<2，RELP 当前环境含 NumPy 2；不要为安装 SAM3 自动降级整个 RELP 环境。
上游还依赖 Triton；Windows 可使用与 PyTorch 匹配的 triton-windows，Linux
使用 triton。该版上游的位置编码构造函数写死 CUDA，故本封装要求 CUDA，
不承诺 CPU/MPS 或 ONNX 支持。这个 wrapper 不修改依赖版本或第三方源码。

当前 Windows RELP 环境的增量安装命令（从仓库根目录执行，已有 torch、
torchvision、numpy、Pillow、iopath、regex、pycocotools、huggingface_hub）：
    uv pip install --python .venv/Scripts/python.exe --no-deps `
        "git+https://github.com/facebookresearch/sam3.git@0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc" `
        timm==1.0.30 ftfy==6.1.1 wcwidth==0.9.2 psutil==7.2.2
    uv pip install --python .venv/Scripts/python.exe --no-deps triton-windows==3.8.0.post29
这里 --no-deps 是有意保护现有 RELP 依赖，不代表已满足上游 NumPy<2 声明。
uv sync 会移除这些未在 pyproject.toml 中声明的附加包；需保留时用
uv sync --inexact，或之后重跑上述增量安装。不为此改动 RELP 全局依赖约束。
Linux 应使用对应 Linux Python 和 PyTorch 配套的 triton，不装 triton-windows。

sam3.pt 受 Hugging Face 访问许可限制；账号获批并完成本机 hf auth login 后，
可显式执行一次（这不是模型初始化时自动下载）：
    .venv/Scripts/hf.exe download facebook/sam3 sam3.pt --local-dir weights/sam3/checkpoints
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "weights/sam3/checkpoints/sam3.pt"
DEFAULT_BPE_PATH = _REPO_ROOT / "weights/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


class SAM3TextResult(TypedDict):
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray


def _local_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(
            f"SAM3 {description} must be a non-empty local file: {resolved}. "
            "Automatic Hugging Face downloads are disabled."
        )
    return resolved


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    # SAM3 processor 返回 CUDA Tensor；公共结果放到 CPU，避免调用方持有显存。
    # NumPy 不支持 torch.bfloat16；AMP 的浮点输出先转 float32。
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


class SAM3Predictor:
    """有状态的单图预测器；同一实例不可跨线程并发使用。

    每张图调用一次 set_image，随后可多次使用点/框或文本提示。官方
    model.predict_inst(state, ...) 会复用 processor 已编码的特征；不另行
    调 interactive_predictor.set_image，避免重复运行图像 backbone。

    device 默认 CUDA；不可用 CUDA 或指定 CPU/MPS 会报错。上游虽然暴露
    device='cpu' 参数，但该版本构建时仍创建 CUDA 位置编码。
    该版本视觉 MLP 的融合算子强制输出 BF16，不能直接关闭 autocast 使用
    float32：后续 Linear 会出现 dtype 不匹配。因此 use_amp 默认 True，
    不支持 False，需 GPU 支持 BF16。本封装只在单次调用内启用 autocast；上游
    sam3.model_builder 在 import 时可能启用 TF32，属于官方实现的行为。
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        bpe_path: str | Path | None = None,
        device: str | torch.device | None = None,
        confidence_threshold: float = 0.5,
        use_amp: bool = True,
    ) -> None:
        # 缺少资源时先报错，不尝试联网，也不先创建大型模型。
        self.checkpoint_path = _local_file(
            DEFAULT_CHECKPOINT_PATH if checkpoint_path is None else checkpoint_path,
            "checkpoint",
        )
        self.bpe_path = _local_file(
            DEFAULT_BPE_PATH if bpe_path is None else bpe_path, "BPE vocabulary"
        )
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        self.device = torch.device(device if device is not None else "cuda")
        if self.device.type != "cuda":
            raise ValueError(
                "This SAM3 runtime requires CUDA: the official pinned builder creates "
                "CUDA positional encodings even when device='cpu'. CPU/MPS are unsupported."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this Python environment.")
        if not use_amp:
            raise ValueError(
                "This pinned SAM3 runtime requires use_amp=True: its fused visual MLP "
                "outputs bfloat16 and FP32 inference without autocast has incompatible dtypes."
            )
        self.use_amp = True
        with torch.cuda.device(self.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("SAM3 requires a CUDA GPU with bfloat16 support.")

        # 延迟导入：安装 sam3 前仍可 import 本模块、检查路径与接口。
        try:
            from sam3.model_builder import build_sam3_image_model, _load_checkpoint
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
        except ImportError as exc:
            raise ImportError(
                "Cannot import the official sam3 image runtime or a required dependency. "
                "See this module's installation notes; keep RELP's existing dependency versions. "
                f"Original import error: {exc}"
            ) from exc

        # 上游部分版本的 builder 只识别字符串 'cuda'，不能正确搬移 cuda:1。
        # 权重先加载到 CPU 再显式 .to(device)，processor 使用同设备。上游构建
        # 还会创建未注册为 buffer 的 CUDA 位置编码缓存，需在指定 GPU 上构建，
        # 仅 .to(cuda:1) 不足以搬移这些缓存；context 退出会恢复此前当前 GPU。
        with torch.cuda.device(self.device):
            self.model = build_sam3_image_model(
                checkpoint_path=None,
                bpe_path=str(self.bpe_path),
                device="cpu",
                load_from_HF=False,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )
            # 该版本 build_tracker() 的构造函数永久 __enter__ 一个 BF16
            # autocast。图像接口不需要这项全局副作用：先退出它，恢复调用者
            # 之前的精度上下文。随后才加载权重，避免坏 checkpoint 的异常
            # 路径留下已开启的 autocast。load_from_HF=False 保证这里也不下载。
            tracker_amp = getattr(self.model.inst_interactive_predictor.model, "bf16_context", None)
            if tracker_amp is not None:
                tracker_amp.__exit__(None, None, None)
                self.model.inst_interactive_predictor.model.bf16_context = None
            # 使用 pinned 官方包的本地加载函数，保留 detector/tracker 参数
            # 映射规则；这不是远程加载接口。升级 sam3 时需重新核查此私有 API。
            _load_checkpoint(self.model, str(self.checkpoint_path))
            self.model = self.model.to(self.device).eval()
        # 上游交互预测器默认填掉低分辨率 mask 中面积 <=256 的孔洞。工业
        # DUT 的开孔/缺口可能有意义，使用公开构造参数关闭这项后处理。
        # 复用已加载的 tracker，既不重载权重，也不改变模型结构。
        self.model.inst_interactive_predictor = SAM3InteractiveImagePredictor(
            self.model.inst_interactive_predictor.model,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        ).eval()
        self.processor = Sam3Processor(
            self.model, device=str(self.device), confidence_threshold=confidence_threshold
        )
        self._state: dict | None = None

    def _autocast(self):
        # 即便调用方的外层 AMP 使用 float16，这里仍按上游所需的 BF16 推理；
        # 每次退出都恢复调用方状态，不影响同线程后续其他模型的推理精度。
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp
        )

    def set_image(self, image_rgb: np.ndarray | Image.Image) -> None:
        """缓存一张 RGB 图的特征；输入不会再做 BGR→RGB 转换。

        ndarray 必须为非空 H×W×3、uint8 [0,255]。RGB PIL 也可用；拒绝灰度、
        RGBA、float 图，避免隐式颜色/范围解释。是否误传 BGR 无法由数值判断，
        调用方必须遵守 RGB 约定。
        """
        if isinstance(image_rgb, np.ndarray):
            if image_rgb.dtype != np.uint8:
                raise TypeError("image_rgb must have dtype uint8, in the range [0,255].")
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or 0 in image_rgb.shape:
                raise ValueError("image_rgb must be a non-empty HxWx3 RGB array.")
            # 官方 processor 的 ndarray 分支按最后两维读取高宽，HWC 会读错。
            # 直接从 RGB 数据创建 PIL，既保护颜色，也准确保留原图 H/W。
            pil_rgb = Image.fromarray(np.ascontiguousarray(image_rgb))
        elif isinstance(image_rgb, Image.Image):
            if image_rgb.mode != "RGB" or min(image_rgb.size) == 0:
                raise ValueError("PIL image must be a non-empty image in RGB mode.")
            pil_rgb = image_rgb
        else:
            raise TypeError("image_rgb must be an RGB uint8 numpy array or an RGB PIL image.")

        # 若下一张图编码失败，不让后续 predict 误用上一张图的特征。
        self.reset_image()
        with torch.inference_mode(), self._autocast():
            self._state = self.processor.set_image(pil_rgb)

    def _image_state(self) -> dict:
        if self._state is None:
            raise RuntimeError("No image set. Call set_image(image_rgb) first.")
        return self._state

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SAM2 风格的点/框分割，原样返回官方 (masks, iou_scores, low_res_logits)。

        point_coords: N×2 XY，point_labels: N 个 1(前景)/0(背景)；box: XYXY。
        normalize_coords=True（默认）时传原图像素坐标；False 时传 [0,1]。
        单提示的输出为 C×H×W、C、C×h×w；多个 box（B×4）时保留官方
        B×C×H×W、B×C、B×C×h×w。同批 points 也必须与 B 维对齐。
        C 为候选 mask 数，不是对象数。return_logits=False 时 masks 为 0/1；
        True 时为原图 logits。第三项始终为低分辨率 logits，可作为下一轮
        mask_input（单提示使用 low_res_logits[best:best+1]）。不要写死 h/w=256。

        文本 prompt 不会参与此接口；若要 VLM/Grounding DINO 框辅助分割，
        在调用处把它的框转换为原图 XYXY 后传 box。DINO 的颜色约定独立。
        """
        state = self._image_state()
        if (point_coords is None) != (point_labels is None):
            raise ValueError("point_coords and point_labels must be provided together.")
        with torch.inference_mode(), self._autocast():
            try:
                return self.model.predict_inst(
                    state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    mask_input=mask_input,
                    multimask_output=multimask_output,
                    return_logits=return_logits,
                    normalize_coords=normalize_coords,
                )
            finally:
                # 官方桥接在异常路径可能遗留 predictor 特征；公开 reset 负责清理。
                self.model.inst_interactive_predictor.reset_predictor()

    def predict_text(self, prompt: str) -> SAM3TextResult:
        """对当前图像进行独立文本分割，返回 CPU numpy 的实例结果。

        masks: N×H×W bool（像素阈值为官方默认 0.5），boxes: N×4 float32
        XYXY 原图像素，scores: N float32。无目标时 N=0，仍保留完整维度。
        scores 为文本检测/存在置信度，与 predict 的 IoU 预测分数不同。
        confidence_threshold 控制实例置信度过滤，不是像素阈值。
        使用简短目标名称，如 'ipad case' / 'textile tablet case'；它不保证
        是 DUT，返回实例应由调用方按业务规则检查。不要用它直接判断缺陷。
        """
        state = self._image_state()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        with torch.inference_mode(), self._autocast():
            self.processor.reset_all_prompts(state)
            try:
                output = self.processor.set_text_prompt(prompt.strip(), state)
                # 官方 N×1×H×W 去掉通道维，绝不 squeeze 掉只有一个实例的 N 维。
                return SAM3TextResult(
                    masks=_numpy(output["masks"]).squeeze(1).astype(bool, copy=False),
                    boxes=_numpy(output["boxes"]).astype(np.float32, copy=False),
                    scores=_numpy(output["scores"]).astype(np.float32, copy=False),
                )
            finally:
                # 原图尺度的 GPU masks 用完即释放，保留 backbone 供下次提示复用。
                # 上游名为 masks_logits 的字段其实已 sigmoid，是概率，不是 logits；
                # 不将它冒充 predict() 的第三项返回。
                self.processor.reset_all_prompts(state)

    def segment_text(self, image_rgb: np.ndarray | Image.Image, prompt: str) -> SAM3TextResult:
        """set_image 后 predict_text；保留当前特征供追加点/框提示。

        每张图处理结束须调用 reset_image；下一次 set_image 也会释放旧图特征，
        因此不会累积多张图的特征。
        """
        self.set_image(image_rgb)
        return self.predict_text(prompt)

    def reset_image(self) -> None:
        """释放当前图像与提示特征；模型权重仍保留，可继续 set_image。

        长批处理每张图结束调用本方法；显式 GC 与清理 CUDA 空闲缓存遵循仓库
        的显存管理约定。有引用的权重仍驻留显存，empty_cache 不会卸载模型。
        """
        self._state = None
        self.model.inst_interactive_predictor.reset_predictor()
        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()


def init_sam3(
    checkpoint_path: str | Path | None = None,
    *,
    bpe_path: str | Path | None = None,
    device: str | torch.device | None = None,
    confidence_threshold: float = 0.5,
    use_amp: bool = True,
) -> SAM3Predictor:
    """对应 init_sam2 的初始化入口；SAM3 不需要单独 config_path。

    默认资源：weights/sam3/checkpoints/sam3.pt 与
    weights/sam3/assets/bpe_simple_vocab_16e6.txt.gz（锚定 RELP 根目录）。
    若显式传入相对路径，则按当前工作目录解析，推荐在 RELP 根目录调用。
    失败抛出带原始原因的异常，避免返回 None 后批处理静默漏掉 DUT 分割。
    """
    return SAM3Predictor(
        checkpoint_path,
        bpe_path=bpe_path,
        device=device,
        confidence_threshold=confidence_threshold,
        use_amp=use_amp,
    )


def get_sam3_predictor(model: SAM3Predictor | None) -> SAM3Predictor | None:
    """对应 get_sam2_predictor；init_sam3 返回的已经是完整预测器。"""
    if model is None or isinstance(model, SAM3Predictor):
        return model
    raise TypeError("Pass the SAM3Predictor returned by init_sam3().")
