经过对 `/Users/an/workspace/RELP` 仓库各层级源码、配置文件和数据目录的全面检查，现将该仓库的代码架构、设计模式以及各个脚本的核心作用详细梳理如下：

---

### 一、 项目定位与核心业务架构

**RELP (Reliability Engineering / Defect Detection & Metrology Platform)** 是一个面向硬件制造与可靠性测试（Reliability Testing）的自动化缺陷检测、形貌计量与质量评分平台。

#### 1. 核心业务流程
1. **自动化采集与任务监控**：从内部 Bug 跟踪系统（Radar）自动拉取各实验组测试图集，完成认证鉴权与防休眠保活。
2. **多模态/大模型视觉推理 (Foundation Models)**：
   - **SAM 2 / SAM 2.1 (Segment Anything Model 2)**：实现任意物体与缺陷像素级高精度轮廓提取（支持 ONNX 和 PyTorch 原生推理）。
   - **Grounding DINO**：基于开放词表文本 Prompt（如 `"elliptic camera"`, `"red tag"`）的零样本（Zero-Shot）部件与缺陷定位。
   - **DINOv2 / DINOv3 + AnyUp**：利用无监督/自监督视觉特征计算异常距离场（Anomaly Map），并通过 AnyUp 进行特征图跨注意力高清上采样。
   - **Detectron2**：有监督的目标检测与实例分割（用于 DUT 主体定位、按键/保护膜检测、特定缺陷分类）。
3. **传统机器视觉过滤与空间惩罚 (Classical CV & Spatial Penalties)**：
   - 包含自适应高斯滤波（Adaptive Gaussian）、形态学 Top-Hat/Black-Hat、HSV/RGB 空间颜色距离过滤、面积与长宽比筛选、孤立锚点匹配（Anti-Nesting Filter）以及关键结构应力区空间惩罚加权（Spatial Penalty Mask）。
4. **精密量测与打分系统 (Metrology & Scoring)**：
   - 结合参考靶标（Reference Tag）或真实 CAD 几何尺寸计算物理毫米级指标（面积 $mm^2$、长度、宽度、曲线弧长等）。
   - 基于特征分箱（Binning）、指数衰减加权以及与人类工程师打分对齐的回归模型进行严重度评分。
5. **报表与可视化闭环**：
   - 自动生成 `Parametric_Output.xlsx` 参数化报表及包含半透明缺陷轮廓的 `_overlay.jpg` 渲染图，打包回传上传至指定 Radar。

---

### 二、 架构模式与技术亮点

- **配置驱动架构 (Configuration-Driven)**：全局由 `RELP_Configuration.xlsx` 完全驱动，通过 `Auto Run`、`Failure Mode`、`Flow`、`Weights`、`Filtering`、`Output` 等多表实现业务流、算子参数与模型权重的完全解耦。
- **绞杀者模式 (Strangler Fig Pattern) 与策略模式 (Strategy Pattern)**：
  系统已摆脱早期单脚本上万行的过程式大循环，通过 `StrategyFactory` 抽象工厂将每种失效模式或算法流程封装为高内聚的 `AnalysisStrategy` 策略子类。
- **动态图/JSON 配置驱动 (DAG Dynamic Routing)**：
  支持通过 JSON 文件（如 `Textile_R692_bubble.json`）声明式编排多节点流水线（Grounding DINO $\to$ SAM2 $\to$ Slicing $\to$ DINOv3 $\to$ ParametricOutput）。

---

### 三、 代码仓库结构图

```text
/Users/an/workspace/RELP
├── RELP_Configuration.xlsx     # 全局主配置文件 (Excel 驱动核心)
├── RELP3_main.py               # 本地/批处理主运行入口
├── RELP_autorun.py             # 7x24h 自动化后台守护服务 (Radar交互、防休眠、调度)
├── pyproject.toml / uv.lock    # Python 项目依赖与环境配置 (基于 uv，Python 3.12)
├── anyup-main/                 # AnyUp 视觉特征超分模块 (ViT/DINO 特征高清重构)
│   ├── anyup/                  # AnyUp 模型网络架构、Cross-Attention 与层定义
│   ├── hubconf.py              # Torch Hub 加载入口
│   └── train.py                # AnyUp 模型训练脚本
├── GUI/                        # 桌面交互式算法调试与可视化工具集
│   ├── CV_GUI.py               # 单图交互式机器视觉调参工具 (阈值、滤波、HSV)
│   ├── CV_GUI Dual.py          # 双图对比与基准差分调参工具
│   ├── Dino_threshold_analyzer_GUI.py # DINO 异常距离场与阈值分析工具
│   ├── KNN clustering GUI.py   # KNN / K-Means 颜色空间聚类可视化工具
│   ├── Heat_Map_Create_GUI 3D.py # 基于 PyQt5 + VTK 的 3D 缺陷热力图分析
│   ├── swapyz_xy.py            # 3D 跌落角度与 STL 零件坐标分析工具
│   └── ODBP_rev1.py            # 光学缺陷边界处理 (ODBP) 算法原型
├── utils/                      # 核心业务逻辑与原子算子库
│   ├── config/
│   │   └── config_manager.py   # 单例配置管理器 (加载与类型化 Excel 配置)
│   ├── core/                   # 核心流水线与执行引擎
│   │   ├── pipeline.py         # 主流水线执行器 (权重解析、失效模式遍历、策略路由)
│   │   ├── schema.py           # Pydantic 数据规范与节点 Payload 契约定义
│   │   └── strategies/         # 策略模式具体实现与动态配置
│   │       ├── base_strategy.py            # 抽象策略基类 (解析评分与输出参数)
│   │       ├── strategy_factory.py         # 策略工厂 (根据 FM 与方法动态派发)
│   │       ├── complex_textile_strategy.py # 纺织品复合缺陷检测策略 (长条/按键面自适应)
│   │       ├── complex_bleach_strategy.py  # 漂白/化学侵蚀检测策略 (Prompt+SAM2+Dino)
│   │       ├── crop_strategy.py            # 多实例密集切图与画布标准化策略
│   │       ├── detectron_seg_strategy.py   # Detectron2 分割策略 (如硅胶脱层)
│   │       ├── dino_strategy.py            # DINOv2/v3 特征异常检测策略
│   │       ├── dynamic_routing_strategy.py # 外部 JSON 动态配置路由策略
│   │       ├── filtering_strategy.py       # 纯传统机器视觉链式滤波策略
│   │       ├── general_fm_dispatcher.py    # 通用多部件/多视口分发器
│   │       ├── grounding_dino_strategy.py  # Grounding DINO 目标定位策略
│   │       ├── grounding_sam_strategy.py   # Grounding DINO + SAM2 联合分割策略
│   │       ├── knn_strategy.py             # KNN 颜色异常聚类检测策略
│   │       ├── macbook_light_bleed_strategy.py # MacBook 屏幕漏光检测策略
│   │       ├── object_detection_strategy.py    # 标准目标检测缺陷识别策略
│   │       ├── canvas_prepare_strategy.py  # 画布规范化预处理策略
│   │       └── configs/                    # 声明式策略配置文件 (如 Textile_R692_bubble.json)
│   ├── wrappers/               # 深度学习模型包装层
│   │   ├── detectron_wrapper.py        # Detectron2 模型加载与类别提取包装器
│   │   ├── grounding_dino_wrapper.py   # Grounding DINO HuggingFace 模型包装器
│   │   └── sam2_wrapper.py             # SAM2 (ONNX 与 PyTorch 原生) 推理包装器
│   ├── advanced_operators.py   # 原子级高级视觉算子 (孤立锚点去重、空间惩罚掩码、相对切片)
│   ├── auth.py                 # AppleConnect / Floodgate / Insight 自动化 Token 获取与缓存
│   ├── base_utils.py           # 基础文本规范化工具 (`_norm`)
│   ├── bbox_ops.py             # BBox 计算、IoU 过滤、缺陷比例计算
│   ├── crop_ops.py             # 旋转对齐 (minAreaRect/PCA)、方向校验校正、DUT 抠图
│   ├── cv_ops.py               # 图像叠加渲染、HSV/RGB 空间过滤、轮廓分析
│   ├── detectron_ops.py        # Detectron2 推理与物理缺陷尺寸计算 (宽度/长度分析)
│   ├── dinov3_utils.py         # DINO 权重加载、异常热力图渲染、AnyUp 联合推理
│   ├── filtering_ops.py        # 超大规模图像滤波算子库 (Top-Hat、自适应高斯、ROI 掩膜)
│   ├── general_fm_ops.py       # 图像路径映射匹配与 General FM Scope 过滤
│   ├── io_ops.py               # 文件压缩/解压、系统路径处理、Radar 文件传输
│   ├── knn_utils.py            # 多阶段 KNN 聚类、空间稀疏度（Sparsity）异常检测
│   ├── output_ops.py           # 参数化输出生成、物理标定换算、KMeans 实例排序、Excel 导出
│   ├── scoring_optimizer.py    # 人类打分与算法特征对齐优化器 (Ridge 回归与 Spearman 相关性)
│   ├── scoring_utils.py        # 指数衰减加权、局部对比度评分、形态学分箱
│   ├── shape_ops.py            # 几何形状过滤 (长宽比、圆度) 与药丸形 (Pill) 对称度分析
│   └── utils_general.py        # 共享工具类与全局状态维护
└── Result/                     # 输出结果目录 (包含各批次 Excel 报表与叠加渲染图)
```

---

### 四、 各脚本主要作用与核心功能详解

#### 1. 入口与自动化控制脚本

| 脚本路径 | 主要作用与核心逻辑 |
| :--- | :--- |
| **`RELP3_main.py`** | **系统主运行入口**。<br>1. 动态加载模块搜索路径（`utils/`, `detectron_all/`, `weights/sam2/`）；<br>2. 开启 PyTorch MPS Fallback 模式（适配 Apple Silicon 硬件加速）；<br>3. 实例化单例 `ConfigManager`，读取 `RELP_Configuration.xlsx` 全量配置；<br>4. 实例化并启动 `DefectDetectionPipeline().run()`，触发缺陷检测主流程。 |
| **`RELP_autorun.py`** | **7x24小时全自动守护进程**。<br>1. **macOS 防休眠/防锁屏**：调用系统 `caffeinate -d -i -s`，脚本退出时通过 `atexit` 自动清理；<br>2. **Kerberos & AppleConnect 鉴权保活**：检查 `klist` 票据有效性，过期时通过 `expect` 脚本自动交互输入密码刷新 Kerberos / AppleConnect 票据；<br>3. **Radar 任务轮询**：多线程并发监听 `Auto Run` 配置中的 Radar ID，基于设定的时间窗口（Frequency）和文件后缀自动下载最新的测试图集与 zip 包；<br>4. **加锁分析与回传**：通过线程互斥锁（`analysis_lock`）串行调用分析流水线，将生成的参数化报表与叠加图打包回传上传至指定 Radar，并将已处理文件夹重命名为 `_analyzed` 归档。 |

---

#### 2. 配置与核心框架层 (`utils/config/`, `utils/core/`)

| 脚本路径 | 主要作用与核心逻辑 |
| :--- | :--- |
| **`config_manager.py`** | **单例模式配置管理器**。<br>负责一次性将 `RELP_Configuration.xlsx` 的所有工作表加载入内存，并将其解析为强类型的 Dataclass（如 `FailureModeConfig`、`FlowConfig`、`OutputConfig`），对外提供高效的表单检索接口，避免重复读取磁盘。 |
| **`pipeline.py`** | **流水线核心调度引擎**。<br>1. 解析全局模型权重（SAM2、Detectron2、DINO 等）；<br>2. 遍历所有生效的 Failure Mode，重置缺陷检测上下文；<br>3. 实现**绞杀者模式 (Strangler Fig Router)**，将当前失效模式及其运行上下文移交给 `StrategyFactory` 进行策略分发与执行。 |
| **`schema.py`** | **数据规范与管道契约 (Data Contracts)**。<br>使用 Pydantic 定义节点数据规范（`PipelineContext`, `NodePayload`, `NodeMetadata`, `FilterAdaptiveGaussian`, `FilterKNN` 等），确保在复杂 DAG 流水线各节点间传递的掩码、BBox、类别标签和置信度具备严谨的类型约束与线程安全。 |

---

#### 3. 策略实现层 (`utils/core/strategies/`)

所有策略均继承自 `base_strategy.py` 中的抽象基类 `AnalysisStrategy`，各自负责特定缺陷类型或算法组合的闭环执行：

| 脚本路径 | 针对的失效模式 / 算法组合 | 主要业务与算法逻辑 |
| :--- | :--- | :--- |
| **`base_strategy.py`** | 抽象基类 | 封装通用的上下文解析逻辑：自动解析 `Output` 表中的 Scoring 参数、Scoring Setting、Parametrics 输出列、Mask 标注颜色、以及色彩空间分箱（Color Space Binning）配置。 |
| **`strategy_factory.py`** | 策略路由工厂 | 根据 Excel 中配置的方法名（`Defect Identification`）或特定 `Failure Mode` 进行优先级匹配，动态分发实例化相应的策略对象。 |
| **`complex_textile_strategy.py`** | 纺织品复合缺陷 (`bubble`, `textile`) | 针对织物外观检测设计：<br>1. 通过长宽比动态判断拍摄面（长条 AB 面 vs 带按键 CD 面）；<br>2. Grounding DINO 自动识别 Red Tag 计算真实物理面积缩放比例；<br>3. 在 CD 面执行按键识别与防套娃去重，并在按键周围构建 1.3 倍空间惩罚掩码（高应力区权重大）；<br>4. 执行 DINOv3 + AnyUp 超分辨率特征异常推断并加权计分。 |
| **`complex_bleach_strategy.py`** | 漂白与化学侵蚀 (`hiaa bleach`) | Zero-Shot 组合流：解析如 `GroundingSAM[elliptic camera.,ratio=1.3]-Dino` 语法，先通过 Grounding DINO 定位关键部件，再由 SAM2 抠出亚像素级精细边缘，最后在掩码内部跑 DINO 异常检测或后处理。 |
| **`crop_strategy.py`** | 密集多实例切图 (`crop`, `dino crop`) | 针对托盘摆放的多样机图：Detectron2 检出多目标 BBox $\to$ SAM2 分别抠图涂黑背景 $\to$ Tight Crop 截断黑边 $\to$ 按预定尺寸（如 1280x1280）等比缩放并 Padding 补白，生成多张规整的标准画布图。 |
| **`detectron_seg_strategy.py`** | 硅胶脱层 (`Silicon Delam`) | 针对硅胶保护壳脱层：基于 Detectron2 检出脱层粗候选区 $\to$ SAM2 结合几何交集精细化边缘 $\to$ 提取真实脱层掩码 $\to$ 统计脱层面积比与位置并生成报表。 |
| **`dino_strategy.py`** | 无监督异常检测 (`dino`) | 基于自监督基础模型：DUT 图像对齐预处理 $\to$ 提取 DINO 深度特征并与无缺陷参考特征比对计算距离场 $\to$ 根据 `Dino_TH` 阈值二值化 $\to$ 生成异常热力图与缺陷指标。 |
| **`dynamic_routing_strategy.py`** | 声明式 DAG 动态流水线 | 读取外部 JSON 文件（如 `Textile_R692_bubble.json`），完全脱离硬编码，按节点依赖执行 GroundingDINO、SAM2、Slicing、DINOv3、ParametricOutput 等原子模块。 |
| **`filtering_strategy.py`** | 传统 CV 链式滤波 (`filtering`) | 解析诸如 `Filtering[BG Subtraction]&&Filtering[Color Distance]` 等宏定义，按顺序执行背景差分、颜色阈值、形态学运算等经典 CV 算子。 |
| **`general_fm_dispatcher.py`** | 通用多部件分发器 (`General FM`) | 当单张图片存在多视口（如 top、bottom、B1、B2）时，按文件名模式匹配切分图片集，并挂载各自特定的检测规则后向下派发。 |
| **`grounding_dino_strategy.py`** | 零样本开放词表检测 (`grounding dino`) | 直接解析文本 Prompt（如 `"scratch"`, `"stain"`），调用 Grounding DINO 获取预测框，支持长宽比过滤与 Top-K 候选框保留。 |
| **`grounding_sam_strategy.py`** | 文本定位 + 实例分割 (`grounding sam`) | 将 Grounding DINO 文本检出的 BBox 作为 Prompt 传入 SAM2，输出对应部件的高清像素掩码并进行后续缺陷提取。 |
| **`knn_strategy.py`** | 颜色空间聚类 (`knn`) | 针对织物变色/污渍（Discoloration）：在 RGB 空间对非黑背景像素执行双阶段 K-Means/KNN 聚类与空间稀疏度（Sparsity）过滤。 |
| **`macbook_light_bleed_strategy.py`** | 屏幕漏光检测 (`light bleed`) | 专用屏幕检测：Grounding DINO 定位 `"white screen"` 获得 DUT 掩膜，寻找 `"round object"` 作为物理标定靶标，最后通过 Stencil-ROI 与全局阈值提取漏光缺陷。 |
| **`object_detection_strategy.py`** | 标准目标检测 (`object detection`) | 针对屏幕划痕/凹坑等缺陷，先执行大件定位抠图，再在对齐后的局部裁切图上跑 Detectron2 缺陷检测。 |
| **`canvas_prepare_strategy.py`** | 画布预处理 (`canvas prepare`) | 纯净画布预处理流：原图定位主体 $\to$ SAM2 抠图 $\to$ Tight Crop 切黑边并 Padding 为方形画布，供后续所有检测环节直接使用。 |

---

#### 4. 模型包装与外部集成层 (`utils/wrappers/`, `utils/auth.py`)

| 脚本路径 | 主要作用与核心逻辑 |
| :--- | :--- |
| **`detectron_wrapper.py`** | 封装 Detectron2 的 `DefaultPredictor` 加载流程，并支持自动从配置目录中的 `classes.rtf` 中反序列化并解析各类别名称（Class Names）。 |
| **`grounding_dino_wrapper.py`** | 封装 HuggingFace Transformers 中的 Grounding DINO 模型，对外提供类 Detectron 的统一 `.predict(image, text_prompt)` 接口，内置 BBox 与 Text 阈值过滤。 |
| **`sam2_wrapper.py`** | SAM 2.1 推理引擎封装：<br>1. `SAM2OnnxPredictor`：通过 ONNX Runtime 分离加载 Encoder 和 Decoder，在 CPU/MPS 环境下高效执行点/框 Prompt 掩码生成；<br>2. PyTorch 原生 `SAM2ImagePredictor` 兼容接口。 |
| **`auth.py`** | **内部统一认证助手**：通过本地 `appleconnect` CLI 获取 Floodgate OAuth Token 与 Insight Token，并带有效时长缓存至磁盘（`token_cache.json`），避免频繁弹窗或超时。 |

---

#### 5. 核心算子与工具库 (`utils/*_ops.py`, `utils/*_utils.py`)

| 脚本路径 | 模块规模 | 主要职责与核心算法 |
| :--- | :--- | :--- |
| **`filtering_ops.py`** | 4200+ 行 | **传统图像处理过滤总调度与算法库**：<br>- `apply_filtering`：链式过滤调度入口；<br>- `apply_adaptive_gaussian`：自适应高斯局部阈值；<br>- `apply_morph_tophat`：形态学 Top-Hat（提取亮斑）与 Black-Hat（提取暗斑）；<br>- `apply_area_filtering`：按物理面积百分比或像素数进行连通域滤除；<br>- `apply_bg_subtraction_filtering` / `apply_odbp_filtering`：背景差分与光学边缘滤波；<br>- `apply_band_roi_filtering` / `apply_stencil_roi_filtering`：几何条带及模版遮罩过滤。 |
| **`output_ops.py`** | 3400+ 行 | **量规参数化计算与报告生成**：<br>- `calculate_parametric_dimensions`：根据参考物标定计算缺陷的实际物理面积（$mm^2$）、长宽、弧长、圆度等；<br>- `build_parametric_output_df` / `parametric_output`：整理并导出符合生产标准的 `Parametric_Output.xlsx`；<br>- `clustering_Kmeans`：通过 K-Means 对图像中的多个产品实例按物理坐标空间排序（如 Instance 1 到 10）；<br>- `resolve_sn_from_excel`：根据文件路径与配置表自动解析产品序列号（SN）。 |
| **`knn_utils.py`** | 2200+ 行 | **颜色聚类与空间稀疏度分析**：<br>- `apply_knn_clustering_with_selection`：在三维色彩空间进行聚类分离目标色块；<br>- `create_sparsity_detection_map` / `detect_sparsity_anomalies`：通过高斯核密度估计计算像素空间稀疏度，检测织物脱丝、稀疏缺陷；<br>- `detect_contour_holes`：检测轮廓内部的空洞缺陷。 |
| **`crop_ops.py`** | 900+ 行 | **几何校正与主体裁切**：<br>- `perform_dut_alignment`：结合 Detectron2 与 SAM2 提取 DUT 掩码，计算 `minAreaRect` 最小外接矩形的主轴旋转角，自动旋转矫正；<br>- `check_orientation_and_flip`：基于特定特征的上下位置关系判断样件是否倒置并自动水平/垂直翻转；<br>- `remove_defects`：在背景提取阶段预先掩膜已知零部件，防止对主体边缘产生误检。 |
| **`dinov3_utils.py`** | 600+ 行 | **DINO 特征与异常热力图工具**：<br>- `cal_anomaly_maps` / `cal_anomaly_maps_anyup`：对比测试图与基准特征的余弦/欧氏距离；<br>- `load_anyup_upsampler`：接入 AnyUp 模块进行特征图高分辨率重建；<br>- `cvt2heatmap` / `show_cam_on_image_transparent`：渲染平滑半透明伪彩色热力图。 |
| **`cv_ops.py`** | 1600+ 行 | **通用视觉与图层渲染**：<br>- `create_overlay_image` / `save_inferred_pic_overlay`：在原图上叠印半透明缺陷色块与轮廓线；<br>- `extract_hsv_config_from_filtering` / `apply_hsv_filtering`：HSV 色彩空间阈值过滤；<br>- `mask_based_matting` / `contour_based_matting`：基于掩码的抠图与背景羽化。 |
| **`bbox_ops.py`** | 370+ 行 | **边界框几何运算**：<br>- `calculate_bbox_iou`：计算 BBox IoU 交并比；<br>- `filter_mask_by_bbox_iou`：根据缺陷连通域落在指定目标框内的面积占比进行保留或滤除；<br>- `create_bbox_mask`：将边界框坐标快速生成二进制掩码。 |
| **`advanced_operators.py`** | 230+ 行 | **高阶原子算子**：<br>- `op_isolated_anchor_filter`：孤岛锚点重叠剔除算法，解决零样本模型在反光、倒角处的“套娃框”与“幽灵框”；<br>- `op_create_spatial_penalty_mask`：生成围绕关键结构（按键、接口）外扩 1.3 倍的空间加权惩罚热区；<br>- `op_relative_slicing`：基于物理边界框比例相对切分有效检测区（如取下半区 Y[0.45:1.0]）。 |
| **`scoring_utils.py`** | 700+ 行 | **质量严重度评分计算**：<br>- `exponential_decay_weights`：生成指数衰减权重向量；<br>- `calculate_complement_score`：对不同特征分箱计算综合缺陷评分；<br>- `calculate_local_contrast_coefficient` / `morph_contrast_binning`：基于形态学与局部对比度的严重度分级。 |
| **`scoring_optimizer.py`** | 130+ 行 | **人类打分对齐引擎 (Human-AI Alignment)**：<br>使用 Ridge 岭回归与 Spearman 秩相关系数，将人工复检等级（Grade A/B/C/D 与 Subgrade）与算法输出的 Parametric 连续特征进行拟合，自动搜索出最优特征权重公式。 |
| **`shape_ops.py`** | 80+ 行 | **轮廓形状学特征量测**：<br>- `apply_shape_filtering`：基于长宽比和圆度阈值滤除杂质；<br>- `check_pill_symmetry`：通过主轴旋转及左右/上下翻转对比，量化药丸形零件（如按键、摄像孔）的对称度。 |
| **`general_fm_ops.py`** | 110+ 行 | **路径映射与作用域截取**：根据图像路径规则匹配部件，并利用 Detectron 检测部件范围，限制缺陷分析作用域（Scope Filtering）。 |
| **`io_ops.py`** | 300+ 行 | **文件与 Radar 传输 I/O**：压缩/解压 zip、跨平台清理 `__MACOSX` 隐藏垃圾文件、路径级联创建等。 |
| **`base_utils.py`** | 5 行 | 极简工具：提供全局通用的字符串正则清洗函数 `_norm()`（移除所有非字母数字并小写化）。 |

---

#### 6. 视觉特征超分模块 (`anyup-main/`)

`anyup-main` 是该项目内嵌的一个前沿视觉特征上采样（Feature Super-Resolution）算法库：
- **`anyup/model.py`**：定义 `AnyUp` 深度神经网络。由于 ViT 和 DINO 模型的 Patch 特征分辨率较低（如 $14 \times 14$ 步长），AnyUp 采用 Cross-Attention 机制，将原始高分辨率 RGB 图像作为 Guidance，引导低分辨率深层语义特征图实现高清上采样。
- **`anyup/layers/attention/`**：包含常规 Cross-Attention、分块注意力（Chunked Attention）以及邻域注意力（NATTEN, Neighborhood Attention）。
- **`anyup/hubconf.py`**：提供 `torch.hub` 兼容的模型构建与预训练权重加载函数（`anyup()`、`anyup_multi_backbone()`）。
- **`train.py`**：AnyUp 模型的训练与微调脚本，基于 Hydra 配置管理，使用余弦相似度与 MSE 复合损失函数。

---

#### 7. 算法调试与可视化 GUI 工具套件 (`GUI/`)

该目录包含多个供算法与可靠性测试工程师在产线部署前独立使用的桌面级调参软件：

| 脚本路径 | 技术栈 | 工具名称与主要作用 |
| :--- | :--- | :--- |
| **`CV_GUI.py`** | Tkinter + OpenCV | **CV 综合调参台**：支持加载样件大图，实时拖动滑块测试全局二值化、自适应高斯、背景差分、ODBP、HSV 颜色过滤以及形态学 Top-Hat/Black-Hat，直观预览缺陷分割效果，用于向 Excel 配置文件回填最佳参数。 |
| **`CV_GUI Dual.py`** | Tkinter + OpenCV | **双图对比调参台**：支持并排加载两张图片（如测试前 vs 测试后，良品 vs 不良品），同步进行差分与阈值分割比对。 |
| **`Dino_threshold_analyzer_GUI.py`** | Tkinter + Matplotlib | **DINO 异常距离分析器**：加载 DINO 模型导出的异常距离 CSV 矩阵，交互式调节 Min/Max 归一化区间与二值化截断阈值，实时在底图上叠加伪彩色热力图。 |
| **`KNN clustering GUI.py`** | Tkinter + Scikit-Learn | **KNN 颜色聚类分析器**：选取图片中的织物变色区域，交互式设定聚类中心数 $K$ 与目标 RGB 距离阈值，可视化观察聚类连通域。 |
| **`Heat_Map_Create_GUI 3D.py`** | PyQt5 + VTK | **3D 缺陷热力图分析系统**：通过 VTK 渲染窗口，读取多批次样品的缺陷三维空间坐标（Excel/NPY），将缺陷发生频次投影渲染在产品 3D 几何表面上。 |
| **`swapyz_xy.py`** | PyQt5 + VTK | **3D 跌落角度与 CAD 零件坐标分析工具**：加载 3D 零件 STL 文件，提供模型旋转对齐、坐标轴对调（Swap YZ/XY）与跌落冲击方位角分析功能。 |
| **`ODBP_rev1.py`** | OpenCV + Rembg | **光学缺陷边界处理原型**：基于多尺度边缘模板匹配与背景抠除的几何定位历史版本代码。 |

---

### 五、 总结

`/Users/an/workspace/RELP` 是一个高度工程化、模块清晰且紧密结合硬件可靠性实际工况的视觉检测系统：
1. **输入与配置**：完全由 `RELP_Configuration.xlsx` 控制，支持灵活的热拔插与多产品/多代际适配；
2. **核心执行**：以 `pipeline.py` 为骨架，通过 `StrategyFactory` 调度各个独立的 `AnalysisStrategy`（覆盖了自监督 DINO、零样本 Grounding DINO/SAM2、目标检测 Detectron2 及经典 CV 算子链）；
3. **输出与闭环**：通过 `output_ops.py` 实现毫米级物理量规测量，配合 `RELP_autorun.py` 实现从 Radar 任务下载到报表回传的无人值守闭环运行。