# RELP (Reliability Engineering Learning Platform) — Agent 指南与架构白皮书

本文档专为在此代码仓库中协同开发、重构分析与维护的 AI Agent 及工程师编写。深入阐明了 RELP 的系统定位、核心视觉技术栈、工程目录结构、设计模式演进、关键架构陷阱防范以及开发维护规范。

---

## 1. 项目定位与核心使命

**RELP (Reliability Engineering Learning Platform)** 是用于硬件产品可靠性工程（Reliability Engineering）的高精度工业视觉缺陷检测、几何计量与量化打分平台。

* **核心业务领域**：覆盖消费电子硬件（包括 Silicon Case 硅胶壳、Textile Case 纺织保护壳、MacBook 机身与屏幕、Bumper Case 边框、Watch Band 表带、Matte Screen 磨砂屏等）在各项可靠性应力测试（如耐磨、耐腐蚀、抗拉伸、抗冲击、防漏光、脱层等）后的瑕疵分析。
* **主要缺陷类型**：
  * **结构与物理破损**：脱层（Delamination）、磨损（Wear/Scratch）、起毛（Fraying）、凹坑（Pit/Dent）、气泡（Bubble）。
  * **光学与表面缺陷**：屏幕漏光（Light Bleed）、漂白/变色（Bleach/Discoloration/Staining）、瑕疵杂质（Debris/Particle）。
* **全闭环流程**：
  `Radar 任务监控/下载` $\rightarrow$ `图像预处理与 DUT 提取对齐` $\rightarrow$ `深度学习/经典算子缺陷检测与分割` $\rightarrow$ `物理尺寸标定与衰减打分` $\rightarrow$ `参数化报表导出 (Excel)` $\rightarrow$ `可视化结果渲染与回传 Radar`。

---

## 2. 技术栈与环境依赖

RELP 深度融合了经典机器视觉与前沿零样本（Zero-Shot）视觉大模型技术，并对执行环境实施严格的版本锁定。

### 2.1 依赖栈与版本要求
* **Python 版本**：`>=3.12, <3.13`（由 `uv` 统一锁定环境，严禁自动大版本升级）。
* **核心框架**：
  * **深度学习与多模态**：`PyTorch >= 2.14.0`, `torchvision >= 0.29.0`, `Detectron2` (FacebookResearch 主干构建), `SAM 2` (Segment Anything Model 2), `Grounding DINO` (Hugging Face Transformers), `AnyUp` (ICLR '26 Oral 通用特征上采样器), `ONNX Runtime`。
  * **图像处理与几何计算**：`OpenCV (opencv-python >= 5.0)`, `scikit-image`, `scikit-learn`, `Pillow`, `pillow-heif` (支持 iPhone HEIC 格式)。
  * **配置与数据流**：`pandas >= 3.0.5`, `openpyxl >= 3.1.5`, `Pydantic >= 2.13.5` (强类型数据校验)。
  * **服务与后端**：`FastAPI >= 0.141.1`, `uvicorn`, `SQLAlchemy >= 2.0.54`, `SQLite3`。
  * **内部工具集成**：`radarclient` (Apple Radar API 客户端), `appleconnect` / `kinit` (Kerberos 身份鉴权)。

### 2.2 硬件与加速策略
* **macOS Apple Silicon (MPS)**：
  全局默认启用 `PYTORCH_ENABLE_MPS_FALLBACK="1"`，当部分高阶张量插值算子（如 `upsample_bicubic2d`）在 MPS 上未实现时自动平滑回退至 CPU。
* **SAM 2 双模引擎**：
  在 `utils/wrappers/sam2_wrapper.py` 中同时提供原生 PyTorch 预测器和针对 CPU/MPS 稳定运行的 `SAM2OnnxPredictor`。
* **进程保活机制**：
  在 macOS 下启动 `caffeinate -d -i -s` 后台守护，防止执行超长批处理任务时系统空闲休眠或锁屏断网。

---

## 3. 完整代码仓库文件与模块结构速查

```
RELP/
├── pyproject.toml              # uv 项目配置文件及依赖约束
├── uv.lock                     # uv 严格依赖锁定文件
├── RELP3_main.py               # 命令行与流水线主执行入口
├── RELP_autorun.py             # Radar 自动化监听、下载、分析、回传常驻守护脚本
├── RELP_Configuration.xlsx     # 业务规则与模型参数的核心 Excel 数据库
├── AGENTS.md                   # Agent 专属架构指导与规范手册 (本文档)
│
├── utils/                      # RELP 核心算法与底层支撑库
│   ├── RELP3_main.py           # 内部流水线入口副本/历史入口
│   ├── __init__.py
│   ├── auth.py                 # AppleConnect / OAuth Token 缓存与提取
│   ├── base_utils.py           # 基础通用工具 (_norm 字符串正则化)
│   ├── bbox_ops.py             # 边界框几何计算 (IoU, NMS, 排序, 缩放, 离群抑制)
│   ├── crop_ops.py             # 物体对齐、旋转纠偏、紧致切图 (Tight Crop) 与 Padding
│   ├── cv_ops.py               # 底层图像矩阵操作 (颜色空间变换, 掩膜合成, 叠加层渲染, 连通域)
│   ├── detectron_ops.py        # Detectron2 目标检测解析、掩膜提取与缺陷几何尺寸度量
│   ├── dinov3_utils.py         # DINO 异常热力图生成、AnyUp 高清特征上采样、透明渲染
│   ├── filtering_ops.py        # 经典机器视觉算子链 (自适应高斯、形态学顶帽、ROI、背景差分等)
│   ├── general_fm_ops.py       # 多工位图片路径正则匹配与检测范围映射 (Scope)
│   ├── io_ops.py               # 文件解压缩、目录创建、路径解析、Radar 传输工具
│   ├── knn_utils.py            # K-Means / KNN 颜色聚类分析与空间稀疏度异常检测
│   ├── output_ops.py           # 参数化输出构建、真实尺寸标定、分箱统计、Excel 导出
│   ├── scoring_optimizer.py    # 针对人眼评级真值的缺陷打分权重自动拟合优化器
│   ├── scoring_utils.py        # 缺陷加权衰减打分、局部对比度度量与互补分计算
│   ├── shape_ops.py            # 几何形状过滤器 (长宽比、圆度、胶囊药丸对称度)
│   ├── utils_general.py        # 全局上下文变量池与历史公共函数库
│   │
│   ├── config/                 # 配置管理层
│   │   └── config_manager.py   # ConfigManager 单例：Excel 缓存与强类型 Dataclass 解析
│   │
│   ├── core/                   # 流水线与调度内核
│   │   ├── pipeline.py         # 核心调度执行器 (DefectDetectionPipeline / run())
│   │   ├── schema.py           # Pydantic 数据规范与 DAG 节点输入输出标准契约
│   │   └── strategies/         # 现代化分析策略库 (Strategy Pattern)
│   │       ├── base_strategy.py              # 策略基类 (AnalysisStrategy)
│   │       ├── strategy_factory.py           # 策略工厂 (StrategyFactory 核心路由)
│   │       ├── detectron_seg_strategy.py     # Detectron2 + SAM2 硅胶脱层专用策略
│   │       ├── dino_strategy.py              # DINO 基础特征距离异常检测策略
│   │       ├── dynamic_routing_strategy.py   # 基于外部 JSON 的 DAG 动态路由策略
│   │       ├── complex_textile_strategy.py   # 纺织保护壳专用复合策略 (长宽比/按键惩罚/DINOv3)
│   │       ├── complex_bleach_strategy.py    # Zero-Shot 漂白缺陷策略 (GDino + SAM2 + PostOp)
│   │       ├── macbook_light_bleed_strategy.py # MacBook 屏幕漏光专用检测与标定策略
│   │       ├── crop_strategy.py              # 多实例目标检测切图与标准画布生成策略
│   │       ├── canvas_prepare_strategy.py    # 标准画布制备与缺陷检测串联策略
│   │       ├── object_detection_strategy.py  # 纯目标检测与 BBox 可视化策略
│   │       ├── grounding_dino_strategy.py    # 纯自然语言 Prompt 驱动的目标检测策略
│   │       ├── grounding_sam_strategy.py     # Grounding DINO + SAM2 实例分割泛化策略
│   │       ├── knn_strategy.py               # 颜色空间聚类与色差变色专用策略
│   │       ├── filtering_strategy.py         # 传统数字图像处理过滤链执行策略
│   │       ├── general_fm_dispatcher.py      # 通用工位/多相机拍摄路径分发器
│   │       └── configs/                      # 策略专用的外部动态 JSON 配置文件
│   │           └── Textile_R692_bubble.json
│   │
│   └── wrappers/               # AI 视觉模型轻量级统一封装层
│       ├── detectron_wrapper.py      # Detectron2 模型载入与 classes.rtf 类别映射
│       ├── sam2_wrapper.py           # SAM 2 (PyTorch / ONNX) 预测器封装
│       └── grounding_dino_wrapper.py # Grounding DINO HuggingFace 模型封装
│
├── conf_gui/                   # 现代化可视化配置与规则校验平台
│   ├── run.py                  # Web 服务启动入口 (Uvicorn + FastAPI)
│   ├── be_carful.md            # 底层配置陷阱与源码比对核心避坑指南 (必读)
│   ├── test_integration.py     # 配置中心数据库 CRUD、校验与导出的集成测试
│   ├── relp_config.db          # SQLite 本地配置数据库
│   ├── backups/                # Excel 导出与同步时的历史时间戳自动备份
│   ├── api/                    # FastAPI 路由与控制器
│   │   ├── main.py             # Web 应用入口、静态文件挂载与跨域配置
│   │   └── routes.py           # 12 张配置表的 RESTful CRUD 与同步导出 API
│   ├── core/                   # 配置平台业务核心
│   │   ├── database.py         # SQLAlchemy Session 与 Engine
│   │   ├── models.py           # 12 张配置表的 SQLAlchemy ORM 数据模型
│   │   ├── excel_bridge.py     # Excel $\leftrightarrow$ SQLite 双向高保真同步引擎
│   │   └── validator.py        # 工业级输入防呆校验规则引擎
│   └── web/                    # 前端单页面应用 (Vanilla JS + CSS)
│       ├── index.html
│       ├── css/style.css
│       └── js/app.js
│
├── anyup-main/                 # [ICLR '26 Oral] AnyUp 框架子模块 (特征上采样支持)
│   ├── anyup/                  # AnyUp 核心模型定义 (ViT 包装器、注意力掩膜、卷积层)
│   ├── config/                 # 训练与评估配置
│   └── hubconf.py              # Torch Hub 接入定义
│
└── GUI/                        # 桌面交互式视觉调参工具箱 (Tkinter 工具集)
    ├── CV_GUI.py               # 交互式阈值分割、形态学与自适应滤波调试器
    ├── CV_GUI Dual.py          # 双图对比与算子同步调参工具
    ├── KNN clustering GUI.py   # 颜色聚类与空间稀疏度参数调试工具
    ├── Dino_threshold_analyzer_GUI.py # DINO 异常阈值切分与热力图调优工具
    ├── ODBP_rev1.py            # 工业光学缺陷亮度处理调试工具
    └── Heat_Map_Create_GUI.py  # 2D/3D 热力图渲染与生成器
```

---

## 4. 核心系统架构与设计模式

RELP 经历了从“面向过程的万行单体大脚本”到“高度模块化、配置驱动、松耦合策略架构”的演进。

```
                    ┌─────────────────────────┐
                    │  RELP_Configuration.xlsx│
                    └────────────┬────────────┘
                                 │ load_excel()
                                 ▼
                     ┌───────────────────────┐
                     │ ConfigManager (单例)   │
                     └───────────┬───────────┘
                                 │
     命令行/Autorun               ▼
┌───────────────────────┐   ┌────────────────────────────────┐
│ RELP3_main.py         ├──▶│ DefectDetectionPipeline.run()   │
│ (--product, --fm...)  │   └───────────────┬────────────────┘
└───────────────────────┘                   │
                                            ▼
                               ┌────────────────────────┐
                               │ StrategyFactory        │ (绞杀者模式路由器)
                               └────────────┬───────────┘
                                            │
        ┌───────────────────────────────────┼──────────────────────────────────┐
        ▼                                   ▼                                  ▼
┌───────────────────────┐       ┌───────────────────────┐          ┌───────────────────────┐
│ DetectronSegStrategy  │       │ ComplexTextileStrategy│          │ DynamicRoutingStrategy│
│ (Detectron2 + SAM2)   │       │ (DINOv3 + AnyUp + 惩罚)│          │ (JSON DAG 驱动架构)   │
└───────────┬───────────┘       └───────────┬───────────┘          └───────────┬───────────┘
            │                               │                                  │
            └───────────────────────────────┼──────────────────────────────────┘
                                            ▼
                              ┌───────────────────────────┐
                              │ Parametric Output/Scoring │ (物理标定与衰减打分)
                              └─────────────┬─────────────┘
                                            ▼
                              ┌───────────────────────────┐
                              │ Result/Parametric_Output  │
                              └───────────────────────────┘
```

### 4.1 绞杀者模式 (Strangler Fig Pattern)
* **设计意图**：旧版系统中存在大量跨度数千行的条件判断与全局变量篡改。新架构采用**绞杀者模式**，在 `utils/core/pipeline.py` 中构建分发路由，将解析出的 `Failure Mode` 与 `Defect Identification` 转交由 `StrategyFactory` 派发。
* **隔离与回退**：每个算法方案均继承自 `AnalysisStrategy` 基类，独立封闭其模型加载、几何变换与后处理逻辑，互不污染，任何单个策略的重构不会影响其他产品线的正常运行。

### 4.2 策略工厂与多重路由策略
`StrategyFactory.get_strategy(method_name, fm_name, context)` 严格遵循四层优先级路由：
1. **工位分发拦截**：若在 `General FM` 表中存在该 Failure Mode 的多相机/多工位映射且未标记执行，优先派发给 `GeneralFmDispatcher` 拆分任务。
2. **专属缺陷精准路由**：
   * `'hiaa bleach'` $\rightarrow$ `ComplexBleachStrategy`
   * `'light bleed'` $\rightarrow$ `MacbookLightBleedStrategy`
   * `'canvas prepare'` / `'k11p_defects'` $\rightarrow$ `CanvasPrepareStrategy`
   * `'bubble'` / `'textile'` $\rightarrow$ `ComplexTextileStrategy`
   * `'crop*'` / `'dino crop*'` / `'groundingcrop*'` $\rightarrow$ `CropStrategy`
   * `'mistral_screen_defect'` $\rightarrow$ `ObjectDetectionStrategy`
3. **算法方法泛化路由**：
   * `'groundingdino'` $\rightarrow$ `GroundingDinoStrategy`
   * `'groundingsam'` $\rightarrow$ `GroundingSamStrategy`
   * `'knn'` $\rightarrow$ `KnnStrategy`
   * `'dino'` $\rightarrow$ `DinoStrategy`
   * `'detectron_seg'` / `'silicon delam'` $\rightarrow$ `DetectronSegStrategy`
   * `'filtering'` $\rightarrow$ `FilteringStrategy`
4. **外部配置驱动**：无法命中的复杂模式回退至读取 JSON DAG 的 `DynamicRoutingStrategy`。

### 4.3 强类型规范与 DAG 数据流契约 (`schema.py`)
为实现工业视觉流水线的组件化拼接，RELP 在 `utils/core/schema.py` 中确立了节点间的数据交换契约：
* **`NodeMetadata`**：携带隐藏通道的高级几何与语义信息，包含 `boxes` (归一化或绝对坐标框)、`instance_masks` (各实例二进制掩膜)、`labels` (语义标签列表) 与 `scores` (置信度得分)。
* **`NodePayload`**：标准节点输出载荷，包含主像素级掩膜 `mask` (numpy ndarray) 与元数据 `metadata: NodeMetadata`。下游节点必须基于此契约消费数据，严禁通过修改全局状态传递私有数据。

### 4.4 闭环自动化流水线 (`RELP_autorun.py`)
`RELP_autorun.py` 是部署于生产测试机房的核心守护引擎：
1. **多线程并发轮询**：解析 `RELP_Configuration.xlsx` 的 `Auto Run` 表，为每个被监控的 Radar ID 生成独立的常驻守护线程。
2. **鉴权安全防护**：每次拉取前调用 `check_and_renew_kerberos_ticket()`，优先使用 `/usr/local/bin/appleconnect`，必要时自动回退至 `kinit` 交互，确保长期运行不掉线。
3. **时间窗口与防重复机制**：利用 `Frequency(hr)` 计算增量，比对本地持久化 `download_history.txt`，跳过已处理附件；自动解压并清洗 `__MACOSX` 干扰包。
4. **互斥锁与动态映射调度**：使用 `analysis_lock = threading.Lock()` 保证模型推理显存不被并发打爆；支持 `Multiple FM Mapping`（如 `[Grade B:Minor_Wear, Grade B:Moderate_Wear]`），按照文件名特征自动将图片切分至临时目录，分别触发针对性的 `RELP3_main.py` 分析。
5. **产物回写**：捕捉标准输出中的 `RELP_RESULT_FOLDER:`，将 `Parametric_Output.xlsx` 及推断图打成 Zip 包，通过 `radar.new_attachment` 自动回传至指派的 Uploading Radar。

### 4.5 可视化配置管理平台 (`conf_gui`)
为消除人工手改 Excel 易发生的格式错误与编码损坏，构建了全功能 Web 配置中心：
* **数据模型层 (`conf_gui/core/models.py`)**：映射全部 12 张工作表（`Auto Run`, `Failure Mode`, `Flow`, `Weights`, `General FM`, `Filtering`, `Output`, `Adaptive Gaussian`, `KNN`, `Dino_TH`, `Scaling`, `Fine_Tuning`, `Reference`）。
* **校验引擎 (`conf_gui/core/validator.py`)**：提供全量前置防呆校验（如检查高斯滤波核 Block Size 是否为奇数、RGB 颜色数组分量是否严格在 0~255 之间、关键主键是否缺失）。
* **双向高保真桥接 (`conf_gui/core/excel_bridge.py`)**：支持从 SQLite 一键同步回写 `RELP_Configuration.xlsx`，回写前自动在 `conf_gui/backups/` 生成带毫秒时间戳的备份。

---

## 5. 关键架构陷阱与操作红线 (Critical Operational Rules)

> ⚠️ **所有 AI Agent 和开发者在修改配置或调用脚本前，必须严格遵守以下 5 条红线：**

### 红线 1：正式表与用例库表的严格隔离
* `RELP_Configuration.xlsx` 中的工作表 `Failure Mode (2)` 和 `Flow (2)` 是**“测试用例与历史实验库”**，沉淀了各硬件形态历史测试用例。
* 系统底层（`ConfigManager.load_excel()` 和 `pipeline.py`）**只读取表名为 `'Failure Mode'` 和 `'Flow'` 的工作表**。
* **严禁用 `Flow (2)` 整体覆盖 `Flow`**：正式 `Flow` 表是经过针对性微调并通过生产验证的参数集。备用表存在大量历史或未验证的实验参数（例如 `Fraying` 的上采样分辨率正式表为 `448` 而备用表为 `160`，`Textile Discoloration` 的 PCA 对齐开关完全相反）。整体覆盖将导致生产算法异常。正确做法是：**仅在正式表中缺失某用例时，针对性单行同步**。

### 红线 2：正式 `Failure Mode` 表保持“单用例激活”原则
* 正式 `Failure Mode` 表在标准生产运行状态下，**建议下方仅保留当前待运行的那 1 行配置**。
* 若表中堆积了多个产品/代际的任务，直接运行 `RELP3_main.py` 会遍历执行所有任务；更严重的是，容易触发下述“红线 3”的同名上下文覆盖问题。

### 红线 3：警惕同名 `Failure Mode` 的全局上下文覆盖 Bug
* **陷阱成因**：在 `utils/core/pipeline.py`（约第 485 行）构造策略上下文时，代码逻辑存在以下历史盲区：
  ```python
  # 即使外部传入了 --product 过滤出了正确的某一行 fm_row:
  # 代码重新在未过滤的全局 DataFrame 中按 Failure Mode 检索并盲目取 iloc[0]!
  fm_row_for_context = fm_df[fm_df[fm_col_name] == fm].iloc[0] if not fm_df[fm_df[fm_col_name] == fm].empty else None
  strat_product = str(fm_row_for_context.get('Product', ''))
  strat_pic_path = str(fm_row_for_context.get('Pic_Path', ''))
  ```
  如果 `Failure Mode` 表中同时存在两条名为 `Crop` 的配置（一条属于 `Silicon_Case`，另一条属于 `Macbook`），执行 `Macbook` 任务时，其图片路径和产品名将被强制覆盖为第 0 行的 `Silicon_Case`！
* **Strategy 内部同质覆盖**：部分旧策略实现类内部依然存在重新向 `ConfigManager` 读取 `Failure Mode` 表并取 `iloc[0]` 的行为。
* **防护规范**：
  1. **日常运行**：正式表保持单任务单行；
  2. **命名防御**：在 `Failure Mode (2)` 中为不同形态赋予唯一名称（如 `Crop_Macbook Pit`、`Dino Crop_Macbook Pit`）；
  3. **代码演进原则**：新写的策略类中，必须无条件使用 `self.context['product']` 和 `self.context['download_path']`，严禁在策略内部通过 `fm_df[fm_df['Failure Mode'] == self.fm].iloc[0]` 覆盖自身状态。

### 红线 4：画布规整化与绝对坐标保护 (Canvas First)
* 在进行复杂缺陷识别时，严禁使用直接将原图拉伸变形送入网络、事后再按比例缩放回原图的粗暴方案。
* 必须遵循标准流程（如 `crop_ops.py` / `CanvasPrepareStrategy` 所示）：
  1. 检测 DUT 外轮廓与主体；
  2. 执行 `Tight Crop` 裁切多余黑边；
  3. 以长边为基准，等比缩放并四边居中填补 Padding，生成标准的正方形画布（Canvas）；
  4. 所有的特征提取、细分切片和缺陷分割均在标准画布坐标系下执行。

### 红线 5：显存与垃圾回收管理
* 流水线大量加载并使用了基础视觉大模型（DINOv3、SAM2、Detectron2、AnyUp）。
* 在 Strategy 循环处理大批量高分辨率工业图像（如 4K/8K 图像）时，必须在每张图或每次推理结束时显式调用：
  ```python
  del intermediate_tensor
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
      torch.mps.empty_cache()
  gc.collect()
  ```

---

## 6. 开发者与 Agent 常用工作流命令

### 6.1 环境构建与依赖同步
```bash
# 进入仓库根目录
cd /Users/an/workspace/RELP

# 验证当前 Python 版本 (必须为 3.12.x)
python3 --version

# 使用 uv 同步依赖
uv sync
```

### 6.2 启动主流水线 (CLI 执行)
```bash
# 1. 默认执行模式 (读取 RELP_Configuration.xlsx 中 Failure Mode 表当前激活的任务)
uv run python RELP3_main.py

# 2. 命令行精确过滤执行 (推荐，精准命中单任务)
uv run python RELP3_main.py --product "Silicon_Case" --generation "2025" --failure_mode "Silicon Delam"

# 3. 覆盖指定图片输入路径执行
uv run python RELP3_main.py --pic_path "/path/to/custom_images" --failure_mode "Textile R692 bubble"
```

### 6.3 启动可视化配置管理服务
```bash
# 启动 Web 配置管理平台 (默认监听 127.0.0.1:8000)
uv run python conf_gui/run.py --open

# 仅执行配置平台集成测试与数据校验
uv run python conf_gui/test_integration.py
```

### 6.4 启动 Radar 自动化监控守护进程
```bash
# 启动守护进程 (前台挂载，自动防休眠，Ctrl+C 优雅退出)
uv run python RELP_autorun.py
```

### 6.5 启动桌面交互式调试调参工具
```bash
# 启动阈值与形态学调试器
uv run python GUI/CV_GUI.py

# 启动 DINO 异常阈值与透明度热力图分析器
uv run python GUI/Dino_threshold_analyzer_GUI.py

# 启动 KNN 颜色聚类调试器
uv run python "GUI/KNN clustering GUI.py"
```

---

## 7. 策略扩展开发指南 (Adding a New Strategy)

当需要为新的产品或缺陷类型开发独立的算法流时，严禁向 `pipeline.py` 中堆叠过程式代码，必须按照以下标准步骤进行策略模式扩展：

### 步骤 1：在 `utils/core/strategies/` 创建新策略文件
新建 `my_new_defect_strategy.py`，继承 `AnalysisStrategy`：
```python
import os
import cv2
import numpy as np
from typing import Dict, Any
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
import utils.utils_general as ug
from utils import cv_ops, output_ops

class MyNewDefectStrategy(AnalysisStrategy):
    """
    针对某新型缺陷的定制分析策略
    """
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context)

    def execute(self) -> None:
        print(f">>> [Strategy] Executing MyNewDefectStrategy for FM: {self.fm}")
        
        # 1. 从上下文获取标准参数 (严禁在策略内部通过 fm_df 盲目 iloc[0])
        product = self.product
        generation = self.generation
        pic_path = self.download_path
        
        # 2. 载入模型权重 (利用 ConfigManager)
        cfg_mgr = ConfigManager()
        weights_df = cfg_mgr.get_sheet('Weights')
        # ...执行具体的模型前向与图像处理逻辑...
        
        # 3. 结果参数化计算与导出
        # 调用 output_ops.build_parametric_output_df(...) 导出 Parametric_Output.xlsx
```

### 步骤 2：在 `StrategyFactory` 中注册路由
编辑 `utils/core/strategies/strategy_factory.py`：
```python
# 导入并根据 Defect Identification 或 Failure Mode 添加分发逻辑
if 'my_new_method' in method_clean or fm_clean == 'my_new_defect':
    from .my_new_defect_strategy import MyNewDefectStrategy
    return MyNewDefectStrategy(context)
```

### 步骤 3：在 `RELP_Configuration.xlsx` 中配置生效
在 `Failure Mode` 表中添加对应用例行，并在 `Flow` 表中将其 `Defect Identification` 字段指定为 `'my_new_method'`。通过 `conf_gui` 运行 `test_integration.py` 验证配置合法性。

---

## 8. 总结与开发守则总结

| 关注维度 | 核心守则 |
| :--- | :--- |
| **环境与版本** | 锁定 Python 3.12，使用 `uv` 管理依赖，禁止静默更新依赖版本。 |
| **代码架构** | 全面遵循策略模式 (Strategy Pattern)，业务逻辑封装于 Strategy 类，原子视觉操作沉淀在 `*_ops.py`。 |
| **数据契约** | 新增流水线节点间通信严格遵从 `schema.py` 中的 `NodePayload` 与 `NodeMetadata`。 |
| **Excel 规则** | `Failure Mode` 生产表仅留单行；`Flow (2)` 严禁全量覆盖 `Flow`；谨防 `iloc[0]` 上下文覆盖。 |
| **模型推理** | 优先遵循 Canvas 标准方形画布流程；长批处理循环必须显式清理显存与垃圾回收。 |
| **自动化任务** | `RELP_autorun.py` 依赖 Kerberos 票据保活与 `caffeinate` 屏幕防休眠，确保文件传输与回传闭环。 |
