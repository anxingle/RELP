# RELP：按 product / failure_mode 运行案例与 Strategy 流程

核对日期：2026-09-21。以 Ubuntu-24.04 中 `/home/an/ssd_work/RELP` 的实际运行方式编写；已确认本次核对时，Windows 仓库与 WSL 仓库的 `RELP_Configuration.xlsx`、入口、流水线和策略源码一致。

本文根据 Excel 与源码核对命令筛选、Strategy 分派和检测步骤，没有启动推理。按要求，不检查或下载 `weights`、`Pics` 目录。下文“可选中/可进入策略”表示配置和分派层面成立，不等于已经验证完整检测成功。

## 1. 先试哪几个

当前正式工作表 `Failure Mode` 只有两项任务：

| 正式表行号 | Product | Generation | Failure Mode | Strategy |
| --- | --- | --- | --- | --- |
| 2 | Textile | R692 | bubble | `ComplexTextileStrategy.execute()` |
| 3 | Silicon_Case | 2025 | Silicon Delam | `DetectronSegStrategy.execute()` |

在 WSL 的仓库根目录执行：

```bash
cd /home/an/ssd_work/RELP

# 你已经试过的硅胶脱层案例
uv run python RELP3_main.py --product Silicon_Case --generation 2025 --failure_mode "Silicon Delam"

# 当前配置下可以直接选中的另一个案例：织物起泡
uv run python RELP3_main.py --product Textile --generation R692 --failure_mode "bubble"
```

接下来建议按本文的启用步骤，逐个试：`Silicon_Case / Crop` → `Textile_Case / Textile Discoloration` → `Watch_Band / Fraying` → `Macbook / Light Bleed`。它们分别有助于理解裁切、KNN 颜色分析、DINO 特征分析和专用规则流程。

`Textile / R692 / bubble` 与 `Textile_Case / 2026 / Bubbling` 是两项不同配置，不能互换名称。后者当前缺少方法配置。

## 2. 更多命令运行前，先启用对应 Excel 行

`Failure Mode (2)` 中有 25 个案例，但程序只读取正式表 `Failure Mode`。同样，`Flow (2)` 不会自动合并进 `Flow`。命令行的三个参数用于筛选正式任务表，不能自动激活备用表中的任务。

每次试验按以下顺序操作：

1. 先为正在使用的 `RELP_Configuration.xlsx` 保存一份备份。
2. 在 `Failure Mode (2)` 找到本文指定的案例行，把 A:I 列复制到正式 `Failure Mode` 表。第一次学习时建议正式表仅保留本次案例的一行任务，保留第 1 行表头。备用表保持原样，便于切换。
3. 检查正式 `Flow` 是否有对应 `Failure Mode`。下面标为“补配置”的案例还需要额外处理；不要直接把整个 `Flow (2)` 覆盖正式表，因为两者的参数并不完全一致。
4. 保存正在运行的 WSL 仓库里的 Excel，再执行相应命令。Windows 的 `C:\Users\anxin\ssd_work\RELP` 与 WSL 的 `/home/an/ssd_work/RELP` 是两份目录，编辑其中一份不等于另一份自动更新。

**每次只保留一个案例还有具体的代码原因。** `pipeline.py` 先按产品和代际筛选，但构造策略上下文时，又从未按产品筛选的同名 Failure Mode 记录中取第一行。因此同时启用 `Silicon_Case / Crop` 和 `Macbook / Crop`，即使命令指定 Macbook，也可能把 Silicon_Case 的配置交给策略。本文采用逐个启用的方式避免这个问题，没有改动业务代码。

所有命令默认使用该行的 `Pic_Path`。如果云盘文件下载到另一个位置，最稳妥的做法是修改该行的 `Pic_Path`。硅胶策略也支持在命令末尾追加 `--pic_path`：

```bash
# 把路径替换为本次案例的真实图片目录；这里只展示参数写法
uv run python RELP3_main.py --product Silicon_Case --generation 2025 --failure_mode "Silicon Delam" --pic_path "/your/downloaded/silicon"
```

这仍然需要正式 `Failure Mode` 中有匹配的任务行。`CropStrategy`、`KnnStrategy` 会在执行时重新读取 Excel 的路径，因此不要把这个命令行覆盖能力推广到所有策略。

## 3. Excel 案例总表

下面的“案例行”均指 `Failure Mode (2)` 的 Excel 行号。状态含义：**直接**＝当前正式表已启用；**启用**＝先按第 2 节复制任务行；**校正**＝需要处理冲突；**补配置**＝不仅需要复制任务行；**待定义**＝当前没有可执行的方法配置。

| 编号 | 案例行 | Product / Generation | Failure Mode | 方法来源 | 实际入口 Strategy | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | 2 | Bumper_Case / 2025 | Bumper Crack | Flow!G6 = Detectron_Seg | DetectronSegStrategy | 启用；复用硅胶实现 |
| T02 | 3 | Silicon_Case / 2025 | Silicon Delam | Flow!G2 = Detectron_Seg | DetectronSegStrategy | 直接 |
| T03 | 4 | Silicon_Case / 2025 | Crop | Flow!G14 = Crop | CropStrategy | 启用 |
| T04 | 5 | Textile_Case / 2025 | Textile Discoloration | Flow!G3 = KNN+Filtering[HSV] | KnnStrategy | 启用 |
| T05 | 6 | Textile_Case / R187 | R187_Textile Staining | Flow!G12 = Filtering[HSV]&&KNN | KnnStrategy | 启用 |
| T06 | 7 | Textile_Case / 2025 | Textile Staining | Flow!G5 = Dino；G7 = KNN | DinoStrategy（工厂取首行） | 校正重复 Flow |
| T07 | 8 | Textile_Case / 2026 | Bubbling | 两份 Flow 均缺少 | 无 | 待定义 |
| T08 | 9 | Watch_Band / B721 | B721_discoloration | Flow!G13 = Dino+KNN[...] | KnnStrategy | 启用；复合方法见下文 |
| T09 | 10 | Watch_Band / B740 | Fraying | Flow!G10 = Dino | DinoStrategy | 启用 |
| T10 | 11 | Watch_Band / B217 | discolor | Flow!G11 = Dino | DinoStrategy | 启用 |
| T11 | 12 | Power_Adapter / 2025 | Crack | Flow!G8 = Dino | DinoStrategy | 启用 |
| T12 | 13 | Macbook / 2025 | Crop | Flow!G14 = Crop | CropStrategy | 启用；与 T03 分开 |
| T13 | 14 | Macbook / 2025 | Macbook General | Flow!G16 = Dino+Filtering[...] | DinoStrategy | 启用；存在执行差异 |
| T14 | 15 | Macbook / J701 | Light Bleed | Flow!G25 = Filtering[Stencil-ROI]+Adaptive Gaussian[Mask] | MacbookLightBleedStrategy | 启用 |
| T15 | 16 | Macbook / K11p | K11p General | Flow!G17 + General FM!第2～4行 | GeneralFmDispatcher → Dino / Filtering | 启用；部分分组规则缺失 |
| T16 | 17 | Macbook / backside | Macbook Pit | Flow!G24 + General FM!第5行 | GeneralFmDispatcher → Dino | 启用 |
| T17 | 18 | Matt_Screen / 2026 | Minor_Wear_map1 | Flow (2)!G28 | DinoStrategy | 补 Flow / 检测规则 |
| T18 | 19 | Matt_Screen / 2026 | Minor_Wear_map2 | Flow (2)!G29 | DinoStrategy | 补 Flow / 检测规则 |
| T19 | 20 | Matt_Screen / 2026 | Moderate_Wear_map3 | Flow (2)!G30 | DinoStrategy | 补 Flow / 检测规则 |
| T20 | 21 | Matt_Screen / 2026 | Minor_Scratch | Flow (2)!G31 | FilteringStrategy | 补 Flow / 检测规则 |
| T21 | 22 | Matt_Screen / 2026 | Screen Defect | 两份 Flow 均缺少 | 无 | 待定义 |
| T22 | 23 | Macbook / K11p | HiAA Bleach | Flow!G28 + 特殊名称路由 | ComplexBleachStrategy | 启用 |
| T23 | 24 | Textile / R692 | bubble | Flow!G29 = Complex_Textile | ComplexTextileStrategy | 直接；正式表路径不同，见下文 |
| T24 | 25 | Macbook / K11p | GroundingCrop['elliptic object', ratio=1.3] | 按名称路由，无 Flow 时使用默认参数 | CropStrategy | 启用；见裁切说明 |
| T25 | 26 | Textile / R692 | GroundingCrop['phone case in the center', Top_Oppt[1]] | 按名称路由，无 Flow 时使用默认参数 | CropStrategy | 启用；见裁切说明 |

`Flow` 中还有一些没有顶层任务行的方法，例如 `Advanced Textile Discoloration`、`Dino Crop`、`Macbook Dent`、`Macbook Paintwear`。它们不能仅凭 Flow 名称，就当作已配置的 product / generation 运行组合。`K11p dent` 则是 T15 分派产生的子任务。

## 4. 所有案例共有的启动过程

```text
RELP3_main.py: main()
  → ConfigManager.load_excel("RELP_Configuration.xlsx")
  → DefectDetectionPipeline.run()
  → 按 product / generation / failure_mode 筛选正式 Failure Mode
  → 找正式 Flow 中该 Failure Mode 的 Defect Identification
  → StrategyFactory.get_strategy(method_name, fm, context)
  → 创建策略，父类读取 Output 等配置
  → strategy.execute()
```

`Strategy` 指“执行这类任务的 Python 类”；`.execute()` 是真正启动该任务处理的方法。`DUT` 指被测样品；`mask` 指标记图中某个区域的像素掩膜；`SN` 指样品编号。

模型、分割、图像规则和评分并非每个策略都相同。下文区分 Excel 写下的分析意图与源码实际执行的步骤。

## 5. Detectron 分割：T01、T02

```bash
# T01：先启用 Failure Mode (2) 第 2 行；使用 Flow 第 6 行
uv run python RELP3_main.py --product Bumper_Case --generation 2025 --failure_mode "Bumper Crack"

# T02：当前已经启用；使用 Flow 第 2 行
uv run python RELP3_main.py --product Silicon_Case --generation 2025 --failure_mode "Silicon Delam"
```

两条命令都调用 `DetectronSegStrategy.execute()`，实现位于 `utils/core/strategies/detectron_seg_strategy.py`。

执行流程：

1. 读取任务、Flow、Output，以及该产品/代际的模型配置。
2. 加载两个 Detectron 模型：一个找缺陷，一个找被测对象 DUT；初始化 SAM2。
3. 遍历图片，读取编号信息。Silicon Delam 的 `SN Mapping=Excel`，会读取与图片同名的 `.xlsx`；Bumper Crack 的 `File_Path` 在这个策略中按图片名称方式处理。
4. 缺陷模型先产生全图缺陷 mask；DUT 模型定位各个对象及需要排除的区域。
5. 结合 SAM2 分割、对象轮廓与缺陷 mask 的交集，细化有效缺陷区域。
6. 结合 Reference、Output 计算面积/轮廓等指标，保存叠加图片和参数结果。

两者的 Output 配置均为 `Individual`、`[Area, Contour]`，分别来自 Output 第 2、7 行。

**T01 的边界：** 代码为 Bumper Crack 选用的也是硅胶专项实现，没有单独的 BumperStrategy。该实现包含硅胶目录解析和 DUT 参考逻辑，不能只根据 `Flow!K6=General` 就认定它会执行完整的 General 参考流程。可用来观察模型推理，但应先核对生成的样品信息和测量基准。

当前此策略处理图片时会构造 `图片主名 + ".jpg"`；仅有 PNG/JPEG 扩展名的数据可能被跳过。结果默认位于 `Result/<product>_<generation>_<failure_mode>_<failure_mode>_Result/`，参数汇总文件为 `Parametric_Output.xlsx`。

## 6. 裁切与整理输入：T03、T12、T24、T25

### 普通 Crop

```bash
# T03：启用 Failure Mode (2) 第 4 行
uv run python RELP3_main.py --product Silicon_Case --generation 2025 --failure_mode "Crop"

# T12：切换为 Failure Mode (2) 第 13 行；不要同时保留两个 Crop 任务
uv run python RELP3_main.py --product Macbook --generation 2025 --failure_mode "Crop"
```

两条命令调用 `CropStrategy.execute()`，共用正式 Flow 第 14 行。其中 `Crop_Output_Dim=1280`。

执行流程：Detectron 找到图中的多个目标框 → SAM2 分割各个目标 → 按目标顺序逐个处理 → 把目标外背景涂黑 → 优先按原始检测框裁切 → 等比缩放并放入 1280×1280 画布 → 保存每个目标的裁切图。

它用于准备、统一图片输入，不执行脱层/裂纹判定，也不在这段实现里生成缺陷参数 Excel。结果位于 `Result/<product>_<generation>_Crop_Result/Crop_Result/`。一张原图有多个目标时会生成带 `Instance` 编号的多张图片。

### 带文字提示的 GroundingCrop

```bash
# T24：启用 Failure Mode (2) 第 25 行；整段 failure_mode 保留双引号
uv run python RELP3_main.py --product Macbook --generation K11p --failure_mode "GroundingCrop['elliptic object', ratio=1.3]"

# T25：启用 Failure Mode (2) 第 26 行
uv run python RELP3_main.py --product Textile --generation R692 --failure_mode "GroundingCrop['phone case in the center', Top_Oppt[1]]"
```

同样调用 `CropStrategy.execute()`，内部切换为 Grounding DINO 找框，再交给 SAM2 分割和裁切。

- T24 的文字提示为 `elliptic object`。`ratio=1.3` 表示保留 `max(宽/高, 高/宽) >= 1.3` 的框，是长宽比筛选条件。
- T25 的文字提示为 `phone case in the center`。`Top_Oppt[1]` 表示检测结果按置信度排序后最多取 1 个。
- 这两个名称在当前两份 Flow 表中都不存在，但工厂可以按 `groundingcrop` 前缀分派。CropStrategy 找不到 Flow 时只打印警告，并继续使用默认值。最终 Grounding 裁切的目标尺寸默认是 1280；已有 Flow 时这条分支读取 `Dino_Input_Dim`。
- 输出目录为 `Result/<product>_<generation>_<完整failure_mode>_Result/Crop_Result/`，主要产物是裁切图片。

源码入口：`utils/core/strategies/crop_strategy.py:21`；名称参数解析在第 109 行附近；Grounding 图像输出在第 295 行附近。

## 7. KNN 颜色分析：T04、T05、T08

### T04：Textile_Case / 2025 / Textile Discoloration

先启用 `Failure Mode (2)` 第 5 行；使用正式 Flow 第 3 行、KNN 第 2～8 行、Filtering 第 5 行、Output 第 3 行。

```bash
uv run python RELP3_main.py --product Textile_Case --generation 2025 --failure_mode "Textile Discoloration"
```

Strategy：`KnnStrategy.execute()`。流程为：准备 DUT 图像和轮廓 → 根据图像颜色匹配 KNN 表里的颜色配置 → 执行两级颜色聚类/异常区域提取 → 对结果执行 `Filtering[HSV]` → 与 DUT 轮廓求交集 → 保存叠加图、mask 和指标结果。当前正式 Flow 的 `PCA_Alignment=No`，不会因为备用 Flow 表写了 Yes 就自动旋转对齐。

### T05：Textile_Case / R187 / R187_Textile Staining

先启用 `Failure Mode (2)` 第 6 行；使用 Flow 第 12 行、KNN 第 9 行、Filtering 第 6～7 行、Output 第 4 行。

```bash
uv run python RELP3_main.py --product Textile_Case --generation R187 --failure_mode "R187_Textile Staining"
```

Strategy：`KnnStrategy.execute()`。基本流程与 T04 相同，并读取该案例 KNN 行中的切片、缩放、聚类参数。该行设置了 `Slicing=[X[0.47:0.535]]`、`Resize=1st=0.25`、`Hole Detection=No`。

**实际顺序仍然是先 KNN、后 HSV。** 虽然 Flow 写成 `Filtering[HSV]&&KNN`，该 Strategy 的代码先调用 KNN，再用正则提取 `Filtering[...]` 执行；不能把字符串顺序当作真实执行顺序。

### T08：Watch_Band / B721 / B721_discoloration

先启用 `Failure Mode (2)` 第 9 行；使用 Flow 第 13 行、KNN 第 10 行、Output 第 11 行。

```bash
uv run python RELP3_main.py --product Watch_Band --generation B721 --failure_mode "B721_discoloration"
```

Strategy：`KnnStrategy.execute()`。实际执行 DUT 预处理 → KNN 颜色配置匹配 → KNN 异常区域提取 → DUT 轮廓限制 → 保存结果。

**Flow 写的是 `Dino+KNN[...]`，工厂却先匹配 KNN，且 KnnStrategy 没有执行 DINO 的分支。** 因而当前命令不能理解成“先 DINO 再 KNN”。这里的方法字符串没有 `Filtering`，也不会因为 Filtering 表存在 B721 行就自动追加 HSV 过滤。

这组策略的主要输出包括 `Inferred Pic/*_knn_overlay.jpg`、调试 mask、`.npy` mask 和参数 Excel。源码入口为 `utils/core/strategies/knn_strategy.py:26`；KNN 调用在第 323 行附近，追加 Filtering 在第 358 行附近。

## 8. DINO 特征分析：T06、T09、T10、T11、T13

这组都进入 `DinoStrategy.execute()`，源码为 `utils/core/strategies/dino_strategy.py`。共有流程如下：

1. 读取 Flow、Dino_TH、Filtering、Output，并加载 DUT、SAM2、DINO 等所需模型。
2. 用 `prepare_dut_image()` 准备被测对象图像：定位、分割、裁切；是否旋转对齐由 PCA 配置控制。
3. 将处理后的图片和 DUT 轮廓送入 `dinov3_utils.process_single_image_pipeline()`，提取特征异常区域。这里的 DINO 与用文本找物体的 Grounding DINO 用途不同。
4. 根据 Dino_TH 阈值生成缺陷 mask；配置了通用缺陷检测器时，还会执行该检测器。
5. 执行实际读取到的后处理，随后计算指标、生成 mask/叠加图和参数 Excel。由 General FM 分派调用时还会接收范围约束和前一阶段的中间结果。

### T09：Watch_Band / B740 / Fraying

先启用 `Failure Mode (2)` 第 10 行；使用 Flow 第 10 行、Dino_TH 第 5 行、Output 第 12 行。

```bash
uv run python RELP3_main.py --product Watch_Band --generation B740 --failure_mode "Fraying"
```

具体流程：DUT 预处理/对齐 → DINO 特征分析 → 以 `Dino_Low=0.1` 形成异常区域 → 计算、汇总缺陷和 DINO 距离分布 → 保存结果。正式 Flow 的 `Upsampling` 为空，`Upsampling Definition=448`；不能照备用 Flow 中的 Yes/160 来解释本次运行。Filtering 中的 Fraying 行写的是 B704，也不应当成 B740 已生效的额外面积规则。

### T10：Watch_Band / B217 / discolor

先启用 `Failure Mode (2)` 第 11 行；使用 Flow 第 11 行、Dino_TH 第 6 行、Filtering 第 15 行、Output 第 10 行。

```bash
uv run python RELP3_main.py --product Watch_Band --generation B217 --failure_mode "discolor"
```

具体流程：DUT 预处理/对齐 → DINO 异常分析（`Dino_Low=0.05`）→ 执行 Post Processing 中的 `Filtering[HSV]` → 计算缺陷占比、颜色分箱和配置的评分 → 保存结果。这是理解“AI 先圈出异常，再用颜色规则限制结果”的一个案例。

### T11：Power_Adapter / 2025 / Crack

先启用 `Failure Mode (2)` 第 12 行；使用 Flow 第 8 行、Dino_TH 第 4 行、Filtering 第 8～12 行、Adaptive Gaussian 第 2 行、Output 第 9 行。

```bash
uv run python RELP3_main.py --product Power_Adapter --generation 2025 --failure_mode "Crack"
```

具体流程：DUT 预处理 → DINO 异常分析（`Dino_Low=0.085`）→ 按 `EcoRel[Save]` 配置执行额外的通用缺陷检测 → 按 Post Processing 依次执行面积百分比过滤、Adaptive Gaussian、像素面积过滤、Shape 过滤 → 计算结果并保存。

Output 配置关注 `Length`，并配置灰度分箱和评分。`Fine Tuning=Yes`、`Curved Line Measurement=Yes` 等表格字段是否被当前具体路径消费，不能仅凭列值认定；本次列出的检测步骤以 execute 的实际调用为准。

### T13：Macbook / 2025 / Macbook General

先启用 `Failure Mode (2)` 第 14 行；使用正式 Flow 第 16 行、Dino_TH 第 8 行、Filtering 第 17 行。

```bash
uv run python RELP3_main.py --product Macbook --generation 2025 --failure_mode "Macbook General"
```

实际入口是 `DinoStrategy`。当前正式 `General FM` 没有 Macbook General 的映射，因此不会自动分派成 `Macbook Dent` 和 `Macbook Paintwear`。

可以确定的执行链是 DUT 预处理 → DINO 异常分析（`Dino_Low=0.01`）→ 按 `Macbook[Save]` 配置进行通用缺陷检测 → 当前可解析的后处理 → 输出。

本项存在两个具体配置/实现差异：

- Flow 的方法写了 `Dino+Filtering[Morph Top-Hat]`，但 DinoStrategy 在没有 `config_group_override` 时，从任务表 Failure Mode 查找追加的 Defect Identification，而当前任务表没有这一列。因此不能声称这条命令已经追加执行 Top-Hat。
- `Flow!J16` 的 Post Processing 实际为不完整的 `Filtering[Area(Pixel)`，缺少闭合括号。完整的面积过滤意图不能靠这段文本可靠表达。

所以它可用于观察当前 DINO 实现，但尚不能作为“完整 DINO + Top-Hat + 面积过滤流程已经复现”的依据。

### T06：Textile_Case / 2025 / Textile Staining——先处理重复配置

正式 Flow 中有两行相同的 Failure Mode：第 5 行使用 Dino，第 7 行使用 KNN。工厂拿第一行方法，但 ConfigManager 的字典会被后一行覆盖，导致策略与后续读取的配置不一致。

如果要尝试 DINO 版本，先备份，再在正式 Flow 中保留第 5 行的 Textile Staining 配置，移除该名称的重复第 7 行；随后启用 `Failure Mode (2)` 第 7 行。这里指的是本次核对时的行号，编辑后应按名称定位。

```bash
uv run python RELP3_main.py --product Textile_Case --generation 2025 --failure_mode "Textile Staining"
```

校正后对应 `DinoStrategy.execute()`：DUT 预处理/对齐 → DINO 异常分析 → 第 5 行的 `RGB Filtering` 后处理（Filtering 第 2～4 行）→ 结果汇总。当前 Dino_TH 没有完全同名的 Textile Staining 项，代码使用默认阈值 0.0；需要针对数据确认阈值，不能把其他 Staining 行直接视为该任务配置。

## 9. General FM 两阶段分派：T15、T16

### T15：Macbook / K11p / K11p General

先启用 `Failure Mode (2)` 第 16 行；使用 Flow 第 17～20 行以及正式 General FM 第 2～4 行。

```bash
uv run python RELP3_main.py --product Macbook --generation K11p --failure_mode "K11p General"
```

外层调用 `GeneralFmDispatcher.execute()`，随后调用具体子策略：

```text
GeneralFmDispatcher
  → 扫描图片，按 General FM 的 Mapping 分组
  → 第一阶段：DinoStrategy
       DUT 预处理 → DINO 异常 mask → 通用检测范围约束
       返回中间图像、轮廓和 mask
  → 第二阶段：按分组调用 FilteringStrategy，复用第一阶段结果
       B1 → Side1 → K11p dent → Morph Black-Hat
       B2 → Side2 → K11p dent → ODBP
       其余 → Remaining → K11p dent → BG Subtraction → Color Distance
  → 结果保存到父任务的结果目录
```

第一阶段使用 `Dino_TH` 第 10 行的 `Dino_Low=0.1`，Flow 的通用检测设置为 `K11p Gen[Save]`。General FM 指定 `Defect Class=dent` 和 `Scope=Defect Detection BBOX`。

第二阶段按正式 Flow 的 `Config_Group` 选方法：Remaining 对应 Flow 第 18 行，Side1 对应第 19 行，Side2 对应第 20 行。`FilteringStrategy` 按顺序调用过滤器，不能把字符串中的 `&&` 直接解释为当前外层已经完成掩膜并集运算。

当前配置边界：Filtering 中有 Remaining 的 BG Subtraction/Color Distance 和 Side1 的 Black-Hat 规则，但没有 Side2 的 ODBP 专项规则；因此 Side2 不应视为已经具备经过确认的检测参数。另外，General FM 第 3 行 Product 写成 `Macbook1`，当前 dispatcher 主要按 Failure Mode 匹配，并不会因此自动排除该行。

### T16：Macbook / backside / Macbook Pit

先启用 `Failure Mode (2)` 第 17 行；使用 Flow 第 24 行、正式 General FM 第 5 行、Dino_TH 第 11 行、Output 第 17 行。

```bash
uv run python RELP3_main.py --product Macbook --generation backside --failure_mode "Macbook Pit"
```

实际调用链：`GeneralFmDispatcher` → 第一阶段 `DinoStrategy` → 第二阶段 `DinoStrategy`。

第一阶段准备 DUT 图像、运行 DINO（`Dino_Low=0.112`），并按 `Backside Pit[Save]` 和 General FM 的 `Defect Class=Pit`、`Scope=BBOX IOU[0.005]` 做范围约束。General FM 目标 Failure Mode 留空，dispatcher 会回到原名称 Macbook Pit；第二阶段复用中间结果，汇总、测量并输出，并非必然再对同一张图完整推理一次。

源码依据：`utils/core/strategies/general_fm_dispatcher.py:137` 开始第一阶段，第 156 行之后组织第二阶段；子策略通过 `inherited_data` 接收前一阶段结果。

## 10. 专用流程：T14、T22、T23

### T14：Macbook / J701 / Light Bleed

先启用 `Failure Mode (2)` 第 15 行；使用 Flow 第 25 行、Filtering 第 29 行、Adaptive Gaussian 第 4 行、Output 第 13 行。

```bash
uv run python RELP3_main.py --product Macbook --generation J701 --failure_mode "Light Bleed"
```

Strategy：`MacbookLightBleedStrategy.execute()`，文件为 `utils/core/strategies/macbook_light_bleed_strategy.py`。

执行流程：

1. Grounding DINO 使用 `white screen.` 提示找屏幕，选择面积最大的框。
2. SAM2 分割屏幕，并取最大外轮廓形成屏幕 mask。
3. 使用 `round object.` 提示寻找圆形参考物，再用 SAM2 得到参考面积；没找到时回退到屏幕面积。
4. 读取 Flow 方法，执行 `Stencil-ROI` 模板区域筛选与 `Adaptive Gaussian`，得到漏光区域。
5. 计算真实缺陷像素数、相对屏幕面积占比、相对参考物面积占比；根据 Output 汇总指标和评分。
6. 保存叠加图、参考 mask 和 `Parametric_Output.xlsx`。

这是按 `Light Bleed` 名称优先分派的专用策略，虽然 Flow 文本以 Filtering 开头，也不会走普通 FilteringStrategy。

### T22：Macbook / K11p / HiAA Bleach

先启用 `Failure Mode (2)` 第 23 行；使用 Flow 第 28 行、Filtering 第 30～34 行、Output 第 23～24 行。

```bash
uv run python RELP3_main.py --product Macbook --generation K11p --failure_mode "HiAA Bleach"
```

Strategy：`ComplexBleachStrategy.execute()`，文件为 `utils/core/strategies/complex_bleach_strategy.py`。工厂优先匹配 HiAA Bleach 名称，不会直接创建普通 GroundingSamStrategy。

执行流程：

1. 从 Flow 的 `GroundingSAM['elliptic object', ratio=1.3]` 解析文字提示和长宽比要求。
2. Grounding DINO 找候选目标，保留满足长宽比要求的框。
3. SAM2 分割这些框，合并目标区域。
4. 一般图片根据 Post Processing 执行 `Band-ROI` 与 `Global Threshold`，检测指定带状区域内的异常。
5. 如果图片路径含 `matte`，加载 `HiAA Bleach (matte)` 的专用 HSV 规则，根据计算出的比例选择 `halo` 或 `no halo` 配置组，并使用相应输出配置。
6. 保存叠加图、调试信息和参数 Excel。

当前 Flow 没有追加 `-Dino`，因此此案例不会启用该专用类中的可选 DINO 特征分析阶段。这里已有的 Grounding DINO 用于文字定位。

### T23：Textile / R692 / bubble

这项当前已在正式 Failure Mode 第 2 行启用；使用 Flow 第 29 行、Dino_TH 第 16 行和 Output 第 25 行。

```bash
uv run python RELP3_main.py --product Textile --generation R692 --failure_mode "bubble"
```

Strategy：`ComplexTextileStrategy.execute()`，文件为 `utils/core/strategies/complex_textile_strategy.py`。

执行流程：

1. Grounding DINO 用 `phone case in the center.` 定位手机壳，选取满足条件且置信度最高的框；SAM2 分割主体。
2. 结合充电孔检测和目标长宽比判断 A/B/CD 面，并从有效区域中去除充电孔等干扰区域。
3. 当前代码使用整个面 `Y[0.0:1.0]`；CD 面另外检测按钮，构建按钮附近的惩罚区域。
4. 寻找红色标签 `red tag.`，用其分割面积作动态参考；无法取得有效标签时回退到壳体面积。
5. DINOv3 对有效壳体区域做异常分析，读取 `Dino_Low=0.02`；当前 Flow 开启 Upsampling，输入尺寸为 `[1280,1280]`。
6. 将异常 mask 限制在有效区域，整理异常距离 CSV，按 Output 中的分箱/评分配置计算结果。代码会标出按钮附近的惩罚区域，但当前惩罚乘数写成 1.0，因此不会实际降低这部分数值。
7. 保存热力图/轮廓叠加图、调试图、CSV 和 `Parametric_Output.xlsx`。

正式任务行的图片路径是 `Pics/phone case/2026/textile case`，备用第 24 行则指向更深的 `ML GradeA:C:D pictures _ 20260716` 子目录。本次直接运行使用正式表路径；如果复制备用行，会改变遍历范围。

## 11. Matt_Screen 四项：有候选方法，但配置尚不齐全

以下命令给出了准确参数组合，**需要先补配置，不属于当前 Excel 直接就绪的案例**。

| 编号 | 启用任务行 | 还需复制到正式 Flow 的行 | Strategy |
| --- | --- | --- | --- |
| T17 | Failure Mode (2) 第 18 行 | Flow (2) 第 28 行 | DinoStrategy |
| T18 | Failure Mode (2) 第 19 行 | Flow (2) 第 29 行 | DinoStrategy |
| T19 | Failure Mode (2) 第 20 行 | Flow (2) 第 30 行 | DinoStrategy |
| T20 | Failure Mode (2) 第 21 行 | Flow (2) 第 31 行 | FilteringStrategy |

```bash
# T17：先处理下文的 Flow、阈值和 Filtering 配置缺口
uv run python RELP3_main.py --product Matt_Screen --generation 2026 --failure_mode "Minor_Wear_map1"

# T18
uv run python RELP3_main.py --product Matt_Screen --generation 2026 --failure_mode "Minor_Wear_map2"

# T19
uv run python RELP3_main.py --product Matt_Screen --generation 2026 --failure_mode "Moderate_Wear_map3"

# T20
uv run python RELP3_main.py --product Matt_Screen --generation 2026 --failure_mode "Minor_Scratch"
```

T17～T19 的配置意图：DUT 预处理 → DINO 异常检测 → Morph Top-Hat → Slicing → 指标和输出。**当前实际能确定的是 DinoStrategy 路径；要完整复现这条意图还存在以下缺口：**

- 正式 Filtering 表没有 Matt_Screen 的 Top-Hat、Slicing 专项参数。
- Dino_TH 第 12、13 行写的是 `Moderate_Wear_map1/map2`，与任务的 `Minor_Wear_map1/map2` 不同。T17/T18 不会匹配到这两行，代码会使用默认阈值 0.0。应先确认业务上应该是哪一种名称/参数，不直接改名套用。
- T19 的 `Moderate_Wear_map3` 能匹配 Dino_TH 第 14 行，`Dino_Low=0.1`。
- DinoStrategy 的 `+Filtering` 读取差异同 T13：直接任务没有配置组时，这部分不会可靠地从 Flow 中取出；Post Processing 中的 Slicing 会进入解析，但仍缺少本案例规则。

T20 的配置意图：DUT 预处理 → Morph Top-Hat 找细小亮线/划痕 → 像素面积过滤 → 输出。实际 `FilteringStrategy.execute()` 会解析 Defect Identification 中的 Top-Hat，并按 Filtering 表参数或实现默认值处理；当前没有 Matt_Screen 专项规则，而且它没有像 DinoStrategy 那样追加解析 Flow 的 Post Processing，因此不能声称后面的 `Filtering[Area(Pixel), Save(Mask+Contour)]` 已执行。

补齐这些业务参数、确认追加步骤的实现后，再把这四项当作完整检测案例。这里没有替你猜测阈值或修改配置。

## 12. 两项当前无法定义有效运行流程的案例

| 编号 | 来源行 | Product / Generation / Failure Mode | 当前缺口 |
| --- | --- | --- | --- |
| T07 | Failure Mode (2) 第 8 行 | Textile_Case / 2026 / Bubbling | 正式和备用 Flow 都没有同名方法；任务行还缺少运行字段 |
| T21 | Failure Mode (2) 第 22 行 | Matt_Screen / 2026 / Screen Defect | 正式和备用 Flow 都没有同名方法，工厂也没有对应名称分支 |

这两项即使复制到正式任务表，也不能从现有配置确定检测 Strategy，所以没有把它们列为可运行命令。需要先向项目负责人确认对应的算法、Flow、专用参数和预期输出。

尤其不要把 `Screen Defect` 自行改成源码里的 `mistral_screen_defect`，也不要把 `Bubbling` 自动替换成 `bubble`。这些是不同的标识，现有 Excel 没有给出它们等价的依据。

## 13. 怎样确认跑到了你期望的策略

运行时先看三类日志：

1. `DEBUG: Filtering execution for Product=...`：确认命令行参数。
2. `Processing Failure Mode: ...` 与 `DEBUG: Matching Flow rows ...`：确认读到了正式任务和 Flow。GroundingCrop 是允许没有 Flow 行的例外。
3. `Executing ...Strategy` 或 `[DISPATCHER] Starting General FM Dispatcher ...`：确认实际策略。出现路由日志只证明开始调用，后面还要看到处理图片、保存结果的信息。

没有匹配任务、没有检测目标、配置不完整时，一些路径会打印消息后正常返回，不一定出现 Python 异常。因此“命令退出码为 0”不能单独证明完成了缺陷检测。

多数检测策略把结果写到 `Result/<product>_<generation>_<failure_mode>_<failure_mode>_Result/`，包括 `Inferred Pic`、`reference` 和有数据时的 `Parametric_Output.xlsx`。Crop 的路径规则不同，见第 6 节。General FM 子任务共享父任务结果目录，不能假定每个子策略都会单独留下完整汇总文件。

## 14. 对照源码的位置

以下位置均相对于本仓库根目录，行号以本次核对版本为准：

| 文件与位置 | 用途 |
| --- | --- |
| [RELP3_main.py:24](C:/Users/anxin/ssd_work/RELP/RELP3_main.py:24) | 入口 main：加载 Excel 后启动 Pipeline |
| [config_manager.py:86](C:/Users/anxin/ssd_work/RELP/utils/config/config_manager.py:86) | 加载正式工作表；Flow 按 Failure Mode 建字典 |
| [pipeline.py:157](C:/Users/anxin/ssd_work/RELP/utils/core/pipeline.py:157) | 参数解析、任务筛选与路由 |
| [pipeline.py:486](C:/Users/anxin/ssd_work/RELP/utils/core/pipeline.py:486) | 构造上下文时取同名 Failure Mode 第一行，需要避免同名多产品任务同时启用 |
| [strategy_factory.py:10](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/strategy_factory.py:10) | Strategy 选择顺序，General FM 与专用名称优先 |
| [base_strategy.py:9](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/base_strategy.py:9) | 初始化、Output 等公共配置 |
| [detectron_seg_strategy.py:48](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/detectron_seg_strategy.py:48) | T01/T02 |
| [crop_strategy.py:21](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/crop_strategy.py:21) | T03/T12/T24/T25 |
| [knn_strategy.py:26](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/knn_strategy.py:26) | T04/T05/T08 |
| [dino_strategy.py:217](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/dino_strategy.py:217) | T06/T09/T10/T11/T13 及部分分派/屏幕任务 |
| [dino_strategy.py:558](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/dino_strategy.py:558) | +Filtering 与 Post Processing 的读取差异 |
| [general_fm_dispatcher.py:28](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/general_fm_dispatcher.py:28) | T15/T16 两阶段分派 |
| [filtering_strategy.py:192](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/filtering_strategy.py:192) | 按配置组取 Flow；第 350 行附近循环执行过滤器 |
| [macbook_light_bleed_strategy.py:24](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/macbook_light_bleed_strategy.py:24) | T14 专用漏光流程 |
| [complex_bleach_strategy.py:19](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/complex_bleach_strategy.py:19) | T22 专用 HiAA Bleach 流程 |
| [complex_textile_strategy.py:26](C:/Users/anxin/ssd_work/RELP/utils/core/strategies/complex_textile_strategy.py:26) | T23 专用织物起泡流程 |

本次只新增本运行指南，没有修改 Excel、检测源码或运行环境，也没有执行上述检测命令。
