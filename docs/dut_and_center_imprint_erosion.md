# 已居中 imprint 数据集的 DUT 内缩处理

`dut_and_center_imprint_erosion.py` 读取已有图片与 SAM2 二值掩码，按欧氏距离向内去除 DUT 外围。无需重新运行 Grounding DINO / SAM2，只依赖 NumPy 和 OpenCV。

## 输入与输出

默认输入为仓库的 `data/center_train_imprint`，不依赖运行命令时的工作目录：

```text
center_train_imprint/
  train/good/<name>.jpg
  masks/<name>_mask.png
```

也支持 `mask/` 目录，以及与图片同 stem 的掩码。若同时存在 `mask/` 和 `masks/`，使用 `--mask-dir` 指定。命名歧义、缺失配对和会造成输出覆盖的重名会报错。图片与掩码尺寸必须一致，不会自动缩放掩码。

默认输出为输入目录旁的 `center_train_imprint_erosion`：

```text
center_train_imprint_erosion/
  train/good/<name>.png
  masks/<name>_mask.png
  previews/<name>.png
  processing_report.csv
  settings.json
```

输出保留输入画布尺寸、坐标、DUT 尺度及完整内区像素，不重新裁剪、缩放或居中。PNG 避免再次 JPEG 压缩，8K 输出会比输入 JPG 占用更多空间。二值输出 mask 表示 `alpha > 0` 的保留范围，包含窄过渡带；不是 alpha 本身。预览左侧为输入，右侧为处理结果。

输出必须是独立的空目录或不存在的目录；不会覆盖输入或已有结果。部分文件处理失败时，CSV 保留失败原因，进程退出码为 1；成功文件仍留在输出中。重新运行时请使用新输出目录。

## 运行

在已安装 OpenCV/NumPy 的 Python 环境中，从仓库根目录执行：

```powershell
# 只检查命名配对和输出位置，不读取图片内容或检查尺寸。
python docs/dut_and_center_imprint_erosion.py --dry-run

# 先以 40 个输入像素内缩，试跑 3 张。
python docs/dut_and_center_imprint_erosion.py --shrink-px 40 --limit 3 --output-root data/center_train_imprint_erosion_preview40

# 使用完整训练集，输出到默认的新目录。
python docs/dut_and_center_imprint_erosion.py --shrink-px 40

# 按每张图的 DUT 外接框短边的 1% 内缩，可适应不同输入分辨率。
python docs/dut_and_center_imprint_erosion.py --shrink-ratio 0.01 --output-root data/center_train_imprint_erosion_ratio001

# 只比较指定样本，增强轮廓平滑；所有参数仍是输入像素单位。
python docs/dut_and_center_imprint_erosion.py --pattern "*R5a-91.jpg" --shrink-px 80 --smooth-sigma 6 --feather-px 6 --output-root data/center_train_imprint_erosion_sample80
```

本机项目已配置 WSL Python；Windows 的默认 `python` 当前未安装 OpenCV。可在 PowerShell 直接调用现有环境，无需安装额外依赖：

```powershell
wsl -d Ubuntu-24.04 -- /home/an/.venvs/dinomaly/bin/python /mnt/c/Users/anxin/ssd_work/Dinomaly/docs/dut_and_center_imprint_erosion.py --dry-run

wsl -d Ubuntu-24.04 -- /home/an/.venvs/dinomaly/bin/python /mnt/c/Users/anxin/ssd_work/Dinomaly/docs/dut_and_center_imprint_erosion.py --shrink-px 40
```

WSL 命令中显式提供目录时应使用 `/mnt/c/...` 路径，而不是 Windows 的 `C:\...` 路径。

## 算法和参数

1. 提取面积最大的 DUT 外轮廓，忽略离散小组件。
2. 默认填孔仅用于构建测距形状，避免腐蚀扩大内部错误孔洞。
3. 可选：对测距形状做高斯平滑并重新二值化，缓解细碎起伏。
4. 显式补零边界，再计算精确欧氏距离变换。即使 DUT 贴着画布边缘，也会正确内缩。
5. 保留距离 `d > shrink_px` 的区域；可在这条新边界的内侧添加窄 alpha 过渡。
6. 与原始 mask 相交，只保留原图中原本有效的像素，并合成黑背景。

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--shrink-px` / `--erosion` | 40 | 输入掩码像素单位的收缩距离，支持小数 |
| `--shrink-ratio` | 未启用 | 收缩距离 = 该图主 DUT 外接框短边 × 比例，与 `--shrink-px` 互斥 |
| `--feather-px` | 2 | 收缩线内侧的额外线性过渡宽度，0 表示硬边 |
| `--smooth-sigma` | 0 | 掩码几何的高斯 sigma，0 表示仅做距离内缩 |
| `--keep-holes` | 未启用 | 启用时也测量到内部孔洞的距离，会使孔洞扩大 |
| `--pattern` | `*` | 输入文件名筛选 |
| `--limit` | 全部 | 仅处理排序后的前 N 张匹配图片 |
| `--preview-count` | 3 | 前 N 张成功图片输出左右对比预览，0 表示不生成 |

距离变换按像素中心测距，有离散栅格效应；这是正常现象。默认 `40` 是对照起点，不是已经确定的最优参数。可以在同一份输入上比较 40、80、120 像素。若 DUT 内有真实孔洞且需要沿孔洞一起内缩，可使用 `--keep-holes`。

**已有抠图中的黑色孔洞不能恢复纹理。** 默认填孔只防止这些孔洞进一步扩大，最终输出仍保留原始孔洞。若希望恢复，需回到未抠背景的原图。与背景连通的深缺口也不会自动变成平整直线；可增加内缩或启用小尺度平滑。

若启用 `--smooth-sigma`，距离基准是平滑后的轮廓，因此内缩量不再严格等于相对原始不规则轮廓的最小距离；最终结果仍不会越出原始 mask。

## 如何选择收缩宽度

- 先根据边缘杂乱区域的宽度选择内缩距离，再检查距离 DUT 边缘最近的 imprint，保留其周围的正常纹理。
- 所有像素参数都作用于当前输入分辨率。约 8000 像素图内缩 40 像素，再缩到 1280 时约为 6.4 像素；若训练输入进一步缩小，等效宽度也进一步缩小。
- `processing_report.csv` 记录每张图的实际收缩像素数、尺寸、输入/输出前景面积及去除比例。去除比例不包括 alpha 过渡带的亮度变化。
- 保持训练和推理的预处理一致。边界仍然是 DUT 与黑背景的交界；本脚本能去掉杂乱外缘，但是否降低异常误报需要用模型结果验证。
- 如果最终评分仍集中在新边界，可以在异常图上使用更靠内的有效评分 mask；这属于模型推理侧的后续调整，本脚本不修改训练或评分代码。

## 验证

```powershell
wsl -d Ubuntu-24.04 --cd /mnt/c/Users/anxin/ssd_work/Dinomaly -- /home/an/.venvs/dinomaly/bin/python -m unittest discover -s tests -p test_dut_and_center_imprint_erosion.py -v
```

测试覆盖：欧氏收缩距离、接触画布边界、孔洞处理、内部纹理与坐标保持、可选平滑的范围约束、空结果报错、中文路径、PNG 输出、配对及尺寸错误、输入保护和拒绝覆盖已有结果。

2026-10-09 本机验证：11 项测试通过；默认数据集的 120 张图均通过文件名配对检查。对用户指定的 `1_Imprint_...R5a-91.jpg`（8091×8091）实际运行的结果如下，均使用默认 `smooth_sigma=0`、`feather_px=2`：

| 内缩 | 去除前景面积比例 | 本地样例目录 |
|---|---:|---|
| 40 px | 2.40% | `runs/imprint_erosion_sample_40` |
| 80 px | 4.77% | `runs/imprint_erosion_sample_80` |
| 160 px | 9.39% | `runs/imprint_erosion_sample_160` |

三组输出都通过了尺寸一致、掩码不向外扩张、背景为零、远离新边缘的内部像素与输入完全一致，以及更大内缩结果包含于更小内缩结果的检查。预览中 40/80 px 仍保留部分底部灰色侧边，160 px 基本去除了该侧边；尚未验证 imprint 保留率或模型误报改善。40/80 px 的文件名筛选各匹配了 2 张图片，160 px 精确匹配了用户指定的 1 张。未批量处理完整训练集。
