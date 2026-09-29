经过对仓库代码（特别是 `pipeline.py`、`config_manager.py`、各大 Strategy 策略实现）以及 `RELP_Configuration.xlsx` 的底层数据进行严格比对，**你的这 4 点判断全部完全正确，且非常精准地切中了当前系统的核心运行逻辑与潜在架构陷阱**。

以下是对这 4 点逻辑的逐一确认和源码级的证据解析：

---

### 1. 确认逻辑 1：从 `Failure Mode (2)` 复制指定行到正式 `Failure Mode` 表
* **结论**：**完全正确。**
* **底层机制**：
  - `RELP_Configuration.xlsx` 中的 `Failure Mode (2)` 实际上是一个**“测试用例库 / 预设参数池”**（库中沉淀了 `Bumper_Case`、`Silicon_Case`、`Textile_Case`、`Watch_Band`、`Macbook`、`Matt_Screen` 等 25 种不同硬件形态的历史配置）。
  - 代码中 `ConfigManager.load_excel()`（`utils/config/config_manager.py` 第 95-99 行）以及 `pipeline.py`（第 224 行）**只读取表名为 `'Failure Mode'` 的工作表**。
  - 因此，日常需要执行某个产品的特定用例时，标准流程确实是从备用库 `Failure Mode (2)` 中找到该行，复制到正式的 `Failure Mode` 表中激活。

---

### 2. 确认逻辑 2：正式 `Failure Mode` 表最好仅保留本次运行的一行任务
* **结论**：**完全正确，这是避免各种隐蔽 Bug 的最佳实践。**
* **底层机制**：
  - 在 Git 仓库的标准初始状态下，正式的 `Failure Mode` 表就只保留了 **1 行** 激活任务（`Textile R692 bubble`）。
  - 如果正式表中堆积了多个不同项目的任务：
    1. 当不带 `--failure_mode` 等参数直接启动时，`pipeline.py` 会遍历其中所有的 Failure Mode 全部跑一遍。
    2. 更严重的是，如果表中存在同名或相似的配置，极易引发下面**第 4 点**剖析的全局上下文错位问题。
  - **最佳操作规范**：正式表保留第 1 行表头，下方**仅保留当前要跑的那 1 行配置**；备用表 `Failure Mode (2)` 保持原样不作破坏，便于随时切换。

---

### 3. 确认逻辑 3：检查正式 `Flow` 表，绝不可用 `Flow (2)` 整体覆盖
* **结论**：**完全正确，绝对不能直接覆盖。**
* **底层证据**：
  通过比对 Git 历史中的两个工作表：
  - **正式 `Flow` 表**：28 行，是经过针对性微调、调试通过的**生产参数表**。
  - **备用 `Flow (2)` 表**：62 行，是历史各种实验、测试阶段混合的**全集并集表**。
  - **参数并不完全一致**，例如以下真实存在的差异：
    - `Fraying`：正式表 `Upsampling` 为 `nan`，而备用表为 `'Yes'`，且 `Upsampling Definition`（上采样尺寸）正式表为 `448.0`，备用表为 `160.0`；
    - `Textile Discoloration`：正式表 `PCA_Alignment` 为 `'No'`，备用表为 `'Yes'`；
    - `Macbook Pit`：正式表 `Defect Detection Setting` 为 `'Backside Pit[Save]'`，备用表为 `nan`；
    - `K11p General`：正式表 `Defect Detection Debug` 为 `nan`，备用表为 `'Yes'`。
  - **结论**：如果粗暴地将 `Flow (2)` 整体覆盖正式 `Flow`，会导致现有调优好的算法流程参数被覆盖为旧版本或实验参数，导致结果异常。因此正确的做法是：**仅当正式表中缺少当前 `Failure Mode` 时，单行查漏补缺。**

---

### 4. 重点确认：同名 `Failure Mode` 上下文覆盖的隐蔽 Bug（100% 存在）

你分析得非常深入，**这个逻辑在当前代码中不仅完全成立，而且是一个典型的上下文解析 Bug**。

#### 源码证据 1：`pipeline.py` 内部构造上下文时的覆盖逻辑
请看 `utils/core/pipeline.py` 第 442-490 行的执行逻辑：
```python
# 1. 这里根据命令行传入的 --product 过滤出了正确的行 (假设只有 Macbook 这行)
if filter_product:
    fm_rows = fm_rows[fm_rows[prod_col].astype(str).apply(_norm) == _norm(filter_product)]
...
fm_row = fm_rows.iloc[0] # 这里拿到的是正确的 Macbook 行！

# 2. 但是到了派发策略上下文时 (Line 485):
# 注意看！这里没有使用已经过滤好的 fm_row，而是重新对原始的未过滤全量表 fm_df 进行了查询：
fm_row_for_context = fm_df[fm_df[fm_col_name] == fm].iloc[0] if not fm_df[fm_df[fm_col_name] == fm].empty else None

strat_product = str(fm_row_for_context.get('Product', '')) # 拿到了未过滤表的第 0 行！
strat_pic_path = str(fm_row_for_context.get('Pic_Path', ''))
```
- **推演过程**：
  如果正式 `Failure Mode` 表中同时存在：
  - 第 0 行：`Product = Silicon_Case`, `Failure Mode = Crop`, `Pic_Path = Pics/silicon`
  - 第 1 行：`Product = Macbook`, `Failure Mode = Crop`, `Pic_Path = Pics/macbook`
  即使命令行传入 `--product Macbook`，外层虽然筛选出了第 1 行，但在第 485 行构造 `strat_context` 时，代码通过 `fm_df[fm_df['Failure Mode'] == 'Crop'].iloc[0]` **直接取出了第 0 行（`Silicon_Case`）的配置**，把 `Pics/silicon` 强行塞给了策略！

#### 源码证据 2：Strategy 内部再次覆盖上下文
不仅 `pipeline.py` 存在这个问题，所有的 Strategy 具体实现类内部也存在这个盲区。例如看 `utils/core/strategies/crop_strategy.py` 第 25-30 行：
```python
# Strategy 内部又重新从 ConfigManager 读取了 Failure Mode 表，并且也是无脑取 iloc[0]
fm_df = cfg_mgr.get_sheet('Failure Mode')
fm_row = fm_df[fm_df['Failure Mode'] == self.fm].iloc[0] if not fm_df.empty else None

self.product = str(fm_row['Product']) if fm_row is not None else self.product
self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
```
这意味着：即便外部传入了正确的 `context`，Strategy 内部又会去查询 `Failure Mode == 'Crop'` 的第一行，从而**再次强制被 `Silicon_Case` 覆盖**。

#### 源码证据 3：`ConfigManager` 的单 Key 字典覆盖
在 `utils/config/config_manager.py` 第 148 行：
```python
self.flow_configs[str(fm)] = FlowConfig(...)
```
在字典中，`Flow` 表也是完全以 `Failure Mode` 单字符串作为 Key 的，根本不支持 `(Product, Generation, Failure Mode)` 复合键。两个同名 `Crop` 会在启动初始化时直接发生字典覆盖。

---

### 五、 针对该问题的日常避坑规范与代码改进建议

基于以上源码核实，在当前代码架构下，日常运行务必遵循以下**操作规范**：

1. **单任务模式（推荐操作）**：
   在正式的 `Failure Mode` 表中，**始终保持只有 1 行待运行的有效任务**。切换任务时，先清空历史行，再从 `Failure Mode (2)` 中拷入目标行，彻底杜绝多行同名冲突。
2. **唯一命名规范（命名防御）**：
   在库表 `Failure Mode (2)` 中，你可以看到原作者也曾踩过这个坑，因此部分用例采用了带前缀的独立命名（例如 `Crop_Macbook Pit`、`Dino Crop_Macbook Pit`、`R187_Textile Staining`），通过赋予唯一的 `Failure Mode` 名字来绕开 `iloc[0]` 匹配 Bug。
3. **彻底根治代码 Bug 的建议**：
   若后续需要向团队或老板汇报重构建议，最优雅的改法是：
   - 将 `pipeline.py` 第 485 行的 `fm_row_for_context = fm_df[...].iloc[0]` 修正为直接使用已完成 `--product` 过滤的 `fm_row`；
   - 在各 Strategy 的 `execute()` 中，优先使用传入的 `self.context['product']` 和 `self.context['download_path']`，而不是重新从 `fm_df` 取 `iloc[0]`。