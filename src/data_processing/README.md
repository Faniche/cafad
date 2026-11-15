# 数据处理代码筛选与迁入

本目录从当前项目旧代码和只读来源 `Z:\plane` 中筛选、抽取可复用的数据处理部分。原文件保留；未处理或改写任何真实数据。迁入日期：2026-10-29。

`sun_phase.py` 实现 Sun 模糊阶段识别，`phase_pipeline.py` 是被正式 `segment` 模块复用的清洗后观测多进程处理核。正式入口只有根目录 `main.py`；服务器资源配置和正式运行方法见 [Sun 阶段处理说明](SUN_PHASE_PROCESSING.md)。

日志统一写入项目根目录 `logs/`，默认不输出命令行；正式入口传入 `--console` 才镜像。格式为 `YYYY-MM-DD HH:MM:SS+时区 文件.py:行号 LEVEL 内容`，配置错误和异常堆栈也保留。

UMN 九列规范化仍是 `umn_data_proc.py` 的可复用函数，尚未接入正式 `preprocess` 模块；不能再通过旧 `--task umn` 命令调用。字段来源、气压高度基准限制及参数见 [UMN 处理说明](UMN_DATA_PROCESSING.md)。

## 可直接复用的工具

| 文件 | 用途 | 使用边界 |
|---|---|---|
| `noise_provenance.py` | 读取旧注入记录、规范时间、验证位置和时间、分块生成精确位置标签 | 标签绑定记录生成时的原 ICAO 组顺序；不能先去重、重排再套用原位置；时间戳仅用于核验 |
| `sampling.py` | 根据均匀物理时间选择最近的真实观测，距离并列取较早记录，选择结果去重 | 输入是非空、严格递增的一维时间数组，`count` 为正整数；请求数不少于观测数时返回全部观测 |
| `anchors.py` | 航向差环绕、锚点距离、稀疏精确距离 DBSCAN | 锚点格式为 `[seconds, x_metres, y_metres, heading_radians]`；保留旧尺度 `10 s / 10000 m / pi`；仍按锚点计算密度 |

`noise_provenance.py` 为字节一致复制。采样和锚点函数从 `Z:\plane\cafad\data.py` 抽取，仅公开函数命名并延迟导入 DBSCAN 依赖，计算规则沿用原实现。

```python
from src.data_processing.sampling import sample_observed_times
from src.data_processing.anchors import anchor_distance, cluster_anchors
from src.data_processing.noise_provenance import (
    load_records, normalize_times, validate_records, label_group,
)

# 均匀时间采样返回实际记录的下标，不插值生成观测。
indices = sample_observed_times(times, count=8)

# 历史注入记录：必须提供原始记录顺序对应的组长度和时间。
records = validate_records(load_records(records_path), group_lengths=group_lengths)
result = label_group(icao, normalize_times(group_times), 0, records,
                     total_length=len(group_times))
point_labels = result["exact_mask"]
if result["position_timestamp_mismatch"]:
    raise ValueError("注入记录与当前数据未对齐，应停止并核查来源")
```

位置标签只说明记录了合成注入，不证明未注入观测不存在自然异常。旧记录中的 ICAO 必须保持字符串，包括前导零；不要静默修复科学计数法或小数形式的损坏 ID。

## 隔离保存的历史参考

`legacy_plane/` 内三个模块按原字节复制，保留相对导入关系，便于查阅和后续抽取。它们不会被包初始化自动调用。

| 文件 | 仍有用的部分 | 与当前讨论方案的区别 |
|---|---|---|
| `data.py` | 字段映射、时间采样、训练分区拟合、参考来源检查、缓存完整性校验、`PreparedData` | 阶段方法仅为升降率阈值近似，**不是 Sun 方法**；`_context_peers` 本身不按正常标签筛选；使用簇中心检索和目标相关参考集合 |
| `synthetic_five_data.py` | 哈希种子、五类扰动函数、注入账本、来源哈希与同源划分 | 默认同时保留 clean/corrupted 两份样本，且从加噪前确定阶段；不符合当前“一个观测版本、加噪后分段”的主流程 |
| `motion_inputs.py` | 真实时间差分、环绕航向变化、训练参考标定与稳健缩放 | 将原六个物理输入变为运动一致性残差；属于旧实验输入适配，不能静默用于原六通道模型 |

历史代码还包含旧默认值和文件输出函数。这里只迁入实现，没有执行这些生成器，也没有宣布它们符合当前方法或恢复了旧实验结果。具体来源、函数及 SHA256 见 `provenance.json`。

## 未迁入的旧代码

本地 `data/` 中的 39 个 Python 文件均做了静态检查和归类；五个加噪目录包含大量相近脚本，没有全部重复复制。

| 原文件/类别 | 不作为新实现迁入的原因 |
|---|---|
| `data/data/code_noise_new_*/code/data_preprocessing.py` | 中心窗口包含被检点的 3-sigma 规则需重新设计；Kalman 固定时间步不适合直接处理非均匀采样 |
| `data/data/code_noise_new_*/code/label_engineering.py` | 依赖时间戳匹配标签，重复时间可能扩大注入位置；存在全量或分 ICAO 拟合缩放、标签与排序对齐等风险 |
| `data/data/code_noise_new_*/code/noise_adder.py` | NumPy/TensorFlow 双随机源、未固定原始种子，且类型/强度账本不完整；扰动机制优先参考隔离的五情景生成器 |
| 两套 `code_uavDATA_process*/uav_dataprocess.py` | 全量或分 ICAO 拟合缩放；序列标签取窗口最后一点，与当前任一点异常则片段异常的定义不同 |
| `statistic.py`、`excel.py`、`test.py`、绘图脚本 | 部分把 ICAO 组当作航班、截断时间精度或使用固定路径；主要属于临时统计/检查/绘图，不是新处理主链 |

上述历史迁入部分没有补定新的清洗规则、正常组覆盖率、固定组记忆或注入比例。新增 Sun 入口仅负责清洗后、加噪前的阶段识别；后续异常生成与训练分组仍需单独实现。

## 依赖与验证

从项目根目录安装所需依赖：

```powershell
python -m pip install -r src/data_processing/requirements.txt
```

从项目根目录运行小型机制检查（使用人工测试数据，不读取真实数据集）：

```powershell
python -B -m unittest discover -s tests -v
```

检查覆盖物理时间采样、航向环绕、精确标签与重复时间、分块一致性、受限记录读取、旧阶段方法边界等。DBSCAN 与稠密精确距离的对照检查在安装 SciPy/scikit-learn 后运行；未安装时明确跳过。复制文件另校验来源哈希，抽取函数检查语法与行为。

2026-10-29 实际核验：项目 `.venv` 中 13 项检查有 12 项通过，1 项 DBSCAN 对照检查因未安装 scikit-learn 跳过；10 个 Python 文件语法解析通过。7 个迁入/抽取文件的来源与目标哈希已核对，本地 39 个候选原文件哈希保持不变。未运行历史生成器、真实数据处理或训练。

同日新增 Sun 阶段入口后，合并检查为 27 项：26 项通过，仍只有上述旧 DBSCAN 对照检查跳过。Sun 五类规则、两个真实子进程、主入口、单位转换以及不同分块/进程数下成员一致性检查均通过；测试没有处理真实整份数据集。
