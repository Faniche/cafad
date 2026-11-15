# 初步清洗数据的 Sun 阶段识别

本入口只处理**初步清洗后、尚未进行新异常注入**的 CSV。它不加噪、不标准化模型特征、不建立正常参考组或训练模型；输出阶段及点到片段的映射，供后续实验使用。

## 方法与来源

论文：Sun, Ellerbroek, Hoekstra (2017), *Flight Extraction and Phase Identification for Large Automatic Dependent Surveillance–Broadcast Datasets*, DOI [10.2514/1.I010520](https://doi.org/10.2514/1.I010520)。

对照来源：[作者 `flightphase.py`](https://github.com/junzis/flight-data-processor/blob/master/flightphase.py)。`sun_phase.py` 独立组织 NumPy/SciPy 实现，沿用作者的隶属函数参数、离散论域、五条模糊规则、样条平滑和 largest-of-maximum 解模糊。没有直接复制整个上游模块，也不依赖 MongoDB、绘图库或 scikit-fuzzy。

输出类型为 `GND` 地面、`CL` 爬升、`DE` 下降、`CR` 巡航、`LVL` 低高度平飞；无有效规则支持时保留 `NA`。方法默认按相对航班起点的 60 秒窗口分类，再合并**连续且同类型**的观测为一个 segment。同一阶段后续再次出现，会形成另一个 segment；不额外发明起飞、着陆或转弯类型。

相对作者旧代码的工程修正：包含最后一个时间窗；不足四个不同时间点时降低样条阶数，单点直接分类；一次处理多个窗口以减少 Python 循环。重复时间点只在分类临时数组中做均值，输出保留每个原始观测。该识别方法是离线方法，样条可能使用整个航班信息。

## 资源配置

根目录 `.env`：

```dotenv
nproc=56
data_proc_usage=80%
```

`Env.data_process_workers()` 用十进制计算后向下取整：`floor(56 * 0.8) = 44`。`nproc` 表示物理核心预算，不会因每核两线程再乘二。可以把使用比例改为 `90%` 得到 50 个进程。进程环境变量覆盖文件同名配置；`NPROC` 与 `DATA_PROC_USAGE` 也支持。

`.env` 按现有 Git 规则忽略，`.env.example` 纳入版本管理。新服务器首次使用时执行 `cp .env.example .env`。管理类位于 `src/utils/env.py`，不需要 python-dotenv。

每个工作进程负责一条完整航班，进程池空闲后自动领取下一任务。`ProcessPoolExecutor` 的多进程队列自带 put/get 同步锁，程序没有让子进程竞争一个普通 Python 列表或共用 CSV 文件句柄。主进程最多保留 `2 * workers` 个在途任务；子进程写各自临时结果，汇总文件只由主进程写。

主入口在导入 NumPy/SciPy 前将 OMP/OpenBLAS/MKL/NumExpr 等内部线程数设为 1，防止 44 个进程各再启动多条数值计算线程。实际利用率还受 CSV 解析、磁盘和航班数量限制；此处没有宣称已在 56 核服务器测得特定吞吐。

## CSV 输入与航班识别

默认识别以下字段，存在歧义时应使用列名参数显式指定：

| 用途 | 常见列名 |
|---|---|
| ICAO | `icao号`、`icao`、`icao24`、`aircraft_id` |
| 呼号 | `call号`、`call`、`callsign` |
| 时间 | `时间`、`time`、`timestamp` |
| 高度 | `水平高度`、`altitude`、`horizontal_altitude` |
| 速度 | `速度`、`speed`、`groundspeed` |

ICAO、CALL 按字符串读取，保留前导零。当前航班/session 定义是：同 ICAO、同 CALL 内按时间排序，邻接观测间隔超过 `--flight-gap-seconds` 时拆开，默认 1800 秒。这是本工程明确的 session 规则，**不是实现了原论文的完整航班提取聚类，也不把 ICAO 直接等同一次真实任务**。

CSV 可以乱序。主进程分块读取，并按 ICAO/CALL 哈希写入临时桶；同一身份跨块的观测始终进入同一个桶。随后逐桶排序、生成完整航班，向进程池流水提交。因此不要求一次把全 CSV 读入内存，也不会把一个跨块航班错误拆成多个任务。读取/分桶先完成，之后飞行任务的生成、计算和结果收集交错进行。临时桶位于新输出目录下，处理结束自动清理，运行时需要临时磁盘空间。

输入应已完成基本清洗。若必要字段缺失、数值非有限或时间无法解析，本入口报错并保存失败诊断，不静默丢点。相同时间的多条记录仍逐条保留。若一条已经独立的连续轨迹在 Sun 识别中出现数值 `ValueError`，不会中断其他轨迹：该轨迹不生成阶段／片段，而写入不可识别报告及逐点行号映射。每个点的 `source_row_index` 是输入 CSV 的零基数据行顺序，不含表头。

### 单位必须显式提供

Sun 原规则使用高度 ft、速度 kt、升降率 ft/min。调用必须提供 `--altitude-unit` 和 `--speed-unit`，不能从中文列名或值域猜单位。

可选实测升降率通过 `--vertical-rate-column` 和 `--vertical-rate-unit` 一起指定；未提供时，以实际时间间隔对高度计算梯度并换算为 ft/min。选择的高度字段和原始单位应与数据字典一致。

日期字符串支持普通小数秒和旧文件的空格小数秒；无时区日期按 UTC 计算存储，这不认证来源的真实时区。数值时间戳必须显式指定 `--timestamp-unit s|ms|us|ns`，整数纳秒不经过浮点转换。

## 服务器调用

正式流程先使用 `preprocess` 写出规范观测，再让 `segment` 读取其成功发布的 manifest。下例**仅假设已经确认高度单位为米、速度为米/秒、航向为度**；实际单位不同时替换参数：

```bash
python main.py -m preprocess \
  --format adsb-csv --input data/raw/d1.csv \
  --altitude-unit m --speed-unit m/s --heading-unit deg
python main.py -m segment \
  --input data/0_preprocess/preprocessed_data_d1/<run_id>#<timestamp>#<seed>/manifest.json
```

`src/main.py` 已移除。运行 ID 由 `data/record_counter.csv` 自动递增，输出固定写入 `data/<顺序号>_<module>/<dataset>/<run_id>#<timestamp>#<seed>/`；下游只接受已发布且输出哈希通过复核的 manifest。

可调参数包括 `--chunk-rows`（默认 200000）、`--buckets`（128）、`--window-seconds`（60）、`--flight-gap-seconds`（1800），以及 `--icao-column`、`--call-column`、`--time-column`、`--altitude-column`、`--speed-column`。

### 日志控制

根入口在加载 `.env` 后写入项目根目录的 UTF-8 日志 `logs/run-YYYYMMDD-HHMMSS-PID.log`，路径不受当前工作目录或数据输出目录影响。

- 默认只写文件；传入全局 `--console` 才同时输出到命令行。
- 该开关不影响文件日志；`.env` 缺失、输入错误和运行异常也会记录。

格式统一为 `年月日 时分秒 py文件:行号 [级别]: 内容`，例如：

```text
2026-10-29 16:30:00 main.py:42 [INFO]: CPU 预算：nproc=56，data_proc_usage=80%，向下取整后 44 个进程
2026-10-29 16:30:03 phase_pipeline.py:340 [INFO]: 已处理 1 个航班 ...
```

使用 `%Y-%m-%d %H:%M:%S`，不附带毫秒。异常堆栈和多行消息的每一行也带同样的时间、来源文件和级别前缀。多个子进程不共写日志，由主进程汇报进度；结果目录的 `configuration.json`、`summary.json` 和失败时的 `failure.json` 保存实际日志路径。直接调用处理函数且未传 logger 时，也默认创建根目录文件日志，并关闭命令行输出。

增加桶数可以减小平均单桶排序内存，增加 chunk_rows 可以减少分块开销；单一身份非常大时，完整航班仍需要放入一个工作进程。吞吐需要在目标服务器测量，不建议直接按逻辑线程数翻倍创建进程。

## 输出与进度

| 文件 | 内容 |
|---|---|
| `configuration.json` | 输入路径与元数据、列映射、单位、航班/方法参数及资源设置 |
| 项目根目录 `logs/sun_phase_*.log` | 启动与 CPU 配置、分块读取进度、每个航班的点数/片段数/阶段数量/进程 PID、最终统计和异常；不再写结果目录的 `processing.log` |
| `flights.csv` | 每航班点数、segment 数、阶段类型数、时间范围、重复时间点数等 |
| `segments.csv` | 每片段唯一 ID、所属航班、阶段、点数、起止时间及首尾原始行号 |
| `points.csv` | 可识别输入点到 flight/segment/phase 的精确映射，不重复复制全部物理特征 |
| `unrecognized_flights.csv` | 每条不可识别轨迹的 ICAO、CALL、点数、时长、首尾时空位置和异常原因 |
| `unrecognized_points.csv` | 不可识别轨迹对应的全部 `source_row_index` 与 flight ID |
| `summary.json` | 可识别／不可识别航班及点数、总片段数；航班点数、每航班片段数、片段点数的最小/最大/平均/中位数；各阶段数量 |
| `failure.json` | 仅在失败时出现，当前处理不能视为完整结果 |

输出按任务完成顺序写入，行顺序可能随进程调度改变；航班 ID、片段 ID、成员和汇总不依赖完成顺序。用 `source_row_index` 恢复输入顺序。首尾行号不表示输入 CSV 中一段连续区间，完整成员以 `points.csv` 为准。

完成前检查 `输入点数 = 可识别点数 + 不可识别点数`，以及 `可识别点数 = 各可识别航班点数之和 = 各片段点数之和`。未加噪，输出没有新增正常/异常标签。

## 检查

```bash
python -B -m unittest discover -s tests -v
```

测试使用人工小样本，覆盖五条模糊规则、最后窗口、短航班、真实时间升降率、纳秒精度、资源预算、两个真实 spawn 子进程、乱序跨块航班、重复时间点、统计与主入口调用。没有在本地执行完整原始 CSV 或服务器性能实验。

日志改动后共 31 项检查：30 项通过，1 项旧 DBSCAN 依赖检查跳过。新增检查覆盖控制台开关、根目录日志、跨工作目录启动、提前配置失败、重复初始化、中文、多行消息及堆栈的格式。
