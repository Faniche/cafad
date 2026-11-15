# UMN 转为 ADS-B 九列格式

统一由 `src/main.py` 组合命令行参数和 `UMNOptions`，调用 `umn_data_proc.run()`。处理范围是 UMN CSV，未读取或修改 `preprocessed_data_d1.csv`。目标表头来自提供的截图：

```text
纬度,经度,水平高度,气压高度,航向,速度,call号,icao号,时间
```

## 运行

在项目根目录使用项目 venv：

```powershell
.\.venv\Scripts\python.exe -B -m src.main --task umn --no-console-log
```

默认读取 `data/processed/umn/csv/`，输出到新的 `data/processed/umn/adsb/`。也可显式指定：

```powershell
.\.venv\Scripts\python.exe -B -m src.main --task umn `
  --input data/processed/umn/csv `
  --output data/processed/umn/adsb_v2 `
  --chunk-rows 100000 `
  --umn-baro-tolerance-seconds 1 `
  --umn-min-track-speed 0.5 `
  --console-log
```

Linux 服务器可使用同一入口：`python -B -m src.main --task umn --no-console-log`。输出目录必须尚不存在，避免覆盖先前结果。已有 Sun 阶段调用保留原样；也可显式使用 `--task sun-phase`。

## 字段与单位

| 输出 | 原始平面字段 | Vireo 原始异步字段 | Baldr 官方同步 EKF 字段 |
|---|---|---|---|
| 纬度 | lat | gps.latitude | dtc.ekf.lat |
| 经度 | lon | gps.longitude | dtc.ekf.lon |
| 水平高度 | alt | gps.altitude_gps | dtc.ekf.alt_ekf |
| 气压高度 | h | air.altitude_baro | dtc.air.pitot.alt_baro |
| 时间 | time | gps.time | dtc.time |
| 速度、航向的北/东速度分量 | vn, ve | gps.Vn, gps.Ve | dtc.ekf.Vn, dtc.ekf.Ve |

- 经纬度保持度，高度保持米。速度取 `hypot(Vn, Ve)`，为水平地速 m/s，不用指示空速 IAS 替代。
- 航向列实际表示地面航迹角：`degrees(atan2(Ve, Vn)) mod 360`，从正北顺时针。地速小于 `--umn-min-track-speed`（默认 0.5 m/s）时输出 `-1`；这是一项可调预处理规则，不是飞行异常阈值。不会把机体 Euler yaw 当成地面航迹角。
- 时间按用户选择保留每条记录的相对秒数；不会制造采集日期，也不能据此判断不同飞行是否同时发生。后续使用 Sun 入口时需显式提供 `--timestamp-unit s`，内部 epoch 表示仅用于数值计算，不认证真实采集日期。
- `icao号` 使用 `UMN_` 加该记录的原始 UUID，避免不同任务混为一条轨迹或与真实六位 ICAO 地址冲突；它是任务标识，不是真实 ICAO 注册地址。`call号` 取文件名中的记录名。
- Baldr 的部分记录只有官方同步 EKF 估计量，来源在每文件 JSON 中明确记录；不会标成未经处理的 GPS 定位。
- **统一列格式不改变垂直基准**：旧字典的 `h`、Baldr 的 `alt_baro` 部分标为 AGL，不能直接声称已换算成 ADS-B 标准气压高度。每条记录保留参考说明，代码不推测机场标高或海平面气压。

## 异步数据、清洗和溯源

Vireo CSV 中包含多种传感器时间轴的并集。只取存在 `gps.time` 的 GPS 包作为位置候选，不把 IMU/气压包行当成额外 GPS 定位。

气压高度使用最近一个**过去或同一时刻**的原始气压观测，默认最大年龄 1 秒；不借用未来值、不插值、不无限前向填充。超过容差的观测默认排除，使用 `--umn-allow-missing-baro` 可保留该 GPS 点并把气压高度留空。该开关会使输出不再满足后续严格阶段入口的所有有限数值要求，需要后续处理缺失值。

默认排除必要 GPS 运动字段非有限、时间非有限、坐标越界以及 UMN 初始化占位 `(0, 0)`；最后一项可以用 `--no-umn-reject-zero-position` 关闭。不裁剪合法负高度，不按速度大小删除有限观测，不对重复时间点去重。每个文件按时间和原始行号稳定排序；同一时间的多个原始点仍保留。无有效观测时生成仅表头的 CSV 并记录 WARNING，不伪造数据。

`provenance/*.rows.csv` 保存输出行号、原始 CSV 零基行号、气压原始行号和气压观测年龄；JSON 保存字段映射、源文件大小/mtime、筛除统计、身份、参数、PID 和时间语义。各项不合格原因可能重叠，`rejected_rows` 统计并集。

## 进程、队列与日志

沿用 `src/utils/env.py` 的 `Env.load()` 和 `Env.data_process_workers()`：根目录 `.env` 中 `nproc=56`、`data_proc_usage=80%` 得到 `floor(56*0.8)=44` 个工作进程预算。实际进程数是预算和文件数的较小值；不按超线程再翻倍，不以本地逻辑 CPU 数擅自替换服务器预算。

在导入数值库/启动子进程前将 OMP、OpenBLAS、MKL、NumExpr 等内部线程设为 1。操作系统调度工作进程；没有硬编码 CPU 亲和性，也不承诺磁盘受限任务一定达到 80% CPU 利用率。

`run()` 枚举文件，只将路径任务入队。子进程循环调用 `multiprocessing.Queue.get()`，每次独占一个文件，完成后领取下一项。该队列内部已有进程间读写同步锁，无需再给取任务加外部互斥锁。主进程在所有任务之后放入每进程一个 `None` 哨兵，不通过不可靠的 `empty()`/`qsize()` 决定退出；它会收齐文件结果并等待所有进程结束。

文件任务失败时记录完整异常并继续取其他任务。若某个子进程异常退出，主进程检测退出码，结束剩余进程并写 `failure.json`，不会永远等待丢失结果。任一文件失败时不发布不完整的合并 CSV。各进程写自己的文件，不共享 CSV 写句柄。

日志沿用项目工具，必须写入项目根目录 `logs/umn_data_proc_*.log`。主入口的 `--console-log` / `--no-console-log` 控制是否同步输出命令行，不影响文件日志。子进程只发送 `QueueHandler` 记录，主进程 `QueueListener` 统一处理文件和控制台，避免多进程直接写同一个日志文件。

```text
2026-10-30 00:00:00 umn_data_proc.py:123 [INFO]: 领取文件任务：...
```

异常和多行消息每一行均带 `年月日 时分秒 文件.py:lineno [LEVEL]: 内容`。新处理代码不使用 `print()`；参数错误与帮助信息也走项目日志。关闭命令行日志时，它们只进入文件。

## 输出

- `files/*.csv`：每个输入文件一个九列 CSV，UTF-8 BOM。
- `preprocessed_data_umn.csv`：主进程在全部任务成功后，按输入文件名顺序合并；只有一份表头。`--no-umn-merge` 可关闭。
- `provenance/*.json`、`provenance/*.rows.csv`：来源、字段、处理规则和逐点回溯。
- `configuration.json`、`summary.json`：输入清单、资源预算、工作进程、统计及日志路径。
- `failure.json`：仅失败时出现，不能把该输出目录当作完整结果。

运行检查：`.venv\Scripts\python.exe -B -m unittest discover -s tests -v`。检查使用人工数据，覆盖真实 spawn 队列进程、恰好一次取任务、失败后继续取任务、无控制台输出时的日志、同源行映射、相对秒、航迹角、IAS/地速区别、异步跨块因果对齐和官方 EKF 字段。真实性能需在目标服务器测量。

2026-10-30 实际验证：合并测试 39 项，38 项通过，1 项旧 DBSCAN 对照检查因可选 scikit-learn 未安装跳过。另从 UMN 目录取 Baldr Flight 1、Baldr Flight 12、Loki Flight 07、Vireo Flight 1 四份真实 CSV，以原文件的只读用途硬链接构成抽样目录，通过主入口处理；`.env` 预算为 44，按文件数实际启动 4 个进程，输出 288,058 行。九列表头、有限数值、角度范围、4个独立任务标识、相对秒及25行日志格式检查通过。抽样结果位置记在项目 `tmp/umn_adsb_latest_smoke.txt`；未读取 ADS-B 大文件，未执行服务器全量性能测试。
