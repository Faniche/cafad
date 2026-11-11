# Flight Path Anomaly Detection

## CAFAD：阶段片段异常检测（开发中）

当前正式入口只有根目录 `main.py`，一次只运行一个模块：

```text
python main.py [全局参数] -m 模块名 [模块参数]
```

已实现第一阶段数据准备：标准 ADS-B CSV 的清洗与单位统一（`preprocess`）、航班级 train／val／test 划分及正常／异常注入计划（`split`）、单副本事件化异常注入（`inject`），以及 Sun 阶段片段生成（`segment`）。事件在整条合格航班内采样；注入后必须对最终观测再运行 `segment`，生成真正供检测使用的片段。`split` 不修改观测；`inject` 严格按计划生成每条航班唯一的最终观测版本，只标记实际改变的物理记录。片段分组、MoE 训练、阈值校准、检测与评价仍待开发；下文旧的 IGCAN 模型描述和命令不是当前正式流程。

最小示例（物理单位必须按数据字典确认）：

```powershell
python main.py -m preprocess --format adsb-csv --input data/raw/preprocessed_data_d1.csv --altitude-unit m --speed-unit m/s --heading-unit deg
python main.py --seed 42 -m split --input data/0_preprocess/preprocessed_data_d1/<run_id>#<timestamp>#42/manifest.json
python main.py --seed 42 -m inject --input data/0_preprocess/preprocessed_data_d1/<run_id>#<timestamp>#42/manifest.json --split-manifest data/1_split/preprocessed_data_d1/<run_id>#<timestamp>#42/manifest.json --protocol configs/injection_protocol_example.yaml
python main.py -m segment --input data/2_inject/preprocessed_data_d1/<run_id>#<timestamp>#42/manifest.json
```

示例协议的最长事件为 30 点，且事件两端各保留 5 个正常上下文点，因此根 `.env` 的 `injection_minimum_trajectory_points=40`。不足 40 点的航班保留在既定分区中，但不参与异常注入并作为正常轨迹输出。不要把参考论文中某次实验使用的 600 点事件长度或 30／60 点检测窗口当作本项目的统一最低点数。

每步只读取已发布且哈希校验通过的上游 `manifest.json`。数据集身份就是 preprocess 输入文件去掉扩展名后的完整文件名（如 `preprocessed_data_d1.csv → preprocessed_data_d1`），后续模块从清单继承；运行编号由 `data/record_counter.csv` 自动分配并记录到根 `run_registry.json`。成功输出固定在 `data/<顺序号>_<module>/<dataset>/<run_id>#<timestamp>#<seed>/`，例如 `data/0_preprocess/...`、`data/3_segment/...`。详情见 [项目管理文档](docs/proj_mgnt.md) 与 [方法讨论稿](docs/CAFAD_第5和第6步方法完善_讨论稿.md)。

服务器端 Web 控制台可通过 `python scripts/run_web_console.py` 启动；监听地址默认为 `0.0.0.0`，唯一监听端口必须由根 `.env` 的 `web_console_port` 设置，且要求同一文件配置 `web_console_token`。访问根页面先显示登录页：服务端只比对 Token 的 SHA-256 摘要，成功后以 8 小时的 `HttpOnly` 同源会话 Cookie 保护模块配置、文件浏览、任务和日志接口。全局和各模块表单配置按会话保存在服务器内存，切换侧栏或刷新页面可恢复；退出登录或服务重启会清除。页面左侧分为总体设置与各业务模块，模块参数直接取自其 `api.py`；网页只选择最近的上游 manifest，服务端从其依赖链推导更早输入并由根入口重新核验。首次打开时“显示控制台日志”默认勾选。其他 Path 参数可打开受限资源浏览器，只列出当前用户目录或项目目录的非隐藏名称，不能下载文件或越出允许根目录。控制台只调用本文件所述的根 `main.py`，不会直接执行任意 shell 文本。公网访问还需另行配置域名反向代理和 TLS；局域网 HTTP 登录会传输 Token，限可信网络使用。

## 项目结构

```
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   └── processed/         # 预处理后的数据
├── src/                   # 源代码
│   ├── models/           # 模型定义
│   ├── data/             # 数据处理
│   └── utils/            # 工具函数
├── notebooks/            # Jupyter notebooks
├── experiments/          # 实验结果和配置
├── docs/                # 文档
└── tests/               # 测试文件
```

## 核心功能

### 1. 个体感知模块
- 使用LSTM/GRU处理单个飞行器的时序轨迹数据
- 提取个体运动模式特征
- 检测个体轨迹异常

### 2. 群体感知模块
- 使用图神经网络建模飞行器间的空间关系
- 捕获群体行为模式
- 分析集体异常行为

### 3. 交叉感知注意力网络
- 融合个体和群体特征
- 多头注意力机制实现信息交互
- 提高异常检测精度

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 数据预处理：
```bash
python src/data/preprocess.py --input data/raw --output data/processed
```

2. 训练模型：
```bash
python src/models/train.py --config experiments/config.yaml
```

3. 评估模型：
```bash
python src/models/evaluate.py --model checkpoints/best_model.pth
```

## 论文参考

本项目基于以下研究方向：
- 时序数据异常检测
- 注意力机制在轨迹分析中的应用
- 图神经网络用于群体行为建模

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT License
