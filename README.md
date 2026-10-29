# Flight Path Anomaly Detection

## 个体和群体交叉感知注意力网络用于飞行器异常轨迹预测

这个项目实现了一个基于深度学习的飞行器轨迹异常检测系统，结合了个体和群体行为分析。

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
