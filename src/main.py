"""
Training script for Individual-Group Cross-Perception Attention Network (IGCAN)
"""

import argparse
from pathlib import Path
import yaml
import logging
import sys
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.attention import IGCAN
from src.utils.trajectory_dataset import (
    generate_random_trajectories,
    pad_trajectories,
    generate_batch_data,
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test_trajectory_generation(logger: logging.Logger):
    """测试轨迹生成功能 (使用logging.debug替代print)"""
    logger.debug("\n" + "=" * 70)
    logger.debug("测试轨迹生成功能")
    logger.debug("=" * 70)

    # 生成100条随机轨迹
    trajectories, lengths = generate_random_trajectories(
        num_trajectories=100, min_length=10, max_length=1000
    )

    logger.debug(f"\n✓ 生成了 {len(trajectories)} 条轨迹")
    logger.debug(f"✓ 长度范围: {min(lengths)} - {max(lengths)}")
    logger.debug(f"✓ 平均长度: {sum(lengths) / len(lengths):.1f}")
    logger.debug(f"\n示例 - 前5条轨迹的形状:")
    for i in range(5):
        logger.debug(f"  轨迹 {i}: {trajectories[i].shape} (长度: {lengths[i]})")

    # 测试填充功能
    logger.debug("\n测试轨迹填充功能...")
    padded, mask = pad_trajectories(trajectories[:10])
    logger.debug(f"✓ 填充后形状: {padded.shape}")
    logger.debug(f"✓ 掩码形状: {mask.shape}")

    # 测试批次生成
    logger.debug("\n测试批次生成功能...")
    batch_padded, batch_mask, batch_lengths = generate_batch_data(batch_size=16)
    logger.debug(f"✓ 批次轨迹形状: {batch_padded.shape}")
    logger.debug(f"✓ 批次掩码形状: {batch_mask.shape}")

    logger.debug("\n✓ 轨迹生成测试通过！")
    logger.debug("=" * 70 + "\n")

    return batch_padded, batch_mask, batch_lengths


def test_igcan_model(logger: logging.Logger):
    """测试IGCAN模型 (使用logging.debug替代print)"""
    logger.debug("\n" + "=" * 70)
    logger.debug("测试IGCAN模型")
    logger.debug("=" * 70)

    # 创建模型
    model = IGCAN(
        input_dim=6,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=1000,
        similarity_threshold=0.7,
        top_k=10,
        hidden_dim=256,
    )

    logger.debug(f"\n✓ IGCAN模型创建成功")
    logger.debug(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 生成测试数据
    logger.debug("\n生成测试数据...")
    batch_padded, batch_mask, batch_lengths = generate_batch_data(
        batch_size=16, min_length=10, max_length=1000, seed=42
    )

    logger.debug(f"✓ 数据形状: {batch_padded.shape}")
    logger.debug(f"✓ 掩码形状: {batch_mask.shape}")

    # 前向传播
    logger.debug("\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(batch_padded, batch_mask)

    # 显示输出
    logger.debug(f"\n✓ 前向传播成功！")
    logger.debug(f"\n输出结果:")
    logger.debug(f"  异常概率形状: {outputs['anomaly_probs'].shape}")
    logger.debug(f"  注意力权重形状: {outputs['attention_weights'].shape}")
    logger.debug(f"  个体特征形状: {outputs['individual_features'].shape}")
    logger.debug(f"  群体特征形状: {outputs['group_features'].shape}")
    logger.debug(f"  轨迹嵌入形状: {outputs['embeddings'].shape}")
    logger.debug(f"  相似度矩阵形状: {outputs['similarity_matrix'].shape}")

    # 分析异常检测结果
    logger.debug(f"\n异常检测结果分析:")
    anomaly_probs = outputs["anomaly_probs"].squeeze()
    logger.debug(f"  最小异常概率: {anomaly_probs.min():.4f}")
    logger.debug(f"  最大异常概率: {anomaly_probs.max():.4f}")
    logger.debug(f"  平均异常概率: {anomaly_probs.mean():.4f}")

    # 分析分组信息
    logger.debug(f"\n分组信息:")
    groups = outputs["groups"]
    logger.debug(f"  总共 {len(groups)} 个轨迹")
    logger.debug(f"  平均群体大小: {sum(len(g) for g in groups) / len(groups):.1f}")

    logger.debug("\n✓ IGCAN模型测试完成！")
    logger.debug("=" * 70 + "\n")

    return model, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Train Individual-Group Cross-Perception Attention Network"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "test", "eval"],
        help="Running mode: train, test, or eval",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting in {args.mode} mode")

    if args.mode == "test":
        # Test trajectory generation and IGCAN model
        logger.info("Running trajectory generation test...")
        batch_padded, batch_mask, batch_lengths = test_trajectory_generation(logger)

        logger.info("Running IGCAN model test...")
        model, outputs = test_igcan_model(logger)
        logger.info("All tests completed successfully!")

    elif args.mode == "train":
        # Load configuration
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            return

        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

        # TODO: Implement training loop
        logger.info("Training mode - Implementation pending")

    elif args.mode == "eval":
        # TODO: Implement evaluation
        logger.info("Evaluation mode - Implementation pending")


if __name__ == "__main__":
    main()
