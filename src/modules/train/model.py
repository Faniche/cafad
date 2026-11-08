"""model.py | 2026-11-08 | 单专家双分支编码、记忆与分类网络 | Codex."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ExpertModel(nn.Module):
    """同一专家内目标与参考共享 Level-1 编码器，专家之间由调用方独立实例化。"""

    def __init__(self, d_model: int, branch_layers: int, branch_heads: int, group_layers: int,
                 group_heads: int, pool_heads: int, memory_slots: int, dropout: float) -> None:
        super().__init__()
        width = 2 * d_model
        if d_model % 2 or d_model % branch_heads or width % group_heads or width % pool_heads:
            raise ValueError('d_model 与各注意力头数不整除，或 d_model 不是偶数')
        self.d_model, self.width = d_model, width
        self.space_embedding = nn.Linear(4, d_model)
        self.kinematic_embedding = nn.Linear(3, d_model)
        branch = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, branch_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                       activation='gelu', batch_first=True, norm_first=True), branch_layers,
            norm=nn.LayerNorm(d_model))
        self.space_encoder, self.kinematic_encoder = branch(), branch()
        self.group_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, group_heads, dim_feedforward=4 * width, dropout=dropout,
                                       activation='gelu', batch_first=True, norm_first=True), group_layers,
            norm=nn.LayerNorm(width))
        self.memory_queries = nn.Parameter(torch.empty(memory_slots, width))
        self.memory_attention = nn.MultiheadAttention(width, pool_heads, batch_first=True, dropout=0.0)
        self.context_attention = nn.MultiheadAttention(width, 1, batch_first=True, dropout=0.0)
        self.classifier = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.memory_queries, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, space: Tensor, kinematic: Tensor, times: Tensor, padding_mask: Tensor) -> Tensor:
        """编码一批变长片段，padding_mask=True 表示忽略。"""
        positional = self._time_encoding(times)
        space_hidden = self.space_encoder(self.space_embedding(space) + positional, src_key_padding_mask=padding_mask)
        kinematic_hidden = self.kinematic_encoder(self.kinematic_embedding(kinematic) + positional, src_key_padding_mask=padding_mask)
        weights = self._time_weights(times, padding_mask)
        return torch.cat((self._pool(space_hidden, weights), self._pool(kinematic_hidden, weights)), dim=-1)

    def memory(self, reference_q: Tensor) -> Tensor:
        """对一个非空正常参考集合生成 [m,2d] 动态或最终记忆。"""
        encoded = self.group_encoder(reference_q.unsqueeze(0))
        query = self.memory_queries.unsqueeze(0)
        memory, _ = self.memory_attention(query, encoded, encoded, need_weights=False)
        return memory.squeeze(0)

    def logits(self, target_q: Tensor, memory: Tensor) -> Tensor:
        """以目标为单查询读取记忆，输出每目标一个异常 logit。"""
        context, _ = self.context_attention(target_q.unsqueeze(1), memory.unsqueeze(0).expand(len(target_q), -1, -1),
                                            memory.unsqueeze(0).expand(len(target_q), -1, -1), need_weights=False)
        return self.classifier(torch.cat((target_q, context.squeeze(1)), dim=-1)).squeeze(-1)

    def _time_encoding(self, times: Tensor) -> Tensor:
        offsets = times - times[:, :1]
        frequencies = torch.exp(torch.arange(0, self.d_model, 2, device=times.device, dtype=times.dtype)
                                * (-math.log(10000.0) / self.d_model))
        result = torch.zeros((*times.shape, self.d_model), device=times.device, dtype=times.dtype)
        result[..., 0::2] = torch.sin(offsets.unsqueeze(-1) * frequencies)
        result[..., 1::2] = torch.cos(offsets.unsqueeze(-1) * frequencies)
        return result

    @staticmethod
    def _time_weights(times: Tensor, padding_mask: Tensor) -> Tensor:
        valid = ~padding_mask
        result = torch.zeros_like(times)
        for index in range(len(times)):
            length = int(valid[index].sum())
            if length == 1:
                result[index, 0] = 1.0
            elif length > 1:
                delta = times[index, 1:length] - times[index, :length - 1]
                result[index, 0] = delta[0] / 2.0
                result[index, length - 1] = delta[-1] / 2.0
                if length > 2:
                    result[index, 1:length - 1] = (delta[:-1] + delta[1:]) / 2.0
        return result

    @staticmethod
    def _pool(hidden: Tensor, weights: Tensor) -> Tensor:
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denominator
