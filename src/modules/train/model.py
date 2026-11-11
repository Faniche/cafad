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
        space_input = self.space_embedding(space)
        kinematic_input = self.kinematic_embedding(kinematic)
        # 时间运算保持FP32，位置编码相加时才转换，避免把AMP嵌入重新提升为FP32。
        space_hidden = self.space_encoder(space_input + positional.to(space_input.dtype), src_key_padding_mask=padding_mask)
        kinematic_hidden = self.kinematic_encoder(kinematic_input + positional.to(kinematic_input.dtype), src_key_padding_mask=padding_mask)
        weights = self._time_weights(times, padding_mask)
        return torch.cat((self._pool(space_hidden, weights), self._pool(kinematic_hidden, weights)), dim=-1)

    def memory(self, reference_q: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """[R,2d] 返回单记忆；[B,R,2d] 返回 B 份独立记忆，参考填充不参与注意力。"""
        single = reference_q.ndim == 2
        references = reference_q.unsqueeze(0) if single else reference_q
        encoded = self.group_encoder(references, src_key_padding_mask=padding_mask)
        query = self.memory_queries.unsqueeze(0).expand(references.shape[0], -1, -1)
        memory, _ = self.memory_attention(query, encoded, encoded, key_padding_mask=padding_mask, need_weights=False)
        return memory.squeeze(0) if single else memory

    def logits(self, target_q: Tensor, memory: Tensor) -> Tensor:
        """以目标为单查询读取记忆，输出每目标一个异常 logit。"""
        memories = memory.unsqueeze(0).expand(len(target_q), -1, -1) if memory.ndim == 2 else memory
        context, _ = self.context_attention(target_q.unsqueeze(1), memories, memories, need_weights=False)
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
        intervals = torch.where(valid[:, 1:] & valid[:, :-1], times[:, 1:] - times[:, :-1], 0.0)
        zero = torch.zeros_like(times[:, :1])
        result = (torch.cat((zero, intervals), dim=1) + torch.cat((intervals, zero), dim=1)) / 2.0
        singleton = (valid.sum(dim=1, keepdim=True) == 1) & valid
        return torch.where(singleton, torch.ones_like(result), result)

    @staticmethod
    def _pool(hidden: Tensor, weights: Tensor) -> Tensor:
        # 长片段加权求和用FP32，避免FP16累加溢出。
        hidden, weights = hidden.float(), weights.float()
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        hidden = hidden.masked_fill(weights.unsqueeze(-1) == 0, 0.0)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denominator
