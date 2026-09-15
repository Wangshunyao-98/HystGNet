from __future__ import annotations

import math

import torch
from torch import nn


class ResidualGRUEncoder(nn.Module):

    def __init__(self, input_size: int = 1, hidden_size: int = 128) -> None:
        super().__init__()
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)
        return out1 + out2


class StaticConditionBroadcasting(nn.Module):

    def __init__(self, param_size: int = 12, hidden_size: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(param_size, hidden_size)

    def forward(self, params: torch.Tensor, seq_len: int) -> torch.Tensor:
        param_features = self.fc(params)
        return param_features.unsqueeze(1).expand(-1, seq_len, -1)


class TemporalSelfAttention(nn.Module):

    def __init__(self, hidden_size: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = math.sqrt(hidden_size)

    def forward(self, seq_features: torch.Tensor) -> torch.Tensor:
        q = self.query(seq_features)
        k = self.key(seq_features)
        v = self.value(seq_features)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.matmul(self.dropout(attention_weights), v)
        return self.norm(seq_features + attended)


class AdaptiveGatedFusion(nn.Module):

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        seq_features: torch.Tensor,
        param_features: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat((seq_features, param_features), dim=-1)
        alpha = self.gate(combined)
        fused = seq_features * alpha + param_features * (1.0 - alpha)
        return self.norm(fused)


class FusionModule(nn.Module):

    def __init__(self, hidden_size: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal_attention = TemporalSelfAttention(hidden_size, dropout)
        self.gated_fusion = AdaptiveGatedFusion(hidden_size)

    def forward(
        self,
        seq_features: torch.Tensor,
        param_features: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.temporal_attention(seq_features)
        return self.gated_fusion(attended, param_features)


class ResponseDecoder(nn.Module):

    def __init__(
        self,
        hidden_size: int = 128,
        output_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru1 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        out1, _ = self.gru1(fused_features)
        out2, _ = self.gru2(out1)
        decoded = self.dropout(out1 + out2)
        return self.fc(decoded)


class HystGNet(nn.Module):

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        param_size: int = 12,
        output_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ResidualGRUEncoder(input_size, hidden_size)
        self.static_broadcast = StaticConditionBroadcasting(param_size, hidden_size)
        self.fusion = FusionModule(hidden_size, dropout)
        self.decoder = ResponseDecoder(hidden_size, output_size, dropout)

    def forward(self, x_seq: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        seq_features = self.encoder(x_seq)
        param_features = self.static_broadcast(params, x_seq.size(1))
        fused_features = self.fusion(seq_features, param_features)
        return self.decoder(fused_features)
