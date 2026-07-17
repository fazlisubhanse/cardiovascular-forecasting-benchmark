"""Small recurrent, causal-convolutional, hybrid, and capacity-audit networks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentRegressor(nn.Module):
    def __init__(self, kind: str, hidden: int, bidirectional: bool, dropout: float):
        super().__init__(); cls = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = cls(1, hidden, batch_first=True, bidirectional=bidirectional)
        width = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout); self.output = nn.Linear(width, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.rnn(x); return self.output(self.dropout(sequence[:, -1, :]))


class CausalConv(nn.Module):
    def __init__(self, input_channels: int, filters: int, kernel_size: int):
        super().__init__(); self.kernel_size=kernel_size; self.conv=nn.Conv1d(input_channels,filters,kernel_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(F.pad(x,(self.kernel_size-1,0))))


class CausalCNNRegressor(nn.Module):
    def __init__(self, filters: int, kernel_size: int, dropout: float):
        super().__init__(); self.cnn=CausalConv(1,filters,kernel_size); self.dropout=nn.Dropout(dropout); self.output=nn.Linear(filters,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        features=self.cnn(x.transpose(1,2)).mean(dim=2); return self.output(self.dropout(features))


class CNNRecurrentRegressor(nn.Module):
    def __init__(self, kind:str, filters:int, hidden:int, dropout:float):
        super().__init__(); self.cnn=CausalConv(1,filters,2); cls=nn.LSTM if kind=="LSTM" else nn.GRU
        self.rnn=cls(filters,hidden,batch_first=True); self.dropout=nn.Dropout(dropout); self.output=nn.Linear(hidden,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        sequence=self.cnn(x.transpose(1,2)).transpose(1,2); encoded,_=self.rnn(sequence)
        return self.output(self.dropout(encoded[:,-1,:]))


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, filters:int, hidden:int, heads:int, dense:int, dropout:float, attention:bool=True, kernel:int=2):
        super().__init__(); self.cnn=CausalConv(1,filters,kernel); self.bilstm=nn.LSTM(filters,hidden,batch_first=True,bidirectional=True)
        width=hidden*2; self.attention_enabled=attention
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True) if attention else None
        self.norm=nn.LayerNorm(width); self.dense=nn.Linear(width,dense); self.dropout=nn.Dropout(dropout); self.output=nn.Linear(dense,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        features=self.cnn(x.transpose(1,2)).transpose(1,2); sequence,_=self.bilstm(features)
        if self.attention is not None:
            attended,_=self.attention(sequence,sequence,sequence,need_weights=False); sequence=self.norm(sequence+attended)
        else:
            sequence=self.norm(sequence)
        pooled=sequence.mean(dim=1); dense=F.relu(self.dense(pooled)); return self.output(self.dropout(dense))


def architecture_size_candidates(name: str) -> list[dict[str, Any]]:
    """Return the prespecified capacity candidates for one architecture."""
    if name in {"LSTM","GRU"}: return [{"hidden":v} for v in (8,16)]
    if name=="BiLSTM": return [{"hidden":v} for v in (4,8)]
    if name=="CausalCNN": return [{"filters":f,"kernel_size":k} for f in (8,16) for k in (2,3)]
    if name in {"CNN_LSTM","CNN_GRU"}: return [{"filters":f,"hidden":8} for f in (8,16)]
    if name=="CNN_BiLSTM_Attention_Compact": return [{"filters":8,"hidden":4,"heads":1,"dense":8,"kernel":2}]
    if name=="CNN_BiLSTM_Compact_NoAttention": return [{"filters":8,"hidden":4,"heads":1,"dense":8,"kernel":2}]
    if name=="CNN_BiLSTM_Attention_Original23K": return [{"filters":32,"hidden":24,"heads":4,"dense":32,"kernel":3}]
    raise ValueError(f"Unknown architecture: {name}")


def build_architecture(name: str, size: dict[str, Any], dropout: float) -> nn.Module:
    """Build a scalar-output network accepting [batch, lookback, 1]."""
    if name in {"LSTM","GRU"}: return RecurrentRegressor(name,int(size["hidden"]),False,dropout)
    if name=="BiLSTM": return RecurrentRegressor("LSTM",int(size["hidden"]),True,dropout)
    if name=="CausalCNN": return CausalCNNRegressor(int(size["filters"]),int(size["kernel_size"]),dropout)
    if name in {"CNN_LSTM","CNN_GRU"}: return CNNRecurrentRegressor(name.split("_")[1],int(size["filters"]),int(size["hidden"]),dropout)
    if name.startswith("CNN_BiLSTM"):
        return CNNBiLSTMAttention(int(size["filters"]),int(size["hidden"]),int(size["heads"]),int(size["dense"]),dropout,
                                 attention="NoAttention" not in name,kernel=int(size["kernel"]))
    raise ValueError(name)
