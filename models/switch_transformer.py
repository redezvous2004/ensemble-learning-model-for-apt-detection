import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
  def __init__(self, d_model, hidden_dim):
    super(Expert, self).__init__()
    self.input_projection = nn.Linear(d_model, hidden_dim)
    self.output_projection = nn.Linear(hidden_dim, d_model)
  def forward(self, x):
    x = F.relu(self.input_projection(x))
    return self.output_projection(x)

class GatingNetwork(nn.Module):
  def __init__(self, d_model, num_experts):
    super(GatingNetwork, self).__init__()
    self.gate = nn.Linear(d_model, num_experts)
  def forward(self, x):
    return F.softmax(self.gate(x), dim=-1)

class MoELayer(nn.Module):
  def __init__(self, d_model, hidden_dim, num_experts, top_k=2):
    super(MoELayer, self).__init__()
    self.experts = nn.ModuleList([Expert(d_model, hidden_dim) for _ in range(num_experts)])
    self.gate = GatingNetwork(d_model, num_experts)
    self.num_experts = num_experts
    self.top_k = top_k
  def forward(self, x):
    batch_size, seq_len, d_model = x.shape
    gating_scores = self.gate(x)
    top_k_scores, top_k_indices = torch.topk(gating_scores, self.top_k, dim=-1)
    top_k_scores_normalized = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

    output = torch.zeros_like(x)
    for i in range(self.num_experts):
      mask = (top_k_indices == i).float()
      if mask.any():
        position_using_expert = mask.any(dim=-1)
        if position_using_expert.any():
          expert_input = x[position_using_expert]
          expert_output = self.experts[i](expert_input)
          expert_weights = torch.zeros(batch_size, seq_len, device=x.device)
          for k in range(self.top_k):
            mask_k = (top_k_indices[..., k] == i)
            expert_weights[mask_k] = top_k_scores[..., k][mask_k]
          weighted_output = torch.zeros_like(x, dtype=expert_output.dtype)
          weighted_output[position_using_expert] = expert_output
          output += weighted_output * expert_weights.unsqueeze(-1)
    return output

class SwitchTransformer(nn.Module):
  def __init__(self, d_model, nhead, num_experts, hidden_dim, dropout=0.1, top_k=2, batch_first=False):
    super(SwitchTransformer, self).__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
    self.moe = MoELayer(d_model, hidden_dim, num_experts, top_k=top_k)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
  def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout(src2)
    src = self.norm1(src)
    src2 = self.moe(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)
    return src