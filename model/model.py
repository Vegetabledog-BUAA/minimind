import math
import struct
import inspect
import time

from LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
#   在 precompute_pos_cis 函数中，end: int = int(32 * 1024) 的意思是将 end 参数的默认值设置为 32 * 1024，即 32768。
#   这个参数表示生成频率矩阵时的时间步长的最大值。 具体来说，end 参数用于定义时间步长 t 的范围：
#   t实际上对应的是输入序列的位置，而 end 则对应了输入序列的最大长度。
#   在这里，end 的默认值为 32768，这意味着生成的频率矩阵将包含 32768 个token的位置信息。
#   这个默认值是为了确保生成的频率矩阵足够长，以便在处理长序列时能够提供足够的信息。
    
#   在 precompute_pos_cis 函数中，torch.outer 和 torch.polar 的作用如下：
#   torch.outer:
#   torch.outer 用于计算两个一维张量的外积。具体来说，如果 t 是一个长度为 m 的张量，
#   freqs 是一个长度为 n 的张量，那么 torch.outer(t, freqs) 将生成一个形状为 (m, n) 的二维张量，
#   其中每个元素是 t 中的一个元素与 freqs 中的一个元素的乘积。
#   这个操作在这里用于生成频率矩阵 freqs，其中每一行对应一个时间步长 t，每一列对应一个频率。
#   torch.polar:
#   torch.polar 用于创建一个复数张量，其中实部和虚部分别由给定的幅度和相位角度组成。
#   在这里，torch.ones_like(freqs) 生成一个与 freqs 形状相同的全为 1 的张量，表示幅度为 1，而 freqs 作为相位角度。
#   torch.polar 将这些幅度和相位角度组合成一个复数张量 pos_cis，其中每个元素的幅度为 1，相位角度由 freqs 确定。
#   总结来说，torch.outer 用于生成频率矩阵，而 torch.polar 用于将这些频率转换为复数形式的旋转矩阵。
#   这个旋转矩阵在后续的旋转嵌入操作中会被使用。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # 生成频率序列
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore 计算张量外积，即形成t*freqs的矩阵
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    # 这里的旋转编码操作与论文中保持一致，采用的复数形式的旋转矩阵。
    def unite_shape(pos_cis, x):
        # 将 pos_cis 的形状转换为与 x 相同
        ndim = x.ndim
        # assert 是 Python 中用于调试和验证程序中条件的一个重要语句。
        # 它可以在程序中某些关键点检查条件是否为真，如果条件为假，则抛出 AssertionError 异常并终止程序执行。
        assert 0 <= 1 < ndim # 确保 x 至少有两个维度。这是为了保证后续代码中的索引操作是有效的
        # 确保 pos_cis 的形状与 x 的形状相匹配 这一步实际上是在MiniMindLM中保证的，此处只是再次确认
        # 实际上，x的第二个维度代表了序列长度，最后一个维度代表了一个头的词向量的维度
        assert pos_cis.shape == (x.shape[1], x.shape[-1]) 
        # 创建一个新的形状列表 shape，其中只有 x 的第二维和最后一维保持不变，其他维度都设置为 1。这是为了使 pos_cis 可以在这些维度上进行广播
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] 
        return pos_cis.view(*shape)

    # 将 xq 张量的最后一个维度重塑为大小为 2 的子维度，并将其转换为复数形式。具体步骤如下：
    # xq.float()：将 xq 转换为浮点数类型。
    # xq.float().reshape(*xq.shape[:-1], -1, 2)：将 xq 的最后一个维度重塑为大小为 2 的子维度。
    # torch.view_as_complex(...)：将重塑后的张量视为复数张量，其中每两个连续的元素表示一个复数的实部和虚部。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    # 将 xq_ 与 pos_cis 相乘，得到一个复数张量，然后将其转换回实数形式，并将最后两个维度展平为一个维度。具体步骤如下：
    # xq_ * pos_cis：将 xq_ 与 pos_cis 相乘，得到一个复数张量。
    # torch.view_as_real(...)：将复数张量转换为实数张量，其中每个复数的实部和虚部分别存储在相邻的两个位置。
    # .flatten(3)：将最后两个维度展平为一个维度。
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0 # 确保 n_heads 可以被 n_kv_heads 整除
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf")) # 生成一个全为负无穷的 mask 张量
        mask = torch.triu(mask, diagonal=1) # 这行代码将 mask 张量的上三角部分（不包括对角线）保留为负无穷大 (-inf)，其余部分保持不变。torch.triu 函数用于返回一个上三角矩阵，其中 diagonal=1 表示从主对角线的上方一行开始。
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim) # 将查询向量分配到多个头中，此时 xq 的形状为 (bsz, seq_len, n_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis) # 每个头的查询向量和键向量都应用旋转编码
        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        ) # 此处的转置操作是为了将查询向量、键向量和值向量的维度顺序调整为 (bsz, n_heads, seq_len, head_dim)
        if self.flash and seq_len != 1: # 如果启用了 Flash Attention 且序列长度不为 1
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1) # 此处重新调整输出的形状，使其与输入张量 x 的形状相同，维度顺序为 (bsz, seq_len, dim)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    config_class = LMConfig  # 指定配置类为 LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()  # 如果没有传入参数，则使用默认的 LMConfig
        super().__init__(self.params)  # 调用父类的构造函数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers  # 获取词汇表大小和层数
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  # 定义词嵌入层
        self.dropout = nn.Dropout(params.dropout)  # 定义 dropout 层
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])  # 定义多个 MiniMindBlock 层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 定义 RMSNorm 层
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 定义输出层
        self.tok_embeddings.weight = self.output.weight  # 共享词嵌入层和输出层的权重
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)  # 注册位置编码缓存
        self.OUT = CausalLMOutputWithPast()  # 定义输出格式

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)  # 初始化 past_key_values
        start_pos = args.get('start_pos', 0)  # 获取起始位置 含义是从 args 字典中获取键 'start_pos' 对应的值。如果该键不存在，则返回默认值 0。
        h = self.dropout(self.tok_embeddings(input_ids))  # 获取词嵌入并应用 dropout
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]  # 获取位置编码
        past_kvs = []  # 初始化 past_kvs 列表
        for l, layer in enumerate(self.layers):  # 遍历每一层
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )  # 计算每一层的输出
            past_kvs.append(past_kv)  # 保存 past_kv
        logits = self.output(self.norm(h))  # 计算输出 logits，应用 RMSNorm，其维度为 (bsz, seq_len, vocab_size)
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))  # 计算辅助损失
        self.OUT.__setitem__('logits', logits)  # 设置输出 logits
        self.OUT.__setitem__('aux_loss', aux_loss)  # 设置辅助损失
        self.OUT.__setitem__('past_key_values', past_kvs)  # 将 past_kvs 设置 为 past_key_values
        return self.OUT  # 返回输出

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args): # 输入id，结束标记id，最大新token数，温度，top-p，流式生成，rp（重复惩罚），使用缓存，pad标记id
        # 参数代表"重复惩罚"(repetition penalty)，是一种用于控制文本生成中重复内容的技术。
        # 具体功能：
        # 识别已经生成过的token（通过set(input_ids.tolist()[0])获取唯一tokens）
        # 调整这些token在下一次生成中的概率
        # 参数值的影响：
        # 当 rp > 1：降低已生成tokens的概率，减少重复
        # 当 rp = 1：不进行调整（默认值）
        # 当 rp < 1：提高已生成tokens的概率，鼓励重复
        # 在语言模型生成中，适当设置 rp（通常为1.1~1.3）可以有效减少文本生成中常见的重复问题，提高生成内容的多样性和连贯性。
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        # 代码使用for循环是为了分别处理批次中的每个输入序列。这样做的原因是：
        # 每个序列可能有不同的填充情况和实际长度
        # 每个序列需要独立进行生成，有各自的上下文和状态
        # 每个序列的生成结束时间可能不同

        # 生成终止条件
        # 生成终止的判断发生在_stream方法中（代码底部可见）：
        # 达到最大长度：while input_ids.shape[1] < max_new_tokens - 1 当序列长度达到指定的最大值时停止生成
        # 生成了结束标记：if input_ids_next.item() == eos_token_id: break 当生成了结束标记（EOS token）时中断循环
        # 这种设计允许模型灵活处理不同长度的输入序列，并为每个序列单独控制生成过程。这样可以提高生成的效率和质量。
        generated = []
        for i in range(input_ids.size(0)):
            # 去除填充标记：提取每个序列中的非填充部分作为实际输入
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            # 调用_stream对每个序列进行逐token生成
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # 收集生成结果：
            # 从生成器中获取每一步的token并拼接
            # 组装完整序列：将原始输入与新生成内容连接，并存入结果列表
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        # 统一长度：将所有生成序列填充到相同长度以便批量返回
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        # 初始化：记录原始序列长度，标记首轮推理，初始化KV缓存为空
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        
        # 主生成循环：持续到达到最大长度限制
        while input_ids.shape[1] < max_new_tokens - 1:
            # 首轮推理或不使用缓存时，处理整个序列
            if first_seq or not use_cache:
                # 进行完整序列前向计算，并标记首轮推理已完成
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False 
            else:
                # 非首轮且使用缓存：只需处理最新生成的token
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                        start_pos=input_ids.shape[1] - 1, **args)
            
            # 提取最后位置的logits和更新的KV缓存
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values # logits是模型输出的概率分布，其维度为三维，每一维分别表示batch_size、序列长度和词表大小
            
            # 重复惩罚：降低已出现token的概率
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            
            # 温度调节：控制采样随机性（低温更确定，高温更随机） 温度越低，概率分布越尖锐，生成结果越确定，相反，温度越高，概率分布越接近，生成结果越随机
            logits /= (temperature + 1e-9)  # 1e-9防止除零
            
            # Top-p（核采样）实现：只从累积概率不超过p的最高概率tokens中采样
            if top_p is not None and top_p < 1.0:
                # 对logits降序排序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                # 计算概率分布
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 找出累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 右移一位确保至少保留一个token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() # 也就是说最后一个false向右移了一位
                sorted_indices_to_remove[:, 0] = False  # 保留最高概率token
                # 将排序后的掩码恢复到原始顺序
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # 将被过滤tokens的概率设为0
                logits[indices_to_remove] = -float('Inf')
                
            # 基于处理后的概率分布采样下一个token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1) # 从多项式分布中采样，返回的是采样结果的索引
            
            # 将新token添加到现有序列
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            
            # 流式返回当前生成结果（只包含新生成部分）
            yield input_ids[:, start:]
            
            # 检查是否生成了结束标记，是则提前终止
            if input_ids_next.item() == eos_token_id:
                break
if __name__ == "__main__":
    FeedForward(LMConfig())