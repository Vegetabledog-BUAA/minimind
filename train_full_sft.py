import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

# 忽略所有警告信息
warnings.filterwarnings('ignore')


def Logger(content):
    """
    日志打印函数，在非分布式训练或者在主进程(rank=0)中打印内容
    
    参数:
        content: 要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算余弦退火学习率
    
    参数:
        current_step: 当前步数
        total_steps: 总步数
        lr: 基础学习率
    
    返回:
        调整后的学习率，包含预热(lr/10)和余弦退火部分
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    训练一个轮次
    
    参数:
        epoch: 当前轮次索引
        wandb: wandb实例，用于记录训练指标
    """
    # 定义交叉熵损失函数，不进行reduction以便后续使用loss_mask
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # 记录本轮次开始时间
    start_time = time.time()
    
    # 遍历数据加载器
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移至指定设备
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，用于忽略填充部分
        
        # 计算当前步的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        # 为优化器更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用上下文管理器，在CUDA设备上进行自动混合精度训练
        with ctx:
            # 前向传播
            res = model(X)
            # 计算交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 调整logits形状为[batch*seq_len, vocab_size]
                Y.view(-1)  # 调整目标形状为[batch*seq_len]
            ).view(Y.size())  # 将结果调整回原始形状

            # 应用损失掩码并求平均
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 添加辅助损失(如MoE负载均衡损失)
            loss += res.aux_loss
            # 梯度累积，除以累积步数
            loss = loss / args.accumulation_steps

        # 缩放损失并反向传播
        scaler.scale(loss).backward()

        # 梯度累积完成后更新模型
        if (step + 1) % args.accumulation_steps == 0:
            # 取消损失缩放，以便进行梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数
            scaler.step(optimizer)
            # 更新缩放因子
            scaler.update()

            # 清除梯度
            optimizer.zero_grad(set_to_none=True)

        # 定期记录训练状态
        if step % args.log_interval == 0:
            # 计算已花费的时间
            spend_time = time.time() - start_time
            # 打印训练信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录到wandb（如果启用）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 切换到评估模式
            model.eval()
            # 确定模型名称（添加MoE标记如果使用MoE）
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            # 获取模型状态字典，处理DDP模式
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存模型
            torch.save(state_dict, ckp)
            # 恢复训练模式
            model.train()


def init_model(lm_config):
    """
    初始化模型和分词器
    
    参数:
        lm_config: 语言模型配置
    
    返回:
        model: 初始化并加载预训练权重的模型
        tokenizer: 分词器
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 初始化模型
    model = MiniMindLM(lm_config)
    
    # 确定预训练检查点路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
    
    # 加载预训练权重
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    
    # 打印模型参数量
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    # 将模型移至指定设备
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练环境
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 初始化进程组，使用nccl后端（适用于GPU）
    dist.init_process_group(backend="nccl")
    # 获取全局进程排名
    ddp_rank = int(os.environ["RANK"])
    # 获取本地进程排名（单节点多GPU）
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # 获取总进程数
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # 设置当前进程使用的设备
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out")  # 输出目录
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=32)  # 批次大小
    parser.add_argument("--learning_rate", type=float, default=5e-5)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型，用于混合精度训练
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用wandb记录
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")  # wandb项目名称
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载器工作线程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=1)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热迭代次数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志记录间隔
    parser.add_argument("--save_interval", type=int, default=100)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地进程排名，DDP使用
    parser.add_argument('--dim', default=512, type=int)  # 模型维度
    parser.add_argument('--n_layers', default=8, type=int)  # 模型层数
    parser.add_argument('--max_seq_len', default=512, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE（混合专家模型）
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")  # 数据集路径

    # 解析命令行参数
    args = parser.parse_args()

    # 创建模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    # 创建必要的目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 计算每批次处理的token数
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 设置随机种子以确保可重复性
    torch.manual_seed(1337)
    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称
    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 根据设备类型设置自动混合精度上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    # 检测是否处于分布式环境
    ddp = int(os.environ.get("RANK", -1)) != -1
    # 初始化分布式相关变量
    ddp_local_rank, DEVICE = 0, "cuda:0"
    # 如果使用DDP，初始化分布式环境
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（如果启用）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)

    # 创建训练数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 创建分布式采样器（如果使用DDP）
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 将数据固定在内存中，加速GPU拷贝
        drop_last=False,  # 不丢弃最后不完整的批次
        shuffle=False,    # 使用采样器时不需要shuffle
        num_workers=args.num_workers,  # 数据加载工作进程数
        sampler=train_sampler  # 采样器
    )

    # 创建梯度缩放器，用于自动混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 如果使用分布式训练，配置DDP
    if ddp:
        # 忽略位置编码参数同步
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 包装模型为DDP模型
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每轮迭代次数
    iter_per_epoch = len(train_loader)
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)