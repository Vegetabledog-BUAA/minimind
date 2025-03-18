# 导入必要的Python库
import os                      # 操作系统接口，用于文件和路径操作
import platform                # 获取平台信息
import argparse                # 命令行参数解析
import time                    # 计时功能
import math                    # 数学函数
import warnings                # 警告控制
import pandas as pd            # 数据处理库
import torch                   # PyTorch深度学习框架
import torch.distributed as dist  # PyTorch分布式训练
from torch import optim, nn    # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载工具
from contextlib import nullcontext  # 上下文管理

from transformers import AutoTokenizer  # Hugging Face的分词器

# 导入自定义模型和配置
from model.model import MiniMindLM  # 主模型
from model.LMConfig import LMConfig  # 模型配置
from model.dataset import PretrainDataset  # 预训练数据集

# 忽略警告信息
warnings.filterwarnings('ignore')


def Logger(content):
    """
    日志打印函数，在分布式训练中仅在主进程(rank 0)打印信息
    
    参数:
        content: 需要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算当前学习率 - 实现了余弦退火策略与最小学习率保证
    
    参数:
        current_step: 当前步数
        total_steps: 总步数
        lr: 基础学习率
    返回:
        调整后的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    训练一个完整的epoch
    
    参数:
        epoch: 当前的epoch编号
        wandb: Weights & Biases对象，用于记录训练指标
    """
    # 定义交叉熵损失函数，reduction='none'以便后续根据loss_mask筛选
    # reduction 参数决定了损失函数如何处理批次中的多个损失值。在 nn.CrossEntropyLoss 中，它有三种可能的值：
    # 'mean' (默认值)：返回所有个体损失的平均值
    # 'sum'：返回所有个体损失的总和
    # 'none'：不进行任何降维操作，保留每个样本的单独损失值
    # 代码中使用 reduction='none' 的原因
    # 在这段代码中使用 reduction='none' 是为了：

    # 先计算每个位置(token)的损失，保留原始形状
    # 然后通过 loss_mask 来筛选出需要计算损失的位置
    # 最后手动计算这些有效位置的平均损失
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    start_time = time.time()  # 记录开始时间
    
    # 遍历数据加载器
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移到指定设备
        X = X.to(args.device)  # 输入张量
        Y = Y.to(args.device)  # 目标张量
        loss_mask = loss_mask.to(args.device)  # 损失掩码张量

        # 更新学习率 - 使用自定义的学习率调度函数
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate) # 当前历元*每个历元的迭代次数+当前迭代次数
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度训练上下文
        # 创建混合精度训练环境，允许模型使用较低精度(如float16或bfloat16)进行计算
        # 降低显存使用，提高训练速度，同时保持数值稳定性    
        with ctx:
            # 前向传播
            res = model(X)
            # 计算损失 - 先计算每个token的损失
            # res.logits形状为[batch_size, seq_len, vocab_size]
            # .view(-1, res.logits.size(-1))将其展平为[batch_size×seq_len, vocab_size]
            # Y.view(-1)将目标也展平为一维向量
            # 计算每个位置的交叉熵损失后，恢复为原始形状[batch_size, seq_len]
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # 通过掩码筛选有效损失并计算平均值
            # 使用loss_mask选择需要计算损失的有效位置(如忽略填充标记)
            # 将损失与掩码相乘，使无效位置损失为零
            # 对有效位置的损失求和后除以有效位置数量，得到平均损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 添加辅助损失(如果模型有额外的正则化损失)
            loss += res.aux_loss
            # 梯度累积：将损失除以累积步数
            # 将损失除以累积步数，为梯度累积做准备
            # 允许模拟更大批次训练，解决显存限制问题
            loss = loss / args.accumulation_steps

        # 反向传播(使用梯度缩放器以防止混合精度训练中的梯度消失)
        # scaler.scale放大损失值，防止混合精度训练中的梯度消失
        # backward()计算所有模型参数的梯度
        scaler.scale(loss).backward()

        # 梯度累积：每accumulation_steps步更新一次参数
        # 当累积足够步数时执行参数更新
        # unscale_将梯度缩放回原始大小
        # clip_grad_norm_裁剪梯度，防止梯度爆炸
        # scaler.step(optimizer)使用优化器更新模型参数
        # scaler.update()调整未来迭代的缩放因子
        if (step + 1) % args.accumulation_steps == 0:
            # 将梯度缩放回正常范围
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 使用缩放器更新参数
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度
            # 清除所有参数的梯度，为下一轮累积做准备
            # set_to_none=True直接将梯度设为None而非零，提高内存效率
            optimizer.zero_grad(set_to_none=True)

        # 定期打印日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录指标到Weights & Biases(如果启用)
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # 切换到评估模式
            # 确定模型文件名(根据是否使用MoE结构)
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            # 提取模型状态字典(考虑DDP封装情况)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存模型
            torch.save(state_dict, ckp)
            model.train()  # 切换回训练模式


def init_model(lm_config):
    """
    初始化模型和分词器
    
    参数:
        lm_config: 语言模型配置对象
    返回:
        model: 初始化的模型
        tokenizer: 初始化的分词器
    """
    # 从预定义位置加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 初始化模型并移至指定设备
    model = MiniMindLM(lm_config).to(args.device)
    # 打印模型参数量
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练环境
    """
    if not ddp: return  # 如果不是分布式训练则直接返回
    global ddp_local_rank, DEVICE  # 声明全局变量

    # 初始化进程组
    dist.init_process_group(backend="nccl")  # NCCL是NVIDIA GPU的推荐后端
    # 获取分布式训练环境变量
    ddp_rank = int(os.environ["RANK"])  # 全局进程排名
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 本地进程排名
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    # 设置当前进程使用的GPU
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py  # 这是启动分布式训练的命令示例
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")  # 输出目录
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=32)  # 批次大小
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 训练精度类型
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用W&B记录实验
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")  # W&B项目名
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载工作线程数
    parser.add_argument("--ddp", action="store_true")  # 是否启用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=8)  # 梯度累积步数，用于增大"等效批次大小"
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热迭代次数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志记录间隔
    parser.add_argument("--save_interval", type=int, default=100)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地进程排名，分布式训练使用
    parser.add_argument('--dim', default=512, type=int)  # 模型隐藏层维度
    parser.add_argument('--n_layers', default=12, type=int)  # 模型层数
    parser.add_argument('--max_seq_len', default=1024, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE(混合专家模型)结构
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")  # 预训练数据路径
    args = parser.parse_args()

    # 创建模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 计算每次迭代处理的token数量
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 设置随机种子以保证可重复性
    torch.manual_seed(1337)
    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置W&B运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 创建混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检测是否在分布式环境中运行
    ddp = int(os.environ.get("RANK", -1)) != -1  # 检查是否设置了RANK环境变量
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 默认值

    # 如果是分布式训练，初始化分布式环境
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化W&B (仅在主进程中)
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)
    
    # 创建数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    
    # 创建数据采样器和加载器
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 在分布式训练中使用分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 将数据固定在内存中，加速GPU传输
        drop_last=False,  # 不丢弃最后一个不完整批次
        shuffle=False,    # 不随机打乱，如使用分布式采样器会自动打乱
        num_workers=args.num_workers,  # 数据加载工作线程数
        sampler=train_sampler  # 使用分布式采样器(如果适用)
    )

    # 创建混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 配置分布式训练模型
    if ddp:
        # 忽略位置编码矩阵在分布式同步中的参数，因为它们是预计算的常量
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 包装模型为DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)