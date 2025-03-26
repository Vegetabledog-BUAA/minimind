import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A，将输入从in_features维映射到rank维
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B，将数据从rank维映射回out_features维
        # 矩阵A高斯初始化，标准做法，确保训练开始时有良好的梯度
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化，确保训练开始时LoRA不干扰原始模型的输出
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))  # 实现A·B矩阵乘法的低秩分解，B(A(x))相当于(B·A)x


def apply_lora(model, rank=16):
    for name, module in model.named_modules():  # 遍历模型中的所有模块
        # 只为方阵（输入输出维度相同）的线性层添加LoRA
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA模块，rank参数决定了低秩分解的维度
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)  # 将LoRA模块作为原始线性层的属性
            original_forward = module.forward  # 保存原始前向传播函数

            # 定义新的前向传播函数，结合原始输出和LoRA输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # 原始输出 + LoRA调整量

            module.forward = forward_with_lora  # 替换前向传播函数


def load_lora(model, path):
    # 加载保存的LoRA权重
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):  # 检查模块是否有LoRA属性
            # 筛选出属于当前模块的LoRA参数，并去除路径前缀
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)  # 加载LoRA权重到对应模块


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):  # 检查模块是否有LoRA属性
            # 保存LoRA参数，并添加模块名称前缀
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)  # 添加到总状态字典
    torch.save(state_dict, path)  # 保存所有LoRA参数到文件
