import random  # 导入随机模块，用于设置随机种子
from tqdm import tqdm  # 导入进度条库，用于显示训练进度
from transformers import AutoTokenizer  # 导入Transformers库的自动分词器
import json  # 导入JSON处理库，用于读写配置文件
from datasets import load_dataset  # 导入Hugging Face数据集库
from tokenizers import (  # 导入tokenizers库的各种组件
    decoders,  # 用于将token ID转换回文本
    models,  # 提供各种分词器模型实现
    normalizers,  # 文本预处理规范化组件
    pre_tokenizers,  # 分词前的预处理组件
    processors,  # 处理分词结果的组件
    trainers,  # 提供训练分词器的工具
    Tokenizer,  # 主分词器类
)
import os  # 导入操作系统模块，用于文件路径操作

random.seed(42)  # 设置随机种子为42，确保结果可复现


def train_tokenizer():
    # 读取JSONL文件并提取文本数据的辅助函数
    def read_texts_from_jsonl(file_path, sample_rate=0.4):
        with open(file_path, 'r', encoding='utf-8') as f:  # 打开JSONL文件
            for line in f:  # 逐行读取
                data = json.loads(line)  # 解析JSON数据
                if random.random() <= sample_rate:
                    yield data['text']  # 仅返回文本字段
            f.close()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'dataset', 'pretrain_hq_new.jsonl')

    # 初始化分词器，使用BPE(字节对编码)算法
    tokenizer = Tokenizer(models.BPE())
    # 设置预分词器为字节级别，不添加前缀空格
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token，包括未知词、开始和结束标记
    special_tokens = ["<unk>", "<s>", "</s>"]

    # 配置BPE训练器
    trainer = trainers.BpeTrainer(
        vocab_size=12800,  # 设置词汇表大小为6400
        special_tokens=special_tokens,  # 添加特殊token
        show_progress=True,  # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 使用字节级别的初始字母表
        threads=4
    )

    # 读取文本数据，返回迭代器
    texts = read_texts_from_jsonl(data_path)

    # 从文本迭代器训练分词器
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器为字节级别，与预分词器对应
    tokenizer.decoder = decoders.ByteLevel()

    # 验证特殊token的索引是否正确分配
    assert tokenizer.token_to_id("<unk>") == 0  # 未知词应该是索引0
    assert tokenizer.token_to_id("<s>") == 1   # 开始标记应该是索引1
    assert tokenizer.token_to_id("</s>") == 2  # 结束标记应该是索引2

    # 创建保存分词器的目录
    tokenizer_dir = os.path.join(script_dir, "../model/minimind_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)  # 创建目录，如果已存在则不报错
    # 保存分词器到JSON文件
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    # 单独保存分词器模型
    tokenizer.model.save(tokenizer_dir)

    # 手动创建配置文件，详细设置分词器的各种参数
    config = {
        "add_bos_token": False,  # 不自动添加开始标记
        "add_eos_token": False,  # 不自动添加结束标记
        "add_prefix_space": False,  # 不添加前缀空格
        "added_tokens_decoder": {  # 特殊token的详细配置
            "0": {  # 索引0的token配置
                "content": "<unk>",  # token内容
                "lstrip": False,  # 不去除左侧空白
                "normalized": False,  # 不标准化
                "rstrip": False,  # 不去除右侧空白
                "single_word": False,  # 不作为单独的词
                "special": True  # 标记为特殊token
            },
            "1": {  # 索引1的token配置
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {  # 索引2的token配置
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],  # 额外的特殊token列表（为空）
        "bos_token": "<s>",  # 开始标记
        "clean_up_tokenization_spaces": False,  # 不清理标记化空格
        "eos_token": "</s>",  # 结束标记
        "legacy": True,  # 使用兼容模式
        "model_max_length": 32768,  # 模型最大长度，适用于长文本
        "pad_token": "<unk>",  # 填充token使用未知词
        "sp_model_kwargs": {},  # SentencePiece模型参数（为空）
        "spaces_between_special_tokens": False,  # 特殊token之间不添加空格
        "tokenizer_class": "PreTrainedTokenizerFast",  # 使用快速分词器类
        "unk_token": "<unk>",  # 未知词token
        # 聊天模板，定义如何格式化对话数据
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是 MiniMind，是一个有用的人工智能助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 将配置保存为JSON文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)  # 保存为美化格式，支持中文

    print("Tokenizer training completed and saved.")  # 打印完成信息


def eval_tokenizer():
    from transformers import AutoTokenizer  # 导入自动分词器

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_dir = os.path.join(script_dir, "../model/minimind_tokenizer")
    # 加载训练好的分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 测试聊天模板应用
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # 应用聊天模板但不分词
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)  # 打印格式化后的对话文本

    # 获取分词器词汇表大小
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    # 对格式化文本进行分词
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))  # 打印分词后的token数量

    # 解码测试，验证分词器的编码和解码是否一致
    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)  # 保留特殊token进行解码
    print('decoder和原始文本是否一致：', response == new_prompt)  # 打印对比结果


def main():
    train_tokenizer()  # 训练分词器
    eval_tokenizer()  # 评估分词器


if __name__ == '__main__':
    main()  # 程序入口