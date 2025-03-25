import json
import re
file_path = 'dataset\pretrain_hq_new.jsonl'  # JSONL文件路径
with open(file_path, 'r', encoding='utf-8') as f:  # 打开JSONL文件
    for line in f:  # 逐行读取
        print(line)
        data = json.loads(line)  # 解析JSON数据
    f.close()