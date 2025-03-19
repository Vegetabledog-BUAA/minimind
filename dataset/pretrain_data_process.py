import json
import re

path = 'dataset\pretrain_hq.jsonl'
new_path = 'dataset\pretrain_hq_new.jsonl'
samples = []
with open(path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line.strip())
        samples.append(data)
    f.close()
with open(new_path, 'w', encoding='utf-8') as f:
    for i in range(len(samples)):
        text = samples[i]['text']
        matches_bos = list(re.finditer(r'<s>', text))
        matches_eos = list(re.finditer(r'</s>', text))
        # 如果没有找到<s>标记，将整个文本作为一个段落
        if not matches_bos:
            print("警告: 未找到<s>标记，将整个文本作为一个段落处理")
            json_line = {"text": text}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        else:
            for i in range(len(matches_bos)):
                start = matches_bos[i].end()  # <s>后的位置
                end = matches_eos[i].start()  # </s>前的位置
                para = text[start:end].strip()
                json_line = {"text": para}
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    f.close()
    