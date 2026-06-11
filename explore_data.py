# explore_data.py
import json
import os
from collections import Counter

# 根据你的目录结构调整路径
# 你的 questions 文件夹在 data/public_dataset_upload/questions/
# 你的 raw 文件夹在 data/public_dataset_upload/raw/

# 查找 questions_a.json 文件
questions_path = None
for root, dirs, files in os.walk('./data'):
    for file in files:
        if file == 'questions_a.json' or file.endswith('.json'):
            questions_path = os.path.join(root, file)
            break
    if questions_path:
        break

if questions_path:
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # 统计题目分布
    domain_stats = Counter([q['domain'] for q in questions])
    type_stats = Counter([q['answer_format'] for q in questions])
    
    print(f"总题数: {len(questions)}")
    print(f"领域分布: {domain_stats}")
    print(f"题型分布: {type_stats}")
else:
    print("未找到 questions_a.json 文件")
    print("请检查文件路径")

# 查看文档清单
docs_path = './data/public_dataset_upload/raw/'
if os.path.exists(docs_path):
    docs = os.listdir(docs_path)
    print(f"文档数量: {len(docs)}")
    if len(docs) > 0:
        print(f"文档示例: {docs[:5]}")
else:
    print("未找到文档文件夹")