# baseline_agent.py
import json
import os
import sys
import dashscope
from dashscope import Generation
from typing import Dict, Any, List

# ============================================================
# 安全的 API Key 加载（优先级：环境变量 > .env 文件）
# ============================================================
def _load_env_file(env_path: str = ".env") -> None:
    """手动加载 .env 文件（无需 python-dotenv 依赖）"""
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            # 去掉引号
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value

_load_env_file()

class BaselineAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key
        self.documents_cache = {}  # 缓存文档内容

    def load_document(self, doc_id: str) -> str:
        """加载完整文档内容（搜索所有领域目录）"""
        # 覆盖全部5个领域: financial_contracts, financial_reports, insurance, regulatory, research
        domains = ['financial_contracts', 'financial_reports', 'insurance', 'regulatory', 'research']
        possible_paths = []
        for domain in domains:
            possible_paths.append(f'./data/public_dataset_upload/raw/{domain}/{doc_id}.pdf')
            possible_paths.append(f'./data/public_dataset_upload/raw/{domain}/{doc_id}.txt')
            # regulatory 领域还有 txt/ 子目录
            possible_paths.append(f'./data/public_dataset_upload/raw/{domain}/txt/{doc_id}.txt')
            # research 领域的文件格式是 pack2_xxx.pdf
        # 额外通用路径
        possible_paths.extend([
            f'./data/public_dataset_upload/raw/{doc_id}/{doc_id}.pdf',
            f'./data/documents/{doc_id}.txt',
        ])

        for path in possible_paths:
            if os.path.exists(path):
                if path.endswith('.pdf'):
                    try:
                        import pdfplumber
                        with pdfplumber.open(path) as pdf:
                            text = ''
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + '\n'
                            return text
                    except ImportError:
                        return f"PDF文件: {path} (需要安装pdfplumber读取)"
                else:
                    # TXT文件直接读取
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()

        return f"未找到文档: {doc_id}"

    def answer_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """直接输入全文让模型回答"""
        doc_ids = question_data.get('doc_ids', [])

        # 读取所有相关文档
        context_parts = []
        for doc_id in doc_ids:
            doc_content = self.load_document(doc_id)
            context_parts.append(f"[文档 {doc_id}]\n{doc_content}")

        context = '\n\n'.join(context_parts)

        # 构建选项文本
        options_text = ""
        for key in ['A', 'B', 'C', 'D']:
            if key in question_data['options']:
                options_text += f"{key}: {question_data['options'][key]}\n"

        # 限制文档长度，避免超过模型上下文窗口
        max_context_chars = 8000
        context_truncated = context[:max_context_chars]

        # 构造prompt
        prompt = f"""请根据以下文档内容回答问题。

## 文档内容
{context_truncated}

## 问题
{question_data['question']}

## 选项
{options_text}

## 要求
请只输出答案字母。
- 单选题/判断题：输出单个字母（如 A）
- 多选题：按字母顺序输出（如 ABC）

答案："""

        # 调用API
        response = self.call_qwen(prompt)

        return {
            'qid': question_data['qid'],
            'answer': response.strip(),
            'prompt_tokens': self.last_prompt_tokens,
            'completion_tokens': self.last_completion_tokens,
            'total_tokens': self.last_prompt_tokens + self.last_completion_tokens
        }

    def call_qwen(self, prompt: str, max_tokens: int = 500) -> str:
        """调用Qwen API（带完整的错误处理）"""
        try:
            response = Generation.call(
                model='qwen3.7-max',  # 有效的模型名称
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.1,  # 降低随机性
                top_p=0.8,
                result_format='message'
            )

            # 打印调试信息
            print(f"API响应状态: {response.status_code}")

            # 检查响应状态
            if response.status_code != 200:
                print(f"API错误: {response.message}")
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0
                return ""

            # 安全获取 token 使用量
            if hasattr(response, 'usage') and response.usage is not None:
                self.last_prompt_tokens = getattr(response.usage, 'input_tokens', 0)
                self.last_completion_tokens = getattr(response.usage, 'output_tokens', 0)
            else:
                print("警告: response.usage 为 None")
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0

            # 安全获取输出内容
            if (hasattr(response, 'output') and response.output is not None
                    and hasattr(response.output, 'choices')
                    and response.output.choices
                    and len(response.output.choices) > 0):
                return response.output.choices[0].message.content
            else:
                print("无法解析API响应中的输出内容")
                return ""

        except Exception as e:
            print(f"API调用错误: {e}")
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            return ""


# 测试函数
def test_baseline():
    """测试Baseline系统"""
    # 加载题目
    with open('./data/public_dataset_upload/questions/group_a/financial_contracts_questions.json',
              'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"共加载 {len(questions)} 道题目")

    # 从环境变量安全获取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 DASHSCOPE_API_KEY")
        print("请通过以下任一方式设置：")
        print("  1. 创建 .env 文件，写入: DASHSCOPE_API_KEY=你的密钥")
        print("  2. 设置环境变量: export DASHSCOPE_API_KEY=你的密钥")
        print("  3. PowerShell: $env:DASHSCOPE_API_KEY='你的密钥'")
        sys.exit(1)
    agent = BaselineAgent(api_key=api_key)

    # 只测试第一题
    first_question = questions[0]
    print(f"题目: {first_question['question']}")
    print(f"选项: {first_question['options']}")
    print(f"预期答案: {first_question.get('answer', '未知')}")
    print(f"文档IDs: {first_question.get('doc_ids', [])}")
    print("-" * 50)

    # 回答
    result = agent.answer_question(first_question)
    print(f"模型答案: {result['answer']}")
    print(f"Token消耗: {result['total_tokens']} (输入: {result['prompt_tokens']}, 输出: {result['completion_tokens']})")


if __name__ == '__main__':
    test_baseline()
