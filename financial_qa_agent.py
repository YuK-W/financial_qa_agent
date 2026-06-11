# financial_qa_agent.py
import os
import re
import json
import dashscope
from dashscope import Generation
from typing import Dict, Any, List, Optional


class FinancialQAAgent:
    """金融长文本智能问答Agent - 整合检索与推理"""
    
    def __init__(self, api_key: str, retrieval_system=None):
        self.api_key = api_key
        dashscope.api_key = api_key
        self.retrieval_system = retrieval_system
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
    
    def answer_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """主流程：回答问题"""
        qid = question_data.get('qid')
        domain = question_data.get('domain', 'financial_contracts')
        question = question_data['question']
        options = question_data['options']
        answer_format = question_data.get('answer_format', 'multi')
        doc_ids = question_data.get('doc_ids', [])
        
        print(f"处理问题: {qid}")
        
        # 阶段1：检索证据（使用doc_ids直接读取）
        evidences = self._retrieve_evidences(doc_ids, question, options)
        
        # 阶段2：构建上下文
        context = self._build_context(evidences)
        
        # 阶段3：构建提示词并调用模型
        prompt = self._build_prompt(question, options, context, answer_format)
        response = self._call_qwen(prompt)
        
        # 阶段4：提取并验证答案
        answer = self._extract_answer(response, answer_format)
        validated_answer = self._validate_answer(answer, answer_format)
        
        return {
            'qid': qid,
            'answer': validated_answer,
            'prompt_tokens': self.last_prompt_tokens,
            'completion_tokens': self.last_completion_tokens,
            'total_tokens': self.last_prompt_tokens + self.last_completion_tokens
        }
    
    def _retrieve_evidences(self, doc_ids: List[str], question: str, 
                            options: Dict[str, str]) -> List[Dict]:
        """从指定文档中检索证据片段"""
        evidences = []
        
        for doc_id in doc_ids:
            # 查找PDF文件
            pdf_path = self._find_pdf_path(doc_id)
            if not pdf_path:
                print(f"  未找到文档: {doc_id}")
                continue
            
            # 读取PDF文本
            text = self._read_pdf(pdf_path)
            if not text:
                continue
            
            # 提取相关片段（基于问题和选项的关键词）
            relevant_chunks = self._extract_relevant_chunks(text, question, options)
            for chunk in relevant_chunks[:3]:  # 每个文档取最多3个片段
                evidences.append({
                    'doc_id': doc_id,
                    'content': chunk
                })
        
        return evidences[:10]  # 总共最多10个片段
    
    def _find_pdf_path(self, doc_id: str) -> Optional[str]:
        """查找PDF文件路径"""
        # 先尝试 financial_contracts 目录（当前数据集）
        base_paths = [
            f'./data/public_dataset_upload/raw/financial_contracts/{doc_id}.pdf',
            f'./data/public_dataset_upload/raw/{doc_id}/{doc_id}.pdf',
        ]
        
        for path in base_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _read_pdf(self, pdf_path: str) -> str:
        """读取PDF文本"""
        try:
            import pdfplumber
            text = ''
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            return text
        except Exception as e:
            print(f"  PDF读取错误: {e}")
            return ''
    
    def _extract_relevant_chunks(self, text: str, question: str, 
                                  options: Dict[str, str]) -> List[str]:
        """提取与问题和选项相关的文本片段"""
        # 合并关键词
        keywords = set()
        keywords.update(self._extract_keywords(question))
        for opt_text in options.values():
            keywords.update(self._extract_keywords(opt_text))
        
        # 按段落切分
        paragraphs = text.split('\n')
        relevant = []
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue
            
            # 计算相关度
            score = 0
            for kw in keywords:
                if kw in para:
                    score += 1
            
            if score > 0:
                # 限制片段长度
                if len(para) > 800:
                    para = para[:800] + '...'
                relevant.append((score, para))
        
        # 按相关度排序
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in relevant[:5]]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单分词：按常见分隔符切分
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
        keywords = [w for w in words if len(w) >= 2]
        return keywords
    
    def _build_context(self, evidences: List[Dict]) -> str:
        """构建上下文"""
        if not evidences:
            return "未找到相关文档内容。"
        
        context_parts = []
        for ev in evidences:
            context_parts.append(f"[{ev['doc_id']}]\n{ev['content']}")
        
        context = '\n\n'.join(context_parts)
        
        # 限制总长度（约3000字符，对应约2000 tokens）
        if len(context) > 3000:
            context = context[:3000] + '...'
        
        return context
    
    def _build_prompt(self, question: str, options: Dict[str, str], 
                      context: str, answer_format: str) -> str:
        """构建提示词"""
        options_text = ""
        for key in ['A', 'B', 'C', 'D']:
            if key in options:
                options_text += f"{key}. {options[key]}\n"
        
        # 根据题型调整输出格式要求
        if answer_format == 'mcq' or answer_format == 'tf':
            output_instruction = "请只输出一个字母（如 A）"
        else:
            output_instruction = "请按字母顺序输出答案（如 ABC）"
        
        return f"""你是一个金融合同分析专家。请根据以下文档内容回答问题。

## 文档内容
{context}

## 问题
{question}

## 选项
{options_text}

## 要求
{output_instruction}

请直接输出答案，不要输出其他解释。"""

    def _call_qwen(self, prompt: str, max_tokens: int = 300) -> str:
        """调用Qwen API"""
        try:
            response = Generation.call(
                model='qwen3.7-max',
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.8,
                result_format='message'
            )
            
            # 记录Token
            if hasattr(response, 'usage') and response.usage:
                self.last_prompt_tokens = getattr(response.usage, 'input_tokens', 0)
                self.last_completion_tokens = getattr(response.usage, 'output_tokens', 0)
            else:
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0
            
            # 提取输出
            if (hasattr(response, 'output') and response.output and 
                hasattr(response.output, 'choices') and response.output.choices):
                return response.output.choices[0].message.content.strip()
            
            return ""
            
        except Exception as e:
            print(f"API调用错误: {e}")
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            return ""
    
    def _extract_answer(self, response: str, answer_format: str) -> str:
        """从响应中提取答案"""
        if not response:
            return ''
        
        # 提取大写字母
        letters = re.findall(r'[A-D]', response.upper())
        
        if answer_format == 'mcq' or answer_format == 'tf':
            return letters[0] if letters else 'A'
        else:
            return ''.join(sorted(set(letters))) if letters else ''
    
    def _validate_answer(self, answer: str, answer_format: str) -> str:
        """验证答案格式"""
        if not answer:
            return 'A'
        
        if answer_format == 'mcq' or answer_format == 'tf':
            if answer in ['A', 'B', 'C', 'D']:
                return answer
            return 'A'
        
        elif answer_format == 'multi':
            valid = [c for c in answer if c in 'ABCD']
            return ''.join(sorted(set(valid))) if valid else ''
        
        return answer


# ============================================================
# 测试函数
# ============================================================
def test_agent():
    """测试Agent"""
    # 加载题目
    with open('./data/public_dataset_upload/questions/group_a/financial_contracts_questions.json',
              'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"共加载 {len(questions)} 道题目\n")
    
    # 初始化Agent
    api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-874c742710dd4845806a40ddfc3e03af')
    agent = FinancialQAAgent(api_key=api_key)
    
    # 测试第一题
    first_question = questions[0]
    print(f"题目: {first_question['question']}")
    print(f"选项: {first_question['options']}")
    print(f"文档IDs: {first_question.get('doc_ids', [])}")
    print("-" * 50)
    
    # 回答
    result = agent.answer_question(first_question)
    print(f"\n模型答案: {result['answer']}")
    print(f"Token消耗: {result['total_tokens']} (输入: {result['prompt_tokens']}, 输出: {result['completion_tokens']})")


if __name__ == '__main__':
    test_agent()