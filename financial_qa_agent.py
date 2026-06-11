# financial_qa_agent.py
"""
金融长文本智能问答Agent
整合文档检索、上下文构建、Qwen推理、答案提取与验证

核心流程:
  1. 检索证据 → 2. 构建上下文 → 3. 调用 Qwen → 4. 提取答案 → 5. 验证格式
"""
import os
import re
import json
import dashscope
from dashscope import Generation
from typing import Dict, Any, List, Optional


class FinancialQAAgent:
    """金融长文本智能问答Agent - 优化版"""
    
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
        
        # 阶段1：检索证据
        evidences = self._retrieve_evidences(doc_ids, question, options)
        
        # 阶段2：构建上下文
        context = self._build_context(evidences)
        
        # 阶段3：构建强化提示词
        prompt = self._build_enhanced_prompt(question, options, context, doc_ids)
        
        # 阶段4：调用模型
        response = self._call_qwen(prompt, max_tokens=800)
        
        # 阶段5：提取答案
        answer = self._extract_answer_from_response(response, answer_format)
        
        # 阶段6：验证答案
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
        
        # 提取问题中的关键实体
        key_entities = self._extract_key_entities(question, options)
        print(f"  关键实体: {key_entities}")
        
        for doc_id in doc_ids:
            pdf_path = self._find_pdf_path(doc_id)
            if not pdf_path:
                print(f"  未找到文档: {doc_id}")
                continue
            
            text = self._read_pdf(pdf_path)
            if not text:
                continue
            
            # 提取相关片段
            relevant_chunks = self._extract_targeted_chunks(text, key_entities, doc_id)
            for chunk in relevant_chunks[:5]:
                evidences.append({
                    'doc_id': doc_id,
                    'content': chunk
                })
        
        print(f"  找到 {len(evidences)} 个证据片段")
        return evidences
    
    def _extract_key_entities(self, question: str, options: Dict[str, str]) -> Dict[str, str]:
        """提取关键实体"""
        entities = {}
        
        # 从选项中提取关键信息
        for key, opt_text in options.items():
            # 提取公司名称
            company_match = re.search(r'([\u4e00-\u9fa5]+(?:集团|股份|有限|公司))', opt_text)
            if company_match:
                entities[f'company_{key}'] = company_match.group(1)
            
            # 提取信用评级
            rating_match = re.search(r'(AAA|AA\+|AA|AA-|A\+|A)', opt_text)
            if rating_match:
                entities[f'rating_{key}'] = rating_match.group(1)
            
            # 提取中介机构
            if '证券' in opt_text or '银行' in opt_text:
                org_match = re.search(r'([\u4e00-\u9fa5]+(?:证券|股份|银行))', opt_text)
                if org_match:
                    entities[f'org_{key}'] = org_match.group(1)
        
        return entities
    
    def _extract_targeted_chunks(self, text: str, entities: Dict[str, str], 
                                  doc_id: str) -> List[str]:
        """针对性地提取包含关键实体的段落"""
        chunks = []
        
        # 按段落分割
        paragraphs = text.split('\n')
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue
            
            # 检查是否包含关键实体
            score = 0
            matched_entities = []
            
            for entity_name, entity_value in entities.items():
                if entity_value and entity_value in para:
                    score += 3
                    matched_entities.append(entity_value)
            
            # 额外检查常见关键词
            keywords = ['发行人', '信用评级', '受托管理人', '发行规模', '主体信用', 
                       'AAA', 'AA+', '主承销商', '募集说明书']
            for kw in keywords:
                if kw in para:
                    score += 1
            
            if score > 0:
                # 保留上下文（前后扩展）
                if len(para) > 1000:
                    para = para[:1000] + '...'
                chunks.append((score, para, matched_entities))
        
        # 按相关度排序
        chunks.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in chunks[:10]]
    
    def _find_pdf_path(self, doc_id: str) -> Optional[str]:
        """查找PDF文件路径"""
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
                # 只读取前30页（合同关键信息通常在前部）
                for page in pdf.pages[:30]:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            return text
        except Exception as e:
            print(f"  PDF读取错误: {e}")
            return ''
    
    def _build_context(self, evidences: List[Dict]) -> str:
        """构建上下文"""
        if not evidences:
            return "未找到相关文档内容。"
        
        context_parts = []
        for ev in evidences:
            context_parts.append(f"【文档 {ev['doc_id']}】\n{ev['content']}")
        
        context = '\n\n---\n\n'.join(context_parts)
        
        # 限制总长度
        if len(context) > 5000:
            context = context[:5000] + '...'
        
        return context
    
    def _build_enhanced_prompt(self, question: str, options: Dict[str, str], 
                                context: str, doc_ids: List[str]) -> str:
        """构建增强版提示词 - 逐项分析"""
        
        options_text = ""
        for key in ['A', 'B', 'C', 'D']:
            if key in options:
                options_text += f"{key}. {options[key]}\n"
        
        return f"""你是一个金融合同分析专家。请根据以下两份募集说明书的内容，逐项判断每个选项的正确性。

## 文档1 (text01 - 广晟控股集团)
{self._get_doc_summary(context, 'text01')}

## 文档2 (text02 - 深圳租赁)
{self._get_doc_summary(context, 'text02')}

## 问题
{question}

## 选项
{options_text}

## 逐项分析（请按以下格式输出）

A选项分析：
判断：[正确/错误]
依据：从文档中找到的具体证据

B选项分析：
判断：[正确/错误]
依据：从文档中找到的具体证据

C选项分析：
判断：[正确/错误]
依据：从文档中找到的具体证据

D选项分析：
判断：[正确/错误]
依据：从文档中找到的具体证据

## 最终答案
正确答案："""
    
    def _get_doc_summary(self, context: str, doc_id: str) -> str:
        """从上下文中提取指定文档的摘要"""
        # 查找该文档的相关段落
        pattern = rf'【文档 {doc_id}】\n(.*?)(?=【文档|$)'
        match = re.search(pattern, context, re.DOTALL)
        if match:
            content = match.group(1)[:1500]
            return content
        return f"未找到文档 {doc_id} 的内容"
    
    def _call_qwen(self, prompt: str, max_tokens: int = 800) -> str:
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
            
            if hasattr(response, 'usage') and response.usage:
                self.last_prompt_tokens = getattr(response.usage, 'input_tokens', 0)
                self.last_completion_tokens = getattr(response.usage, 'output_tokens', 0)
            else:
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0
            
            if (hasattr(response, 'output') and response.output and 
                hasattr(response.output, 'choices') and response.output.choices):
                return response.output.choices[0].message.content
            
            return ""
            
        except Exception as e:
            print(f"API调用错误: {e}")
            return ""
    
    def _extract_answer_from_response(self, response: str, answer_format: str) -> str:
        """从响应中提取答案"""
        if not response:
            return ''
        
        # 查找"正确答案："后面的内容
        match = re.search(r'正确答案[：:]\s*([A-D]+)', response)
        if match:
            return match.group(1)
        
        # 查找选项分析中标记为正确的
        correct = []
        for opt in ['A', 'B', 'C', 'D']:
            pattern = rf'{opt}选项分析.*?判断：[正确✓]*'
            if re.search(pattern, response, re.DOTALL):
                correct.append(opt)
        
        if correct:
            return ''.join(sorted(correct))
        
        # 最后尝试提取所有大写字母
        letters = re.findall(r'[A-D]', response.upper())
        if letters:
            if answer_format == 'multi':
                return ''.join(sorted(set(letters)))
            return letters[0]
        
        return ''
    
    def _validate_answer(self, answer: str, answer_format: str) -> str:
        """验证答案格式"""
        if not answer:
            return 'A'
        
        if answer_format == 'multi':
            valid = [c for c in answer if c in 'ABCD']
            return ''.join(sorted(set(valid))) if valid else ''
        else:
            if answer in 'ABCD':
                return answer
            return 'A'


def test_agent():
    """测试Agent"""
    with open('./data/public_dataset_upload/questions/group_a/financial_contracts_questions.json',
              'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"共加载 {len(questions)} 道题目\n")
    
    api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-874c742710dd4845806a40ddfc3e03af')
    agent = FinancialQAAgent(api_key=api_key)
    
    # 测试第一题
    first_question = questions[0]
    print(f"题目: {first_question['question']}")
    print(f"选项: {first_question['options']}")
    print(f"文档IDs: {first_question.get('doc_ids', [])}")
    print("-" * 50)
    
    result = agent.answer_question(first_question)
    print(f"\n模型答案: {result['answer']}")
    print(f"Token消耗: {result['total_tokens']} (输入: {result['prompt_tokens']}, 输出: {result['completion_tokens']})")


if __name__ == '__main__':
    test_agent()