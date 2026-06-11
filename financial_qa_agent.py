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
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashscope
from dashscope import Generation
from typing import Dict, Any, List, Optional, Tuple

from config import config  # Bug 1 修复: 统一配置


class FinancialQAAgent:
    """金融长文本智能问答Agent"""

    def __init__(self, api_key: str = None, retrieval_system=None):
        """
        初始化Agent

        Args:
            api_key: API Key，不传则从 config 读取
            retrieval_system: 可选，FinancialRetrievalSystem 实例
        """
        self.api_key = api_key or config.api_key
        dashscope.api_key = self.api_key
        self.retrieval_system = retrieval_system
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    # ================================================================
    # 主流程
    # ================================================================
    def answer_question(self, question_data: Dict[str, Any],
                        preloaded_docs: Dict[str, str] = None) -> Dict[str, Any]:
        """
        回答单道题目（主入口）

        Args:
            question_data: 题目数据 dict，含 qid, question, options, doc_ids 等
            preloaded_docs: 可选，{doc_id: text_content} 预加载的文档内容，
                           传入后跳过 PDF 读取，直接使用缓存内容

        Returns:
            {'qid', 'answer', 'prompt_tokens', 'completion_tokens', 'total_tokens'}
        """
        qid = question_data.get('qid')
        domain = question_data.get('domain', 'financial_contracts')
        question = question_data['question']
        options = question_data.get('options', {})
        answer_format = question_data.get('answer_format', 'mcq')
        doc_ids = question_data.get('doc_ids', [])

        print(f"处理问题: {qid}  [{config.DOMAIN_NAMES.get(domain, domain)}]")

        # 阶段1：检索证据（若有 preloaded_docs 则跳过 PDF 读取）
        evidences = self._retrieve_evidences(doc_ids, question, options,
                                             preloaded_docs=preloaded_docs)

        # 阶段2：构建上下文
        context = self._build_context(evidences)

        # 阶段3：构建动态 Prompt（Bug 4 修复: 不再硬编码文档名）
        prompt = self._build_dynamic_prompt(question, options, context, domain)

        # 阶段4：调用模型
        response = self._call_qwen(prompt, max_tokens=config.MAX_OUTPUT_TOKENS)

        # 阶段5：提取答案（Bug 2 修复: 多策略提取）
        answer = self._extract_answer_from_response(response, answer_format)

        # 阶段6：验证答案格式
        validated_answer = self._validate_answer(answer, answer_format)

        return {
            'qid': qid,
            'answer': validated_answer,
            'prompt_tokens': self.last_prompt_tokens,
            'completion_tokens': self.last_completion_tokens,
            'total_tokens': self.last_prompt_tokens + self.last_completion_tokens
        }

    # ================================================================
    # 证据检索
    # ================================================================
    def _retrieve_evidences(self, doc_ids: List[str], question: str,
                            options: Dict[str, str],
                            preloaded_docs: Dict[str, str] = None) -> List[Dict]:
        """
        从指定文档中检索证据片段。

        优先使用 retrieval_system（如果已初始化并索引），
        否则回退到简单的关键词匹配。

        Args:
            doc_ids: 文档 ID 列表
            question: 问题文本
            options: 选项 dict
            preloaded_docs: 预加载的文档内容 {doc_id: text}

        Returns:
            [{'doc_id': str, 'content': str, 'relevance': float}, ...]
        """
        evidences = []
        preloaded_docs = preloaded_docs or {}

        # 尝试使用检索系统（如果已提供）
        if self.retrieval_system and self.retrieval_system.chunks:
            try:
                results = self.retrieval_system.retrieve(
                    question, options, top_k=config.RETRIEVAL_TOP_K
                )
                for r in results:
                    evidences.append({
                        'doc_id': r.get('doc_id', 'unknown'),
                        'content': r.get('content', ''),
                        'relevance': r.get('relevance', 0)
                    })
                print(f"  [检索系统] 找到 {len(evidences)} 个证据片段")
                return evidences[:config.EVIDENCE_MAX_TOTAL]
            except Exception as e:
                print(f"  检索系统异常，回退到关键词检索: {e}")

        # 回退：逐文档提取关键段落
        key_entities = self._extract_key_entities(question, options)

        for doc_id in doc_ids:
            # Bug 6 修复: 优先使用预加载文档
            if doc_id in preloaded_docs:
                print(f"  📦 使用缓存文档: {doc_id}")
                text = preloaded_docs[doc_id]
            else:
                pdf_path = self._find_pdf_path(doc_id)
                if not pdf_path:
                    print(f"  未找到文档: {doc_id}")
                    continue
                print(f"  📄 读取文档: {doc_id}")
                text = self._read_pdf(pdf_path)

            if not text:
                continue

            relevant_chunks = self._extract_targeted_chunks(text, key_entities, doc_id)
            for chunk, score, matched in relevant_chunks[:config.EVIDENCE_MAX_PER_DOC]:
                evidences.append({
                    'doc_id': doc_id,
                    'content': chunk,
                    'relevance': score
                })

        # 按相关度排序
        evidences.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        print(f"  找到 {len(evidences)} 个证据片段")
        return evidences[:config.EVIDENCE_MAX_TOTAL]

    # ================================================================
    # Bug 3 修复: _find_pdf_path 覆盖全部 5 个领域
    # ================================================================
    def _find_pdf_path(self, doc_id: str) -> Optional[str]:
        """
        在全部 5 个领域的 raw 目录下搜索 PDF 文件。

        搜索策略:
          1. 遍历 5 个领域目录，匹配 {doc_id}.pdf
          2. 兜底：检查 {doc_id}/{doc_id}.pdf 子路径
          3. 支持 .txt 格式作为备选

        Args:
            doc_id: 文档ID，如 'text01'

        Returns:
            找到的文件绝对路径，或 None
        """
        # 策略1：遍历全部 5 个领域目录（Bug 3 核心修复）
        for domain in config.DOMAINS:
            domain_dir = config.get_raw_path(domain)

            # 1a: 标准路径 domain/doc_id.pdf
            pdf_path = os.path.join(domain_dir, f"{doc_id}.pdf")
            if os.path.exists(pdf_path):
                return pdf_path

            # 1b: txt 备用路径（部分 regulatory 文件在 txt/ 子目录）
            txt_path = os.path.join(domain_dir, f"{doc_id}.txt")
            if os.path.exists(txt_path):
                return txt_path

            # 1c: regulatory 领域的 txt/ 子目录
            txt_sub_path = os.path.join(domain_dir, "txt", f"{doc_id}.txt")
            if os.path.exists(txt_sub_path):
                return txt_sub_path

        # 策略2：兜底 —— doc_id 作为子目录名
        # 部分数据集文件结构为 raw/{doc_id}/{doc_id}.pdf
        fallback_path = os.path.join(config.raw_dir, doc_id, f"{doc_id}.pdf")
        if os.path.exists(fallback_path):
            return fallback_path

        return None

    def _read_pdf(self, file_path: str) -> str:
        """
        读取 PDF/TXT 文件内容。

        Args:
            file_path: 文件路径（支持 .pdf 和 .txt）

        Returns:
            文本内容字符串，读取失败返回空字符串
        """
        try:
            # TXT 文件直接读取
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            # PDF 文件
            import pdfplumber
            text = ''
            with pdfplumber.open(file_path) as pdf:
                pages = pdf.pages[:config.MAX_PDF_PAGES]
                for page in pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'

            if not text and len(pdf.pages) > config.MAX_PDF_PAGES:
                print(f"  ⚠️ 文档超过{config.MAX_PDF_PAGES}页，只读取了前{config.MAX_PDF_PAGES}页")

            return text
        except Exception as e:
            print(f"  PDF读取错误 [{file_path}]: {e}")
            return ''

    def _extract_key_entities(self, question: str, options: Dict[str, str]) -> Dict[str, str]:
        """从问题+选项中提取关键实体（公司名、金额、评级等）"""
        entities = {}

        for key, opt_text in options.items():
            # 提取公司名称
            company_match = re.search(r'([一-龥]+(?:集团|股份|有限|公司))', opt_text)
            if company_match:
                entities[f'company_{key}'] = company_match.group(1)

            # 提取信用评级
            rating_match = re.search(r'(AAA|AA\+|AA|AA-|A\+|A)', opt_text)
            if rating_match:
                entities[f'rating_{key}'] = rating_match.group(1)

            # 提取中介机构
            if '证券' in opt_text or '银行' in opt_text:
                org_match = re.search(r'([一-龥]+(?:证券|股份|银行))', opt_text)
                if org_match:
                    entities[f'org_{key}'] = org_match.group(1)

        # 从问题中也提取关键实体
        q_companies = re.findall(r'([一-龥]+(?:集团|股份|有限|公司))', question)
        for i, c in enumerate(q_companies):
            entities[f'q_company_{i}'] = c

        return entities

    def _extract_targeted_chunks(self, text: str, entities: Dict[str, str],
                                  doc_id: str) -> List[Tuple[str, int, List[str]]]:
        """提取包含关键实体的段落，按相关度排序"""
        paragraphs = text.split('\n')
        scored_chunks = []

        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue

            score = 0
            matched = []

            for entity_name, entity_value in entities.items():
                if entity_value and entity_value in para:
                    score += 3
                    matched.append(entity_value)

            # 金融合同常见关键词加分
            keywords = ['发行人', '信用评级', '受托管理人', '发行规模', '主体信用',
                        'AAA', 'AA+', '主承销商', '募集说明书', '担保', '票面利率']
            for kw in keywords:
                if kw in para:
                    score += 1

            if score > 0:
                if len(para) > 1000:
                    para = para[:1000] + '...'
                scored_chunks.append((para, score, matched))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks

    # ================================================================
    # Bug 4 修复: 动态 Prompt 构建（不再硬编码文档名）
    # ================================================================
    def _build_context(self, evidences: List[Dict]) -> str:
        """
        将证据片段拼接为上下文。
        每个证据标注文档来源，按相关度排序。
        """
        if not evidences:
            return "未找到相关文档内容。"

        context_parts = []
        for i, ev in enumerate(evidences, 1):
            doc_id = ev.get('doc_id', 'unknown')
            content = ev.get('content', '')
            context_parts.append(f"【证据{i} - 来源: {doc_id}】\n{content}")

        context = '\n\n---\n\n'.join(context_parts)

        # 限制总长度
        if len(context) > config.MAX_CONTEXT_CHARS:
            context = context[:config.MAX_CONTEXT_CHARS] + '\n\n... [上下文已截断]'

        return context

    def _build_dynamic_prompt(self, question: str, options: Dict[str, str],
                               context: str, domain: str) -> str:
        """
        Bug 4 修复: 动态构建 Prompt。

        - 不再硬编码 text01/text02
        - 支持任意数量的证据片段
        - 按领域自动调整分析要点
        """
        # 构建选项文本
        options_text = "\n".join(
            f"{key}. {options[key]}" for key in ['A', 'B', 'C', 'D']
            if key in options
        )

        # 领域特定的分析要点
        domain_tips = self._get_domain_tips(domain)

        domain_name = config.DOMAIN_NAMES.get(domain, "金融文档")

        # 动态统计上下文中的证据数量
        evidence_count = context.count("【证据")

        return f"""你是一个{domain_name}分析专家。请严格依据以下文档内容回答问题。

## 文档证据（共 {evidence_count} 个相关片段）
{context}

## 问题
{question}

## 选项
{options_text}

## 分析要点
{domain_tips}

## 输出要求
1. 逐项分析每个选项的正确性，引用文档中的具体证据
2. 最后以"正确答案："开头，输出最终答案
   - 单选题/判断题：输出单个字母（如 A）
   - 多选题：按字母顺序输出，无分隔符（如 ABC）

请开始分析："""

    def _get_domain_tips(self, domain: str) -> str:
        """根据领域返回对应的分析要点（Bug 4 配套: 领域适配）"""
        tips = {
            'financial_contracts': (
                "1. 核对发行主体、发行规模、信用评级是否与文档一致\n"
                "2. 注意受托管理人、主承销商等中介机构名称\n"
                "3. 检查债券期限、票面利率等关键数值"
            ),
            'financial_reports': (
                "1. 核对财务数据（营业收入、净利润、总资产等）是否与报表一致\n"
                "2. 注意同比/环比变化方向与幅度\n"
                "3. 检查合并报表范围与会计政策"
            ),
            'insurance': (
                "1. 仔细核对保险责任的范围和限制条件\n"
                "2. 注意除外责任条款\n"
                "3. 计算现金价值时注意缴费年限和退保时间\n"
                "4. 身故保险金注意区分意外身故和疾病身故"
            ),
            'regulatory': (
                "1. 严格匹配法条原文，不要自行解释\n"
                "2. 注意'应当'、'可以'、'必须'等强制性词汇\n"
                "3. 注意适用条件和例外情形\n"
                "4. 注意法规的施行日期和适用范围"
            ),
            'research': (
                "1. 核对研报中的评级（买入/增持/中性/减持/卖出）\n"
                "2. 注意目标价与当前价的对比\n"
                "3. 关注盈利预测与风险提示"
            ),
        }
        return tips.get(domain, "1. 逐项核对选项与文档原文是否一致\n2. 注意数值、日期、名称等关键信息")

    # ================================================================
    # API 调用
    # ================================================================
    def _call_qwen(self, prompt: str, max_tokens: int = None) -> str:
        """调用 Qwen API"""
        max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        try:
            response = Generation.call(
                model=config.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.8,
                result_format='message'
            )

            if response.status_code != 200:
                print(f"  API错误 [{response.status_code}]: {response.message}")
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0
                return ""

            if hasattr(response, 'usage') and response.usage:
                self.last_prompt_tokens = getattr(response.usage, 'input_tokens', 0)
                self.last_completion_tokens = getattr(response.usage, 'output_tokens', 0)
            else:
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0

            if (hasattr(response, 'output') and response.output
                    and hasattr(response.output, 'choices')
                    and response.output.choices):
                return response.output.choices[0].message.content

            return ""

        except Exception as e:
            print(f"  API调用错误: {e}")
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            return ""

    # ================================================================
    # Bug 2 修复: 多策略答案提取
    # ================================================================
    def _extract_answer_from_response(self, response: str, answer_format: str) -> str:
        """
        从模型响应中提取答案 —— 4 层策略递进。

        策略优先级:
          S1: "正确答案：X" 模式匹配（最可靠的显式声明）
          S2: 逐选项判断提取（模型在分析中标记了哪些选项正确）
          S3: 选项字母频率统计（从全文提取所有 A-D 字母）
          S4: Fallback —— 返回空字符串，由 _validate_answer 兜底

        Args:
            response: Qwen 模型返回的原始文本
            answer_format: 'mcq'(单选), 'tf'(判断), 'multi'(多选)

        Returns:
            提取到的答案字符串，如 'A', 'ABC', ''
        """
        if not response or not response.strip():
            return ''

        # ================================================================
        # S1: "正确答案：X" 显式声明（最可靠）
        # ================================================================
        # 匹配模式:
        #   "正确答案：A"  "正确答案: ABC"  "正确答案是 B"
        #   "最终答案：CD" "答案：A"       "**答案：B**"
        s1_patterns = [
            r'正确答案[：:\s是]+\s*([A-D]+)',
            r'最终答案[：:\s是]+\s*([A-D]+)',
            r'(?:^|\n)\s*答案[：:\s是]+\s*([A-D]+)',
            r'\*\*答案[：:\s]*\*\*\s*([A-D]+)',
        ]
        for pattern in s1_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip().upper()
                # 确保只含 A-D
                answer = ''.join(c for c in answer if c in 'ABCD')
                if answer:
                    print(f"  [答案提取-S1] '{answer}' (via 显式声明)")
                    if answer_format in ('mcq', 'tf'):
                        return answer[0]
                    return ''.join(sorted(set(answer)))

        # ================================================================
        # S2: 逐选项判断提取（分析型响应）
        # ================================================================
        # 模型常见的逐项分析格式:
        #   "A选项分析：判断：正确"  → A 正确
        #   "B选项：错误，因为..."   → B 错误
        correct_opts = []
        for opt in ['A', 'B', 'C', 'D']:
            # 匹配 "{opt}选项...判断：正确" 或 "{opt}...正确"
            # 注意: 这里用 f-string 动态构建正则，而非字面量 {opt}
            patterns = [
                rf'{opt}[选项]?[^。\n]*?[判断：:\s]*(?:正确|✓|✅|√)',
                rf'{opt}[选项]?[^。\n]*?(?:是正确的|为正确|是对的)',
                rf'[判断：:\s]*正确[^。\n]*?{opt}[选项]?',
            ]
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    correct_opts.append(opt)
                    break
            else:
                # 宽松匹配：看该选项段落整体倾向
                block_match = re.search(
                    rf'{opt}\s*选项[^A-D]*?(?={chr(ord(opt)+1)}\s*选项|正确答案|最终答案|$)',
                    response, re.DOTALL
                )
                if block_match:
                    block = block_match.group()
                    pos_words = len(re.findall(r'正确|✓|✅|符合|一致|是对的', block))
                    neg_words = len(re.findall(r'错误|✗|❌|不符合|不一致|是不对的', block))
                    if pos_words > neg_words:
                        correct_opts.append(opt)

        if correct_opts:
            answer = ''.join(sorted(set(correct_opts)))
            print(f"  [答案提取-S2] '{answer}' (via 逐项判断, 正确选项: {correct_opts})")
            if answer_format in ('mcq', 'tf'):
                return answer[0]
            return answer

        # ================================================================
        # S3: 字母频率统计（适用于模型直接输出字母的简短响应）
        # ================================================================
        # 从最后一段或最后 200 字符中集中提取
        tail = response[-200:] if len(response) > 200 else response
        all_letters = re.findall(r'[A-D]', tail.upper())

        if all_letters:
            if answer_format in ('mcq', 'tf'):
                # 单选/判断：取最后出现的单个字母
                answer = all_letters[-1]
                print(f"  [答案提取-S3] '{answer}' (via 末尾字母)")
                return answer
            else:
                # 多选：取去重排序后的所有字母
                answer = ''.join(sorted(set(all_letters)))
                print(f"  [答案提取-S3] '{answer}' (via 字母频率)")
                return answer

        # ================================================================
        # S4: Fallback
        # ================================================================
        print(f"  [答案提取-S4] 无法提取，返回空字符串")
        return ''

    def _validate_answer(self, answer: str, answer_format: str) -> str:
        """验证答案格式，确保合规"""
        if not answer:
            # 无答案时默认返回 A（宁可答错也不能格式错误）
            return 'A' if answer_format in ('mcq', 'tf') else ''

        if answer_format == 'multi':
            # 多选题：去重 + 排序 + 只保留 A-D
            valid = sorted(set(c for c in answer.upper() if c in 'ABCD'))
            return ''.join(valid) if valid else ''
        else:
            # 单选/判断：取第一个有效字母
            first = answer[0].upper() if answer else 'A'
            return first if first in 'ABCD' else 'A'


# ================================================================
# 测试函数
# ================================================================
def test_agent():
    """测试 Agent 基础功能"""
    # 查找题目文件
    question_path = config.get_question_path("group_a", "financial_contracts")
    if not os.path.exists(question_path):
        print(f"题目文件不存在: {question_path}")
        return

    with open(question_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"共加载 {len(questions)} 道题目\n")

    agent = FinancialQAAgent()

    # 测试第一题
    first_question = questions[0]
    print(f"题目: {first_question['question']}")
    print(f"选项: {first_question['options']}")
    print(f"文档IDs: {first_question.get('doc_ids', [])}")
    print("-" * 50)

    result = agent.answer_question(first_question)
    print(f"\n模型答案: {result['answer']}")
    print(f"Token消耗: {result['total_tokens']} "
          f"(输入: {result['prompt_tokens']}, 输出: {result['completion_tokens']})")


if __name__ == '__main__':
    test_agent()