# financial_qa_agent.py
"""
金融长文本智能问答Agent
整合文档检索、上下文构建、Qwen推理、答案提取与验证

核心流程:
  1. 检索证据 → 2. 证据聚合 → 3. 领域Prompt → 4. Qwen推理 → 5. 答案提取 → 6. 验算
支持: A榜(doc_ids) / B榜(全领域检索) / 多轮推理模式
"""
import os
import re
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashscope
from dashscope import Generation
from typing import Dict, Any, List, Optional, Tuple

from config import config
from domain_prompts import DomainPromptBuilder          # P1-1
from multi_step_reasoning import MultiStepReasoner      # P1-2
from numerical_calculator import FinancialCalculator    # P1-3
from evidence_aggregator import EvidenceAggregator      # P1-4


class FinancialQAAgent:
    """金融长文本智能问答Agent — A榜/B榜双模式"""

    def __init__(self, api_key: str = None, retrieval_system=None,
                 use_multi_step: bool = False, use_b_mode: bool = False):
        """
        初始化Agent

        Args:
            api_key: API Key，不传则从 config 读取
            retrieval_system: 可选，FinancialRetrievalSystem 实例
            use_multi_step: True=启用多轮推理(3轮API调用, 更准确但Token更多)
            use_b_mode: True=B榜模式(无doc_ids时全领域检索)
        """
        self.api_key = api_key or config.api_key
        dashscope.api_key = self.api_key
        self.retrieval_system = retrieval_system
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self._indexed_doc_ids = set()       # A榜: 已索引文档ID
        self._all_docs_indexed = set()      # B榜: 已全量索引的领域

        # P1 模块
        self.use_multi_step = use_multi_step
        self.use_b_mode = use_b_mode
        self.reasoner = MultiStepReasoner(self) if use_multi_step else None
        self.calculator = FinancialCalculator()
        self.aggregator = EvidenceAggregator()

    # ================================================================
    # 主流程
    # ================================================================
    def answer_question(self, question_data: Dict[str, Any],
                        preloaded_docs: Dict[str, str] = None) -> Dict[str, Any]:
        """
        回答单道题目（主入口）— A榜/B榜双模式。

        Args:
            question_data: 题目数据 dict
            preloaded_docs: 可选，{doc_id: text_content} 预加载文档

        Returns:
            {'qid', 'answer', 'prompt_tokens', 'completion_tokens', 'total_tokens'}
        """
        qid = question_data.get('qid')
        domain = question_data.get('domain', 'financial_contracts')
        question = question_data['question']
        options = question_data.get('options', {})
        answer_format = question_data.get('answer_format', 'mcq')
        doc_ids = question_data.get('doc_ids', [])

        domain_name = config.DOMAIN_NAMES.get(domain, domain)

        # P1-5 B榜模式: 无doc_ids → 全领域检索
        if self.use_b_mode or (not doc_ids and not preloaded_docs):
            doc_ids = self._resolve_doc_ids_b_mode(domain)
            print(f"处理: {qid} [{domain_name}] [B榜] {len(doc_ids)} docs")
        else:
            print(f"处理: {qid} [{domain_name}] [A榜]")

        # 阶段1: 索引 + 检索证据
        self._ensure_documents_indexed(doc_ids)
        evidences = self._retrieve_evidences(doc_ids, question, options,
                                             preloaded_docs=preloaded_docs)

        # P1-4: 证据聚合 + 矛盾检测
        agg_result = self.aggregator.aggregate(evidences, question, options)
        if agg_result['contradictions']:
            print(f"  [聚合] 检测到 {len(agg_result['contradictions'])} 处跨文档矛盾")
        evidences = agg_result['aggregated'] or evidences

        # 阶段2: 构建上下文
        context = self._build_context(evidences)

        # 阶段3: P1-1 领域专用Prompt
        prompt = self._build_domain_prompt(question, options, context, domain)

        # P1-2: 多轮推理模式 vs 单轮模式
        if self.use_multi_step and self.reasoner:
            # 多轮推理 (3次API调用, 更准确)
            ms_result = self.reasoner.reason_with_verification(
                question, options, evidences, answer_format
            )
            answer = ms_result['answer']
            print(f"  [多轮] 置信度={ms_result['confidence']:.2f}")
        else:
            # 单轮推理 (默认)
            response = self._call_qwen(prompt, max_tokens=config.MAX_OUTPUT_TOKENS)
            answer = self._extract_answer_from_response(response, answer_format)

        # P1-3: 数值验算
        answer = self._verify_numeric(answer, question, options, evidences)

        # 阶段5: 验证答案格式
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
    # ================================================================
    # P1-5 B榜: 无doc_ids时全领域检索
    # ================================================================
    def _resolve_doc_ids_b_mode(self, domain: str) -> List[str]:
        """B榜模式: 根据domain获取该领域所有可用文档ID"""
        raw_dir = config.get_raw_path(domain)
        if not os.path.exists(raw_dir):
            return []
        doc_ids = []
        for fname in os.listdir(raw_dir):
            name, ext = os.path.splitext(fname)
            if ext.lower() in ('.pdf', '.txt'):
                doc_ids.append(name)
        # 确保全量索引
        self._index_all_docs_in_domain(domain, doc_ids)
        return doc_ids

    def _index_all_docs_in_domain(self, domain: str, doc_ids: List[str]) -> None:
        """B榜: 索引领域内全部文档（解析阶段不计Token）"""
        if domain in self._all_docs_indexed:
            return
        new_ids = [d for d in doc_ids if d not in self._indexed_doc_ids]
        if not new_ids:
            self._all_docs_indexed.add(domain)
            return
        self._ensure_documents_indexed(new_ids)
        self._all_docs_indexed.add(domain)

    # ================================================================
    # P1-3: 数值验算
    # ================================================================
    def _verify_numeric(self, answer: str, question: str,
                         options: Dict[str, str], evidences: List[Dict]) -> str:
        """对数值类选项进行验算修正"""
        # 检测是否涉及数值计算
        num_keywords = ['亿元', '万元', '亿元', '%', '倍', '增长', '下降',
                        '总额', '规模', '金额', '利率', '比例']
        has_numeric = any(kw in question for kw in num_keywords)
        for opt_text in options.values():
            has_numeric = has_numeric or any(kw in opt_text for kw in num_keywords)

        if not has_numeric or not answer or not evidences:
            return answer

        # 对选中的选项做数值验算
        evidence_text = ' '.join(ev.get('content', '') for ev in evidences[:3])
        corrected = []
        changed = False

        for ch in answer:
            if ch in options:
                opt_text = options[ch]
                # 验算该选项的数值是否与证据一致
                ok, msg = self.calculator.verify_option_amount(opt_text, evidence_text)
                if not ok and '金额不一致' in msg:
                    # 数值不匹配，标记但暂不自动修正（避免过度修改）
                    print(f"  [验算] 选项{ch}数值可疑: {msg}")
                corrected.append(ch)
            else:
                corrected.append(ch)

        return ''.join(corrected) if not changed else answer

    # ================================================================
    # P0-3: 文档索引桥接
    # ================================================================
    def _ensure_documents_indexed(self, doc_ids: List[str]) -> None:
        """
        P0-3: 确保文档已索引到 retrieval_system（解析阶段不计Token）。
        若未提供 retrieval_system 则自动创建 FinancialRetrievalSystem 实例。
        """
        if not doc_ids:
            return

        # 自动初始化检索系统（若未提供）
        if self.retrieval_system is None:
            from retrieval_system import FinancialRetrievalSystem
            self.retrieval_system = FinancialRetrievalSystem()

        # 按领域分组 doc_ids（同一领域共享一个索引目录）
        domain_groups: Dict[str, List[str]] = {}
        for doc_id in doc_ids:
            if doc_id in self._indexed_doc_ids:
                continue
            # 找到该文档所属的领域
            pdf_path = self._find_pdf_path(doc_id)
            if not pdf_path:
                continue
            # 从路径中提取领域名: raw/{domain}/{doc_id}.pdf
            for domain in config.DOMAINS:
                if f"/{domain}/" in pdf_path.replace("\\", "/"):
                    if domain not in domain_groups:
                        domain_groups[domain] = []
                    domain_groups[domain].append(doc_id)
                    break

        # 按领域索引
        for domain, ids in domain_groups.items():
            docs_dir = config.get_raw_path(domain)
            print(f"  [索引] {domain}: {len(ids)} 个文档 ({', '.join(ids)})")
            self.retrieval_system.index_documents(docs_dir, ids)
            self._indexed_doc_ids.update(ids)

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
                print(f"  [Cache] {doc_id}")
                text = preloaded_docs[doc_id]
            else:
                pdf_path = self._find_pdf_path(doc_id)
                if not pdf_path:
                    print(f"  未找到文档: {doc_id}")
                    continue
                print(f"  [Read] {doc_id}")
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

            # PDF 文件（解析阶段不计Token，全量读取）
            import pdfplumber
            text = ''
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                max_pages = config.MAX_PDF_PAGES if config.MAX_PDF_PAGES > 0 else total_pages
                pages = pdf.pages[:max_pages]
                for page in pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'

                if config.MAX_PDF_PAGES > 0 and total_pages > config.MAX_PDF_PAGES:
                    print(f"  [Warn] {total_pages}页，截断读取前{config.MAX_PDF_PAGES}页")

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

    # P1-1: 领域专用Prompt（替代原 _build_dynamic_prompt + _get_domain_tips）
    def _build_domain_prompt(self, question: str, options: Dict[str, str],
                              context: str, domain: str) -> str:
        """使用 DomainPromptBuilder 构建5领域专用Prompt"""
        return DomainPromptBuilder.build_prompt(domain, question, options, context)

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
        #   "Answer: A"   "answer: CD"    (英文响应兼容)
        s1_patterns = [
            r'正确答案[：:\s是]+\s*([A-D]+)',
            r'最终答案[：:\s是]+\s*([A-D]+)',
            r'(?:^|\n)\s*答案[：:\s是]+\s*([A-D]+)',
            r'\*\*答案[：:\s]*\*\*\s*([A-D]+)',
            r'(?i)\banswer\s*[：:\s]+\s*([A-D]+)',
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