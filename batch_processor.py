# batch_processor.py
"""
批量处理模块

核心优化策略:
  1. 按文档分组题目，相同文档只读取一次 → 减少 I/O
  2. 预加载文档内容后传入 Agent → 跳过重复的 PDF 解析
  3. 单题失败不影响其他题目 → 隔离错误
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, List
from collections import OrderedDict

from config import config


class BatchProcessor:
    """
    批量处理多道题目，通过文档预加载 + 复用减少 I/O 和 Token 浪费。

    使用方式:
        from financial_qa_agent import FinancialQAAgent
        agent = FinancialQAAgent()
        processor = BatchProcessor(agent)
        results = processor.batch_answer(questions)
    """

    def __init__(self, agent):
        """
        Args:
            agent: FinancialQAAgent 实例（需实现 answer_question 方法）
        """
        self.agent = agent
        # 文档内容缓存：{doc_id: text_content}
        self._doc_cache: Dict[str, str] = {}
        # 缓存统计
        self._cache_hits = 0
        self._cache_misses = 0

    # ================================================================
    # Bug 5 修复: 实现 load_document 方法
    # ================================================================
    def load_document(self, doc_id: str) -> str:
        """
        加载文档内容，内置缓存机制。

        流程:
          1. 检查缓存 → 命中则直接返回
          2. 未命中 → 搜索 PDF 文件 → 读取 → 入缓存 → 返回

        Args:
            doc_id: 文档 ID，如 'text01'

        Returns:
            文档的文本内容，找不到文档返回空字符串
        """
        # 检查缓存
        if doc_id in self._doc_cache:
            self._cache_hits += 1
            return self._doc_cache[doc_id]

        self._cache_misses += 1

        # 搜索文档路径（复用 config 的统一路径管理）
        pdf_path = self._find_document_path(doc_id)
        if not pdf_path:
            print(f"  ⚠️ [BatchProcessor] 未找到文档: {doc_id}")
            return ""

        # 读取文档内容
        try:
            text = self._read_document(pdf_path)
            if text:
                # 入缓存
                self._doc_cache[doc_id] = text
                print(f"  📄 [BatchProcessor] 加载文档: {doc_id} "
                      f"({len(text)} 字符) → 已缓存")
                return text
            else:
                print(f"  ⚠️ [BatchProcessor] 文档为空: {doc_id}")
                return ""
        except Exception as e:
            print(f"  ❌ [BatchProcessor] 读取文档失败 [{doc_id}]: {e}")
            return ""

    def _find_document_path(self, doc_id: str) -> str:
        """
        在全部 5 个领域目录中搜索文档路径。
        与 FinancialQAAgent._find_pdf_path 保持一致的搜索策略。
        """
        for domain in config.DOMAINS:
            domain_dir = config.get_raw_path(domain)

            pdf_path = os.path.join(domain_dir, f"{doc_id}.pdf")
            if os.path.exists(pdf_path):
                return pdf_path

            txt_path = os.path.join(domain_dir, f"{doc_id}.txt")
            if os.path.exists(txt_path):
                return txt_path

            txt_sub_path = os.path.join(domain_dir, "txt", f"{doc_id}.txt")
            if os.path.exists(txt_sub_path):
                return txt_sub_path

        fallback_path = os.path.join(config.raw_dir, doc_id, f"{doc_id}.pdf")
        if os.path.exists(fallback_path):
            return fallback_path

        return ""

    def _read_document(self, file_path: str) -> str:
        """读取 PDF 或 TXT 文件内容"""
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        try:
            import pdfplumber
            text = ''
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages[:config.MAX_PDF_PAGES]:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            return text
        except Exception as e:
            print(f"  PDF解析错误 [{file_path}]: {e}")
            return ""

    # ================================================================
    # Bug 6 修复: 预加载内容传入 Agent
    # ================================================================
    def batch_answer(self, questions_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理多道题目，核心优化:

          1. 按 doc_id 分组 → 相同的 doc_id 只读取一次
          2. 预加载所有涉及的文档 → 传入 Agent 的 preloaded_docs 参数
          3. Agent 检测到 preloaded_docs 后跳过 PDF 读取
          4. 单题异常隔离 → 一题失败不影响其他题

        Args:
            questions_batch: 题目数据列表

        Returns:
            结果列表 [{'qid', 'answer', 'prompt_tokens', 'completion_tokens', 'total_tokens'}, ...]
        """
        if not questions_batch:
            return []

        # ---- 步骤1: 收集所有涉及的 doc_ids ----
        all_doc_ids = set()
        for q in questions_batch:
            for doc_id in q.get('doc_ids', []):
                all_doc_ids.add(doc_id)

        print(f"\n{'='*60}")
        print(f"📦 批量处理: {len(questions_batch)} 道题目, "
              f"涉及 {len(all_doc_ids)} 个文档")
        print(f"{'='*60}")

        # ---- 步骤2: 预加载所有文档（Bug 6 核心修复） ----
        preloaded_docs = {}
        for doc_id in all_doc_ids:
            content = self.load_document(doc_id)
            if content:
                preloaded_docs[doc_id] = content

        print(f"📊 预加载完成: {len(preloaded_docs)}/{len(all_doc_ids)} 个文档")
        print(f"   缓存命中: {self._cache_hits}, 缓存未命中: {self._cache_misses}")

        # ---- 步骤3: 按文档分组显示 ----
        doc_groups = {}
        for q in questions_batch:
            for doc_id in q.get('doc_ids', []):
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(q['qid'])

        for doc_id, qids in doc_groups.items():
            cached = "📦缓存" if doc_id in self._doc_cache else "📄读取"
            print(f"  {cached} {doc_id}: 关联题目 {qids}")

        # ---- 步骤4: 逐题处理，传入预加载内容 ----
        results = []
        for i, q in enumerate(questions_batch, 1):
            qid = q.get('qid', f'unknown_{i}')
            try:
                # Bug 6 核心: 将预加载内容传入 Agent
                # Agent 在 answer_question 中检测到 preloaded_docs 后，
                # 会跳过 _find_pdf_path + _read_pdf，直接使用缓存内容
                result = self.agent.answer_question(q, preloaded_docs=preloaded_docs)
                results.append(result)
            except Exception as e:
                print(f"  ❌ 题目 {qid} 处理异常: {e}")
                # 单题失败不中断整批
                results.append({
                    'qid': qid,
                    'answer': '',
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'error': str(e)
                })

        # ---- 步骤5: 汇总统计 ----
        total_tokens = sum(r.get('total_tokens', 0) for r in results)
        valid_results = [r for r in results if r.get('answer')]
        print(f"\n{'='*60}")
        print(f"✅ 批处理完成: {len(valid_results)}/{len(results)} 成功")
        print(f"   Token 消耗: {total_tokens} (平均 {total_tokens // max(len(results), 1)}/题)")
        print(f"   缓存命中: {self._cache_hits}, 缓存未命中: {self._cache_misses}")
        print(f"{'='*60}\n")

        return results

    def clear_cache(self):
        """清空文档缓存（释放内存）"""
        self._doc_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        print("🧹 文档缓存已清空")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """返回缓存统计信息"""
        return {
            'cached_docs': len(self._doc_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'doc_ids': list(self._doc_cache.keys()),
        }


# ================================================================
# 测试
# ================================================================
if __name__ == '__main__':
    from financial_qa_agent import FinancialQAAgent
    import json

    print("=" * 60)
    print("测试批量处理器")
    print("=" * 60)

    # 加载题目
    question_path = config.get_question_path("group_a", "financial_contracts")
    if not os.path.exists(question_path):
        print(f"题目文件不存在: {question_path}")
        print("请确认数据已下载到正确位置")
        sys.exit(1)

    with open(question_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # 只取前 3 题测试
    test_questions = questions[:3]

    agent = FinancialQAAgent()
    processor = BatchProcessor(agent)

    results = processor.batch_answer(test_questions)

    print("\n结果汇总:")
    for r in results:
        print(f"  {r['qid']}: {r['answer']} (tokens: {r['total_tokens']})")
