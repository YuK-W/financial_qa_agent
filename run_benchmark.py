# run_benchmark.py
"""
A榜全量测试脚本 — 5领域100题 + 生成 answer.csv

用法:
    python run_benchmark.py

输出:
    - answer.csv  (符合大赛格式: qid, answer, prompt_tokens, completion_tokens, total_tokens + summary行)
    - 终端输出:   每领域准确率、题型准确率、总Token消耗

依赖:
    financial_qa_agent.py, retrieval_system.py, config.py, batch_processor.py
"""
import os
import sys
import json
import csv
import time
from typing import Dict, Any, List
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from logger import log                                        # P2-1
from exceptions import FinancialQABaseError                   # P2-4
from financial_qa_agent import FinancialQAAgent
from batch_processor import BatchProcessor

# 尝试导入 tqdm，没有则用简单循环
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        for i, item in enumerate(iterable):
            print(f"\r[{desc}] {i+1}/{len(iterable)}", end="", flush=True)
            yield item
        print()


class ABenchmark:
    """A榜全量测试"""

    def __init__(self, agent: FinancialQAAgent):
        self.agent = agent
        self.batch_processor = BatchProcessor(agent)
        self.results: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def load_all_questions(self) -> List[Dict]:
        """加载全部 5 个领域 100 道题目"""
        all_questions = []
        for domain in config.DOMAINS:
            path = config.get_question_path("group_a", domain)
            if not os.path.exists(path):
                log.warning(f"question file not found: {path}")
                continue
            with open(path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            # 确保每道题有 domain 字段
            for q in questions:
                if 'domain' not in q:
                    q['domain'] = domain
            all_questions.extend(questions)
            print(f"  {config.DOMAIN_NAMES.get(domain, domain)}: {len(questions)} 题")

        return all_questions

    def run(self, dry_run: bool = False):
        """
        运行全量测试

        Args:
            dry_run: True=只跑前2题验证流程，False=全量100题
        """
        print("=" * 60)
        print("A榜全量测试")
        print("=" * 60)

        # 加载题目
        print("\n加载题目:")
        all_questions = self.load_all_questions()
        print(f"\n总计: {len(all_questions)} 题")

        if dry_run:
            all_questions = all_questions[:2]
            print("[DRY RUN] Only first 2 questions")

        # 逐领域处理（利用批处理共享文档索引）
        domain_questions = defaultdict(list)
        for q in all_questions:
            domain_questions[q.get('domain', 'financial_contracts')].append(q)

        for domain, questions in domain_questions.items():
            domain_name = config.DOMAIN_NAMES.get(domain, domain)
            print(f"\n{'='*60}")
            print(f"领域: {domain_name} ({len(questions)} 题)")
            print(f"{'='*60}")

            # 预加载该领域所有涉及的文档
            doc_ids = set()
            for q in questions:
                doc_ids.update(q.get('doc_ids', []))

            # 领域内逐题处理
            for q in tqdm(questions, desc=f"  {domain_name}"):
                try:
                    result = self.agent.answer_question(q)
                    result['domain'] = domain
                    result['expected'] = q.get('answer', '')
                    self.results.append(result)

                    self.total_prompt_tokens += result['prompt_tokens']
                    self.total_completion_tokens += result['completion_tokens']

                except Exception as e:
                    log.error(f"{q.get('qid', '?')} failed: {e}")
                    self.results.append({
                        'qid': q.get('qid', '?'),
                        'answer': '',
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'domain': domain,
                        'expected': q.get('answer', ''),
                        'error': str(e),
                    })

        # 生成结果
        self._print_stats()
        self._generate_csv()

    def _print_stats(self):
        """输出准确率和 Token 统计"""
        total = len(self.results)
        correct = sum(1 for r in self.results if r['answer'] == r.get('expected', ''))
        accuracy = correct / total * 100 if total > 0 else 0

        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        avg_tokens = total_tokens / total if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}")
        print(f"  总题数:     {total}")
        print(f"  正确数:     {correct}")
        print(f"  准确率:     {accuracy:.1f}%")
        print(f"  总Token:    {total_tokens:,}  (预算: 5,000,000)")
        print(f"  平均Token:  {avg_tokens:,.0f} / 题")

        token_budget = 5_000_000
        token_score = max(0, min(1, (token_budget - total_tokens) / token_budget))
        final_score = 100 * (accuracy / 100) * (0.7 + 0.3 * token_score)
        print(f"  TokenScore: {token_score:.4f}")
        print(f"  FinalScore: {final_score:.2f}")

        # 按领域统计
        print(f"\n  领域准确率:")
        domain_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'tokens': 0})
        for r in self.results:
            d = r.get('domain', 'unknown')
            domain_stats[d]['total'] += 1
            if r['answer'] == r.get('expected', ''):
                domain_stats[d]['correct'] += 1
            domain_stats[d]['tokens'] += r['total_tokens']

        for domain in config.DOMAINS:
            if domain in domain_stats:
                s = domain_stats[domain]
                acc = s['correct'] / s['total'] * 100
                name = config.DOMAIN_NAMES.get(domain, domain)
                print(f"    {name}: {s['correct']}/{s['total']} ({acc:.1f}%) | Token: {s['tokens']:,}")

        # 按题型统计
        print(f"\n  题型准确率:")
        type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in self.results:
            # answer_format 从题目原始数据中获取; 这里用 expected 的长度推断
            expected = r.get('expected', '')
            if len(expected) == 1:
                qtype = '单选/判断'
            else:
                qtype = '多选'
            type_stats[qtype]['total'] += 1
            if r['answer'] == expected:
                type_stats[qtype]['correct'] += 1

        for qtype, s in type_stats.items():
            acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
            print(f"    {qtype}: {s['correct']}/{s['total']} ({acc:.1f}%)")

        # Token 效率
        print(f"\n  Token 效率:")
        print(f"    总输入 Token: {self.total_prompt_tokens:,}")
        print(f"    总输出 Token: {self.total_completion_tokens:,}")
        print(f"    输入/输出比:  {self.total_prompt_tokens / max(self.total_completion_tokens, 1):.1f}:1")

    def _generate_csv(self):
        """生成 answer.csv（符合大赛格式）"""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        csv_path = os.path.join(config.project_root, "answer.csv")

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'qid', 'answer', 'prompt_tokens', 'completion_tokens', 'total_tokens'
            ])
            writer.writeheader()

            # summary 行（大赛要求第一行）
            writer.writerow({
                'qid': 'summary',
                'answer': '',
                'prompt_tokens': self.total_prompt_tokens,
                'completion_tokens': self.total_completion_tokens,
                'total_tokens': total_tokens,
            })

            # 每题结果
            for r in self.results:
                writer.writerow({
                    'qid': r['qid'],
                    'answer': r['answer'],
                    'prompt_tokens': r['prompt_tokens'],
                    'completion_tokens': r['completion_tokens'],
                    'total_tokens': r['total_tokens'],
                })

        print(f"\nanswer.csv generated: {csv_path}")
        print(f"   行数: {len(self.results) + 1} (含 summary)")

        # 输出前几行供验证
        print(f"   前5行预览:")
        with open(csv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 6:
                    break
                print(f"     {line.rstrip()}")


# ================================================================
# 入口
# ================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="A榜全量测试")
    parser.add_argument("--dry-run", action="store_true",
                        help="只跑前2题验证流程")
    args = parser.parse_args()

    start = time.time()

    agent = FinancialQAAgent()
    benchmark = ABenchmark(agent)
    benchmark.run(dry_run=args.dry_run)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
