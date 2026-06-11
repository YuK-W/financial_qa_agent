# error_analysis.py
"""
错误分析工具 — P2-3

输入: answer.csv + questions JSON
输出: 按领域/题型/Token效率的多维度准确率分析

用法:
    python error_analysis.py answer.csv
"""
import os
import sys
import json
import csv
from typing import Dict, Any, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from logger import log


class ErrorAnalyzer:
    """多维度错误分析器"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.results: List[Dict] = []
        self.ground_truth: Dict[str, str] = {}    # qid → expected answer
        self.question_meta: Dict[str, Dict] = {}   # qid → {domain, answer_format, ...}
        self._load_csv()
        self._load_ground_truth()

    # ================================================================
    # 数据加载
    # ================================================================
    def _load_csv(self):
        """加载 answer.csv（跳过 summary 行）"""
        if not os.path.exists(self.csv_path):
            log.error(f"文件不存在: {self.csv_path}")
            return

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['qid'] == 'summary':
                    self.summary_tokens = {
                        'prompt_tokens': int(row['prompt_tokens']),
                        'completion_tokens': int(row['completion_tokens']),
                        'total_tokens': int(row['total_tokens']),
                    }
                    continue
                self.results.append({
                    'qid': row['qid'],
                    'answer': row['answer'],
                    'prompt_tokens': int(row['prompt_tokens']),
                    'completion_tokens': int(row['completion_tokens']),
                    'total_tokens': int(row['total_tokens']),
                })

        log.info(f"加载 {len(self.results)} 条结果")

    def _load_ground_truth(self):
        """从题目文件中加载正确答案和元数据"""
        for domain in config.DOMAINS:
            path = config.get_question_path("group_a", domain)
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            for q in questions:
                qid = q.get('qid', '')
                self.ground_truth[qid] = q.get('answer', '')
                self.question_meta[qid] = {
                    'domain': q.get('domain', domain),
                    'answer_format': q.get('answer_format', 'mcq'),
                    'question': q.get('question', '')[:80],
                    'doc_ids': q.get('doc_ids', []),
                }

    # ================================================================
    # 分析维度
    # ================================================================
    def analyze(self) -> Dict[str, Any]:
        """全维度分析"""
        return {
            'overall': self._analyze_overall(),
            'by_domain': self._analyze_by_domain(),
            'by_type': self._analyze_by_type(),
            'token_efficiency': self._analyze_token_efficiency(),
            'error_details': self._analyze_errors(),
            'worst_questions': self._top_errors(10),
        }

    def _analyze_overall(self) -> Dict:
        total = len(self.results)
        correct = sum(1 for r in self.results
                      if r['answer'] == self.ground_truth.get(r['qid'], ''))
        accuracy = correct / total * 100 if total > 0 else 0

        total_tokens = sum(r['total_tokens'] for r in self.results)
        avg_tokens = total_tokens / total if total > 0 else 0

        token_score = max(0, min(1, (5_000_000 - total_tokens) / 5_000_000))
        final_score = 100 * (accuracy / 100) * (0.7 + 0.3 * token_score)

        return {
            'total': total, 'correct': correct, 'accuracy': f"{accuracy:.1f}%",
            'total_tokens': total_tokens, 'avg_tokens': f"{avg_tokens:,.0f}",
            'token_score': f"{token_score:.4f}",
            'final_score': f"{final_score:.2f}",
        }

    def _analyze_by_domain(self) -> Dict[str, Dict]:
        """按5领域统计准确率"""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'tokens': 0})
        for r in self.results:
            meta = self.question_meta.get(r['qid'], {})
            domain = meta.get('domain', 'unknown')
            stats[domain]['total'] += 1
            if r['answer'] == self.ground_truth.get(r['qid'], ''):
                stats[domain]['correct'] += 1
            stats[domain]['tokens'] += r['total_tokens']

        result = {}
        for domain in config.DOMAINS:
            if domain in stats:
                s = stats[domain]
                name = config.DOMAIN_NAMES.get(domain, domain)
                result[name] = {
                    'total': s['total'],
                    'correct': s['correct'],
                    'wrong': s['total'] - s['correct'],
                    'accuracy': f"{s['correct']/s['total']*100:.1f}%",
                    'total_tokens': s['tokens'],
                    'avg_tokens': f"{s['tokens']/s['total']:,.0f}",
                }
        return result

    def _analyze_by_type(self) -> Dict[str, Dict]:
        """按题型统计"""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in self.results:
            meta = self.question_meta.get(r['qid'], {})
            fmt = meta.get('answer_format', 'mcq')
            type_name = {'mcq': '单选题', 'tf': '判断题', 'multi': '多选题'}.get(fmt, fmt)
            stats[type_name]['total'] += 1
            if r['answer'] == self.ground_truth.get(r['qid'], ''):
                stats[type_name]['correct'] += 1

        return {
            name: {
                **s,
                'accuracy': f"{s['correct']/s['total']*100:.1f}%",
            }
            for name, s in stats.items()
        }

    def _analyze_token_efficiency(self) -> Dict:
        """Token效率分析"""
        inefficient = [r for r in self.results if r['total_tokens'] > 5000]
        efficient = [r for r in self.results if r['total_tokens'] <= 2000]

        per_domain_tokens = defaultdict(list)
        for r in self.results:
            meta = self.question_meta.get(r['qid'], {})
            domain = meta.get('domain', 'unknown')
            per_domain_tokens[domain].append(r['total_tokens'])

        domain_avg = {}
        for domain, tokens in per_domain_tokens.items():
            name = config.DOMAIN_NAMES.get(domain, domain)
            domain_avg[name] = f"{sum(tokens)/len(tokens):,.0f}"

        return {
            'inefficient_count': len(inefficient),
            'efficient_count': len(efficient),
            'avg_per_domain': domain_avg,
            'max_single': max(r['total_tokens'] for r in self.results),
            'min_single': min(r['total_tokens'] for r in self.results),
        }

    def _analyze_errors(self) -> List[Dict]:
        """输出所有错误题目的详细信息"""
        errors = []
        for r in self.results:
            expected = self.ground_truth.get(r['qid'], '')
            if r['answer'] != expected:
                meta = self.question_meta.get(r['qid'], {})
                errors.append({
                    'qid': r['qid'],
                    'domain': config.DOMAIN_NAMES.get(meta.get('domain', ''), ''),
                    'type': meta.get('answer_format', ''),
                    'expected': expected,
                    'actual': r['answer'],
                    'tokens': r['total_tokens'],
                    'doc_ids': ', '.join(meta.get('doc_ids', [])),
                    'question': meta.get('question', ''),
                })
        return errors

    def _top_errors(self, n: int = 10) -> List[Dict]:
        """Token消耗最高的N道错题"""
        errors = self._analyze_errors()
        return sorted(errors, key=lambda x: x['tokens'], reverse=True)[:n]

    # ================================================================
    # 输出报告
    # ================================================================
    def print_report(self):
        """打印完整分析报告"""
        analysis = self.analyze()

        print("=" * 60)
        print("Error Analysis Report")
        print("=" * 60)

        # 总体
        o = analysis['overall']
        print(f"\n--- Overall ---")
        print(f"  Questions:    {o['total']}")
        print(f"  Correct:      {o['correct']}")
        print(f"  Accuracy:     {o['accuracy']}")
        print(f"  TokenScore:   {o['token_score']}")
        print(f"  FinalScore:   {o['final_score']}")
        print(f"  Total Tokens: {o['total_tokens']:,}")
        print(f"  Avg Tokens:   {o['avg_tokens']}")

        # 按领域
        print(f"\n--- By Domain ---")
        for name, s in analysis['by_domain'].items():
            print(f"  {name}: {s['correct']}/{s['total']} ({s['accuracy']}) | avg: {s['avg_tokens']} tokens")

        # 按题型
        print(f"\n--- By Type ---")
        for name, s in analysis['by_type'].items():
            print(f"  {name}: {s['correct']}/{s['total']} ({s['accuracy']})")

        # Token 效率
        t = analysis['token_efficiency']
        print(f"\n--- Token Efficiency ---")
        print(f"  >5000 tokens:     {t['inefficient_count']} questions")
        print(f"  <=2000 tokens:     {t['efficient_count']} questions")
        print(f"  Max single:        {t['max_single']:,}")
        print(f"  Min single:        {t['min_single']:,}")
        print(f"  Avg per domain:    {t['avg_per_domain']}")

        # 错误详情 (前5)
        print(f"\n--- Top Errors (by token waste) ---")
        for e in analysis['worst_questions'][:5]:
            print(f"  {e['qid']} [{e['domain']}] expected={e['expected']} got={e['actual']} "
                  f"({e['tokens']:,} tokens)")
            print(f"    Q: {e['question'][:100]}")
            print(f"    Docs: {e['doc_ids']}")

    def save_report(self, output_path: str = None):
        """保存 JSON 分析报告"""
        if output_path is None:
            output_path = os.path.join(config.project_root, "error_report.json")
        analysis = self.analyze()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {output_path}")


# ================================================================
# 入口
# ================================================================
if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "answer.csv"
    analyzer = ErrorAnalyzer(csv_path)
    analyzer.print_report()
    analyzer.save_report()
