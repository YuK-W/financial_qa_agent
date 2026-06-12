# experience.py
"""
Agent 经验学习系统 — 让系统越用越聪明

三大机制:
  1. 自我纠错: 低置信度 → 换策略重答
  2. 领域经验库: 记录最佳Prompt策略 → 下次优先使用
  3. Bad Case 积累: 记录失败模式和规避策略
"""
import os
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ExperienceManager:
    """经验学习管理器"""

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "experience.json"
            )
        self.storage_path = storage_path
        self.memory = self._load()

    # ================================================================
    # 1. 自我纠错 — 低置信度换策略
    # ================================================================
    RETRY_STRATEGIES = [
        'default',           # 默认: 领域Prompt + 单轮
        'multi_step',        # 多轮推理
        'strict_evidence',   # 严格证据: 要求逐句引用原文
        'reverse_check',     # 反向验证: 先假设选项错误，找反证
        'majority_vote',     # 多数投票: 同一问题问3次取多数
    ]

    def should_retry(self, qid: str, confidence: float, answer: str) -> Tuple[bool, str]:
        """
        判断是否需要换策略重试。

        Returns:
            (是否需要重试, 推荐的下一个策略)
        """
        if confidence >= 0.6:
            return False, ''

        # 记录低置信度
        used_strategies = self.memory['low_confidence'].get(qid, [])
        available = [s for s in self.RETRY_STRATEGIES if s not in used_strategies]

        if not available:
            return False, ''

        next_strategy = available[0]
        return True, next_strategy

    def record_attempt(self, qid: str, strategy: str, confidence: float, answer: str):
        """记录一次尝试"""
        if qid not in self.memory['low_confidence']:
            self.memory['low_confidence'][qid] = []
        self.memory['low_confidence'][qid].append(strategy)

    # ================================================================
    # 2. 领域经验库 — 记录最优策略
    # ================================================================
    def record_success(self, domain: str, answer_format: str,
                        strategy: str, features: Dict[str, Any]):
        """
        记录成功的答题模式。

        features 示例:
          {'has_table': True, 'doc_count': 3, 'question_type': 'comparison'}
        """
        key = f"{domain}_{answer_format}"
        if key not in self.memory['patterns']:
            self.memory['patterns'][key] = []

        self.memory['patterns'][key].append({
            'strategy': strategy,
            'features': features,
            'time': time.time(),
        })

        # 只保留最近 50 条
        if len(self.memory['patterns'][key]) > 50:
            self.memory['patterns'][key] = self.memory['patterns'][key][-50:]

    def get_best_strategy(self, domain: str, answer_format: str,
                           features: Dict[str, Any]) -> str:
        """
        查询该领域+题型下最成功的策略。
        """
        key = f"{domain}_{answer_format}"
        patterns = self.memory['patterns'].get(key, [])
        if not patterns:
            return 'default'

        # 匹配相似特征
        best_score = 0
        best_strategy = 'default'
        for p in patterns:
            score = self._feature_similarity(p['features'], features)
            if score > best_score:
                best_score = score
                best_strategy = p['strategy']
        return best_strategy

    def _feature_similarity(self, f1: Dict, f2: Dict) -> float:
        """计算特征相似度 (0~1)"""
        if not f1 or not f2:
            return 0.0
        matches = 0
        for key in set(f1.keys()) | set(f2.keys()):
            if f1.get(key) == f2.get(key):
                matches += 1
        return matches / max(len(f1), len(f2), 1)

    # ================================================================
    # 3. Bad Case 积累 — 记录失败模式
    # ================================================================
    def record_bad_case(self, qid: str, domain: str, answer_format: str,
                         question: str, answer: str, fail_reason: str,
                         features: Dict[str, Any]):
        """记录失败案例和推测原因"""
        self.memory['bad_cases'].append({
            'qid': qid,
            'domain': domain,
            'answer_format': answer_format,
            'question': question[:100],
            'answer': answer,
            'fail_reason': fail_reason,
            'features': features,
            'time': time.time(),
        })

        # 只保留最近 200 条
        if len(self.memory['bad_cases']) > 200:
            self.memory['bad_cases'] = self.memory['bad_cases'][-200:]

    def get_avoidance_hints(self, domain: str, answer_format: str) -> List[str]:
        """
        获取该领域+题型的避坑提示（给Prompt用）。
        """
        relevant = [
            bc for bc in self.memory['bad_cases']
            if bc['domain'] == domain and bc['answer_format'] == answer_format
        ]
        if not relevant:
            return []

        # 统计最常见的失败原因
        reason_counts = defaultdict(int)
        for bc in relevant[-20:]:
            reason_counts[bc['fail_reason']] += 1

        hints = []
        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for reason, count in top_reasons:
            if reason == 's4_extraction_fail':
                hints.append("务必以'正确答案：X'格式输出最终答案")
            elif reason == 'no_evidence':
                hints.append("若文档中找不到证据，明确声明'无法确定'而非猜测")
            elif reason == 'numeric_mismatch':
                hints.append("注意区分亿元/万元等单位，统一单位后再比较")
            elif reason == 'multi_select_miss':
                hints.append("多选题务必逐个选项独立判断，不要默认只有一个正确")
        return hints

    # ================================================================
    # 持久化
    # ================================================================
    def _load(self) -> Dict:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'low_confidence': {},     # qid → [strategy1, strategy2, ...]
            'patterns': {},           # domain_format → [{strategy, features}]
            'bad_cases': [],          # [{qid, domain, fail_reason, ...}]
            'stats': {                # 全局统计
                'total_questions': 0,
                'retry_count': 0,
                'retry_success_rate': 0.0,
            },
        }

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Experience] save failed: {e}")

    @property
    def stats(self) -> Dict[str, Any]:
        return self.memory['stats']


# ================================================================
# 测试
# ================================================================
if __name__ == '__main__':
    exp = ExperienceManager("test_experience.json")

    # 记录成功
    exp.record_success('financial_contracts', 'multi',
                       'multi_step', {'has_table': True, 'doc_count': 2})

    # 查询最佳策略
    best = exp.get_best_strategy('financial_contracts', 'multi',
                                  {'has_table': True, 'doc_count': 2})
    print(f"1. best strategy: {best}")

    # 记录 bad case
    exp.record_bad_case('fc_a_001', 'financial_contracts', 'multi',
                        '测试问题', 'ABD', 's4_extraction_fail',
                        {'doc_count': 2})

    # 获取避坑提示
    hints = exp.get_avoidance_hints('financial_contracts', 'multi')
    print(f"2. avoidance hints: {hints}")

    # 自我纠错
    should, strategy = exp.should_retry('fc_a_001', 0.3, 'A')
    print(f"3. retry: {should}, strategy: {strategy}")

    print("Experience system OK")
    # Clean up test file
    os.remove("test_experience.json")
