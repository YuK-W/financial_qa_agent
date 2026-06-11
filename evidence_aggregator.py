# evidence_aggregator.py
"""
证据聚合模块 — 多文档证据联合判断 + 跨文档矛盾检测

核心能力:
  1. 聚合: 同一问题下多文档、多段落的证据合并去重
  2. 矛盾检测: 发现不同文档之间的冲突陈述
  3. 加权排序: 按相关度+来源可信度给证据赋权
"""
import re
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict


class EvidenceAggregator:
    """多文档证据聚合器"""

    def __init__(self):
        self.contradictions: List[Dict] = []  # 检测到的矛盾记录

    # ================================================================
    # 主流程: 聚合 + 矛盾检测
    # ================================================================
    def aggregate(
        self, evidences: List[Dict], question: str, options: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        对多文档证据进行聚合分析。

        Args:
            evidences: [{'doc_id', 'content', 'relevance'}, ...]
            question: 问题文本
            options: 选项字典

        Returns:
            {
                'aggregated': List[Dict],      # 去重加权后的证据列表
                'contradictions': List[Dict],   # 检测到的矛盾
                'coverage': Dict[str, float],   # 每个选项的证据覆盖率
                'key_facts': List[str],         # 提取的关键事实
            }
        """
        if not evidences:
            return self._empty_result()

        # 1. 提取每个证据的关键主张
        claims = self._extract_claims(evidences)

        # 2. 跨文档矛盾检测
        contradictions = self._detect_contradictions(claims, evidences)

        # 3. 按相关度加权排序
        aggregated = self._weight_and_rank(evidences, question, options)

        # 4. 选项覆盖率
        coverage = self._calc_coverage(aggregated, options)

        # 5. 提取关键事实
        key_facts = self._extract_key_facts(claims)

        return {
            'aggregated': aggregated,
            'contradictions': contradictions,
            'coverage': coverage,
            'key_facts': key_facts,
        }

    # ================================================================
    # 主张提取
    # ================================================================
    def _extract_claims(self, evidences: List[Dict]) -> List[Dict]:
        """
        从每个证据片段提取关键主张。

        金融文档中常见的数值主张模式:
          - "发行人/公司名称是 X"
          - "发行规模/金额为 X 亿元"
          - "信用评级为 AAA/AA+/..."
          - "票面利率为 X%"
        """
        claims = []
        claim_patterns = [
            # (类别, 正则)
            ('主体名称', r'(发行人|公司名称|发行主体)[是为：:]\s*([^\n，。]+)'),
            ('发行规模', r'(发行规模|发行金额|募集资金)[^为]*[是为：:]\s*([^\n，。]+)'),
            ('信用评级', r'(信用评级|主体评级|债项评级)[是为：:]\s*([^\n，。]+)'),
            ('票面利率', r'(票面利率|发行利率)[是为：:]\s*([^\n，。]+)'),
            ('债券期限', r'(债券期限|期限)[是为：:]\s*([^\n，。]+)'),
            ('主承销商', r'(主承销商|牵头主承销商)[是为：:]\s*([^\n，。]+)'),
            ('受托管理人', r'(受托管理人|债券受托管理人)[是为：:]\s*([^\n，。]+)'),
        ]

        for ev in evidences:
            content = ev.get('content', '')
            doc_id = ev.get('doc_id', '')
            for category, pattern in claim_patterns:
                match = re.search(pattern, content)
                if match:
                    value = match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip()
                    if len(value) < 100:  # 过滤过长匹配
                        claims.append({
                            'doc_id': doc_id,
                            'category': category,
                            'value': value,
                            'full_match': match.group(),
                        })
        return claims

    # ================================================================
    # 矛盾检测 (P1-4 核心)
    # ================================================================
    def _detect_contradictions(
        self, claims: List[Dict], evidences: List[Dict]
    ) -> List[Dict]:
        """
        检测跨文档矛盾陈述。

        策略: 同一类别(category)、来自不同文档(doc_id)的主张进行比较。
        若值不同 → 标记为潜在矛盾。
        """
        contradictions = []
        # 按类别分组
        by_category = defaultdict(list)
        for claim in claims:
            by_category[claim['category']].append(claim)

        for category, cat_claims in by_category.items():
            # 同一类别下来自不同文档的主张
            doc_values = defaultdict(set)
            for c in cat_claims:
                doc_values[c['doc_id']].add(c['value'])

            if len(doc_values) < 2:
                continue  # 只有一个文档有主张，无矛盾

            # 比较不同文档的值
            doc_ids = list(doc_values.keys())
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    vals_i = doc_values[doc_ids[i]]
                    vals_j = doc_values[doc_ids[j]]
                    # 集合不同 → 可能矛盾
                    if vals_i != vals_j:
                        contradictions.append({
                            'category': category,
                            'doc_a': doc_ids[i],
                            'value_a': list(vals_i),
                            'doc_b': doc_ids[j],
                            'value_b': list(vals_j),
                            'severity': 'high' if not (vals_i & vals_j) else 'medium',
                        })

        self.contradictions = contradictions
        return contradictions

    # ================================================================
    # 加权排序
    # ================================================================
    def _weight_and_rank(
        self, evidences: List[Dict], question: str, options: Dict[str, str]
    ) -> List[Dict]:
        """按综合相关度排序，去重相似内容"""
        # 提取问题+选项关键词
        q_words = set(re.findall(r'[一-龥a-zA-Z0-9]{2,}', question))
        for opt_text in options.values():
            q_words.update(re.findall(r'[一-龥a-zA-Z0-9]{2,}', opt_text))

        for ev in evidences:
            content = ev.get('content', '')
            relevance = ev.get('relevance', 0)

            # 关键词命中加分
            content_words = set(re.findall(r'[一-龥a-zA-Z0-9]{2,}', content))
            keyword_hits = len(q_words & content_words)
            relevance += keyword_hits * 0.5

            # 实体命中加分
            amounts = len(re.findall(r'\d+(?:\.\d+)?\s*[亿万千百]?\s*元?', content))
            relevance += amounts * 0.3

            ev['relevance'] = relevance

        # 排序 + 去重
        seen = set()
        ranked = []
        for ev in sorted(evidences, key=lambda x: x.get('relevance', 0), reverse=True):
            content_key = ev.get('content', '')[:100]  # 前100字符去重
            if content_key not in seen:
                seen.add(content_key)
                ranked.append(ev)

        return ranked[:10]  # 最多10条

    # ================================================================
    # 覆盖率计算
    # ================================================================
    def _calc_coverage(
        self, evidences: List[Dict], options: Dict[str, str]
    ) -> Dict[str, float]:
        """计算每个选项被证据覆盖的程度 (0.0~1.0)"""
        coverage = {}
        all_text = ' '.join(ev.get('content', '') for ev in evidences)

        for key, opt_text in options.items():
            opt_words = set(re.findall(r'[一-龥a-zA-Z0-9]{2,}', opt_text))
            if not opt_words:
                coverage[key] = 0.0
                continue
            hits = sum(1 for w in opt_words if w in all_text)
            coverage[key] = hits / len(opt_words)

        return coverage

    # ================================================================
    # 关键事实提取
    # ================================================================
    def _extract_key_facts(self, claims: List[Dict]) -> List[str]:
        """从所有主张中提取不重复的关键事实"""
        seen = set()
        facts = []
        for claim in claims:
            fact = f"[{claim['category']}] {claim['value']}"
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
        return facts

    def _empty_result(self) -> Dict:
        return {
            'aggregated': [], 'contradictions': [],
            'coverage': {}, 'key_facts': []
        }


# ================================================================
# 测试
# ================================================================
if __name__ == '__main__':
    aggregator = EvidenceAggregator()

    # 模拟多文档证据
    test_evidences = [
        {'doc_id': 'text01', 'content': '发行人: 广东省广晟控股集团有限公司。发行规模: 不超过50亿元。主体信用评级: AAA。', 'relevance': 10},
        {'doc_id': 'text02', 'content': '发行人: 深圳租赁有限公司。发行规模: 30亿元。主体信用评级: AA+。', 'relevance': 8},
        {'doc_id': 'text01', 'content': '主承销商: 中信证券。票面利率: 2.95%。', 'relevance': 5},
    ]
    test_question = "发行人的主体信用评级是什么？"
    test_options = {'A': 'AAA', 'B': 'AA+', 'C': 'AA', 'D': 'A'}

    result = aggregator.aggregate(test_evidences, test_question, test_options)

    print(f"聚合后证据: {len(result['aggregated'])} 条")
    print(f"矛盾数: {len(result['contradictions'])}")
    for c in result['contradictions']:
        print(f"  矛盾: [{c['category']}] {c['doc_a']}={c['value_a']} vs {c['doc_b']}={c['value_b']} ({c['severity']})")
    print(f"覆盖率: {result['coverage']}")
    print(f"关键事实: {result['key_facts']}")
    print("\nP1-4: 证据聚合模块完成")
