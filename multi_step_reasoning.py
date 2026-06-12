# multi_step_reasoning.py
"""
多轮推理与自检验证模块

流程: 初步推理 → 证据回溯验证 → 一致性检查 → 最终答案

每轮调用Qwen API，后一轮以前一轮结论为输入，
通过"自检"机制减少幻觉、提高跨文档推理的准确性。
"""
from typing import Dict, Any, List, Tuple


class MultiStepReasoner:
    """
    三阶段多轮推理器。

    使用方式:
        reasoner = MultiStepReasoner(agent)
        result = reasoner.reason_with_verification(question, options, evidences)
    """

    def __init__(self, agent):
        """
        Args:
            agent: FinancialQAAgent 实例（需实现 call_qwen 方法）
        """
        self.agent = agent

    # ================================================================
    # ⑦ 多数投票
    # ================================================================
    def majority_vote(
        self, question: str, options: Dict[str, str],
        evidences: List[Dict], answer_format: str, rounds: int = 3
    ) -> Dict[str, Any]:
        """同一问题独立推理N次，取多数答案。"""
        from collections import Counter
        answers = []
        for i in range(rounds):
            result = self.reason_with_verification(question, options, evidences, answer_format)
            if result['answer']:
                answers.append(result['answer'])
        if not answers:
            return self._empty_result()
        # 取多数
        counter = Counter(answers)
        winner = counter.most_common(1)[0][0]
        confidence = counter[winner] / len(answers)
        return {
            'answer': winner,
            'reasoning': f'Majority vote: {dict(counter)}, winner={winner}',
            'verified': confidence >= 0.67,
            'confidence': confidence,
            'rounds': rounds,
        }

    # ================================================================
    # 三阶段主流程
    # ================================================================
    def reason_with_verification(
        self, question: str, options: Dict[str, str],
        evidences: List[Dict], answer_format: str = 'mcq'
    ) -> Dict[str, Any]:
        """
        多轮推理 + 逐轮自检。

        Returns:
            {'answer': str, 'reasoning': str, 'verified': bool,
             'confidence': float, 'rounds': int}
        """
        # ---- 第1轮: 初步推理 ----
        preliminary = self._initial_reasoning(question, options, evidences)
        if not preliminary:
            return self._empty_result()

        # ---- 第2轮: 证据回溯验证 ----
        verified = self._verify_with_evidence(
            preliminary, evidences, question, options
        )

        # ---- 第3轮: 一致性检查 ----
        consistency = self._check_consistency(
            verified, options, question, answer_format
        )

        # ---- 提取最终答案 ----
        final_answer = self._extract_final_answer(consistency, answer_format)
        confidence = self._estimate_confidence(verified, consistency)

        return {
            'answer': final_answer,
            'reasoning': consistency,
            'verified': '正确' in verified or '一致' in consistency,
            'confidence': confidence,
            'rounds': 3,
        }

    # ================================================================
    # 第1轮: 初步推理
    # ================================================================
    def _initial_reasoning(
        self, question: str, options: Dict[str, str], evidences: List[Dict]
    ) -> str:
        """分析问题，确定每个选项需要哪些证据类型，给出初步判断。"""
        options_text = self._fmt_options(options)
        evidence_text = self._fmt_evidences(evidences)

        prompt = (
            "你是一个金融分析推理专家。请根据提供的文档证据，对以下问题进行初步推理。\n\n"
            f"## 问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            f"## 文档证据\n{evidence_text}\n\n"
            "## 推理要求\n"
            "1. 对每个选项(A/B/C/D)逐项分析\n"
            "2. 每个选项标注: [需证据: 具体需要哪类证据]\n"
            "3. 每个选项标注: [初步判断: 正确/错误/不确定]\n"
            "4. 说明不确定的原因（证据不足/矛盾/缺失）\n\n"
            "仅输出推理过程，不输出最终答案。"
        )
        return self._safe_call(prompt, max_tokens=600)

    # ================================================================
    # 第2轮: 证据回溯验证 (P1-2 核心实现)
    # ================================================================
    def _verify_with_evidence(
        self, reasoning: str, evidences: List[Dict],
        question: str, options: Dict[str, str]
    ) -> str:
        """将第1轮的推理结论与原始证据逐条对照验证。

        核心逻辑:
          - 提取第1轮中标注为'正确'的选项
          - 从证据原文中查找支撑或反驳的句子
          - 标记'已核实'或'证据矛盾'或'证据不足'
        """
        evidence_text = self._fmt_evidences(evidences)
        options_text = self._fmt_options(options)

        prompt = (
            "你是一个金融事实核查员。现在对以下推理过程进行证据回溯验证。\n\n"
            f"## 原始问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            f"## 第1轮推理结论\n{reasoning}\n\n"
            f"## 原始文档证据\n{evidence_text}\n\n"
            "## 验证要求\n"
            "逐条核查第1轮推理中每个结论:\n"
            "1. 在原始文档中定位对应的证据句（用引号引用原文）\n"
            "2. 判断结论与证据是否一致: [已核实] / [证据矛盾] / [证据不足]\n"
            "3. 如[证据矛盾]，说明矛盾的具体内容\n"
            "4. 如[证据不足]，说明缺少哪类证据\n\n"
            "请输出: 逐选项验证结果 + 修正后的判断。"
        )
        return self._safe_call(prompt, max_tokens=600)

    # ================================================================
    # 第3轮: 一致性检查 (P1-2 核心实现)
    # ================================================================
    def _check_consistency(
        self, verified_reasoning: str, options: Dict[str, str],
        question: str, answer_format: str
    ) -> str:
        """检查修正后的答案与选项集合的逻辑一致性。

        检查点:
          - 单选题: 不能同时选两个答案
          - 多选题: 选中的选项之间不能互相矛盾
          - 判断题: 答案必须是A或B之一
          - 答案与选项文本的语义是否一致
        """
        options_text = self._fmt_options(options)
        format_hint = (
            "单选题，只能输出1个字母" if answer_format in ('mcq', 'tf')
            else "多选题，按字母顺序输出(如ABC)"
        )

        prompt = (
            "你是一个逻辑一致性审核员。检查以下验证结果是否存在内部矛盾。\n\n"
            f"## 原始问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            f"## 验证后的推理\n{verified_reasoning}\n\n"
            f"## 题型约束\n{format_hint}\n\n"
            "## 检查要点\n"
            "1. 选中的选项是否互相矛盾？(如A说增长、B选说下降，不能同时选A和B为正确)\n"
            "2. 答案数量是否符合题型约束？\n"
            "3. 是否有选项在验证阶段被遗漏？\n"
            "4. 综合判断后，修正最终答案\n\n"
            "请输出: 一致性检查结论 + '最终正确答案：X'"
        )
        return self._safe_call(prompt, max_tokens=400)

    # ================================================================
    # 辅助方法
    # ================================================================
    def _extract_final_answer(self, consistency_text: str, answer_format: str) -> str:
        """从一致性检查结果中提取最终答案"""
        import re
        if not consistency_text:
            return ''
        match = re.search(r'最终正确答案[：:\s]+([A-D]+)', consistency_text)
        if not match:
            match = re.search(r'正确答案[：:\s]+([A-D]+)', consistency_text)
        if match:
            answer = ''.join(c for c in match.group(1) if c in 'ABCD')
            if answer_format in ('mcq', 'tf'):
                return answer[0] if answer else 'A'
            return ''.join(sorted(set(answer))) if answer else ''
        # Fallback: 提取所有 A-D 字母
        letters = re.findall(r'[A-D]', consistency_text)
        if letters:
            if answer_format in ('mcq', 'tf'):
                return letters[-1]
            return ''.join(sorted(set(letters)))
        return ''

    def _estimate_confidence(self, verified: str, consistency: str) -> float:
        """估算答案置信度 (0.0~1.0)"""
        score = 0.5  # 起始 0.5
        # 验证阶段: 被核实越多，置信度越高
        verified_count = verified.count('[已核实]')
        contradiction_count = verified.count('[证据矛盾]')
        insufficient_count = verified.count('[证据不足]')
        if verified_count > 0:
            score += 0.2 * min(verified_count, 2)
        if contradiction_count > 0:
            score -= 0.2 * contradiction_count
        if insufficient_count > 0:
            score -= 0.1 * insufficient_count
        # 一致性检查: 通过加分
        if '一致' in consistency or '无矛盾' in consistency:
            score += 0.1
        if '修正' in consistency:
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _safe_call(self, prompt: str, max_tokens: int = 600) -> str:
        """安全调用Agent的Qwen API"""
        try:
            return self.agent._call_qwen(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"  [MultiStep] API call failed: {e}")
            return ""

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'answer': '', 'reasoning': '', 'verified': False,
            'confidence': 0.0, 'rounds': 0
        }

    # ================================================================
    # 格式化工具
    # ================================================================
    @staticmethod
    def _fmt_options(options: Dict[str, str]) -> str:
        parts = []
        for key in ['A', 'B', 'C', 'D']:
            if key in options:
                parts.append(f"{key}. {options[key]}")
        return '\n'.join(parts)

    @staticmethod
    def _fmt_evidences(evidences: List[Dict]) -> str:
        if not evidences:
            return "(无可用证据)"
        parts = []
        for i, ev in enumerate(evidences[:5], 1):
            doc_id = ev.get('doc_id', 'unknown')
            content = ev.get('content', '')[:600]
            parts.append(f"[证据{i} - {doc_id}]\n{content}\n")
        return '\n'.join(parts)


# ================================================================
# 测试
# ================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("MultiStepReasoner 结构测试")
    print("=" * 50)

    # Mock agent for testing
    class MockAgent:
        def _call_qwen(self, prompt, max_tokens=600):
            return "Mock response: 选项A正确, 选项B错误. 正确答案：A"

    mock_agent = MockAgent()
    reasoner = MultiStepReasoner(mock_agent)

    test_question = "发行人的主体信用评级是什么？"
    test_options = {'A': 'AAA', 'B': 'AA+', 'C': 'AA', 'D': 'A'}
    test_evidences = [
        {'doc_id': 'text01', 'content': '主体信用评级为AAA，评级展望稳定。', 'relevance': 10},
    ]

    result = reasoner.reason_with_verification(
        test_question, test_options, test_evidences, answer_format='mcq'
    )

    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']}")
    print(f"已验证: {result['verified']}")
    print(f"推理轮次: {result['rounds']}")
    print("\nP1-2: 多轮推理模块完成")
