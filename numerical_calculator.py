# numerical_calculator.py
"""
金融数值计算与中文数字解析

支持:
  - 中文数字→阿拉伯数字转换 (如"十亿三千万"→1030000000)
  - 金额单位转换 (亿/万/千/百 → 标准数值)
  - 比例/阈值/增长率计算
  - 选项数值验算: 从文档提取数值→验证选项中的计算结果
"""
import re
from typing import List, Tuple, Optional, Union


class FinancialCalculator:
    """金融数值计算器 — 支持中文数字解析和多步骤验算"""

    # ================================================================
    # 中文数字 → 阿拉伯数字 转换
    # ================================================================
    _CN_NUM = {
        '零': 0, '〇': 0,
        '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000,
        '万': 10000, '亿': 100000000,
    }

    # 大写数字
    _CN_NUM_UPPER = {
        '壹': 1, '贰': 2, '叁': 3, '肆': 4,
        '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9,
        '拾': 10, '佰': 100, '仟': 1000,
        '萬': 10000, '億': 100000000,
    }

    @classmethod
    def cn_to_arabic(cls, text: str) -> Optional[int]:
        """
        将中文数字字符串转换为阿拉伯数字。

        支持: "十亿"→1000000000, "三千五百万"→35000000,
              "百分之五"→5(返回数值部分), "三点一四"→3.14

        Args:
            text: 中文数字字符串

        Returns:
            阿拉伯数字，无法解析返回 None
        """
        if not text:
            return None

        text = text.strip()
        all_num_chars = {**cls._CN_NUM, **cls._CN_NUM_UPPER}

        # 1. 百分比: "百分之X" → 提取 X
        pct_match = re.match(r'百分之(.+)', text)
        if pct_match:
            val = cls.cn_to_arabic(pct_match.group(1))
            return val  # 返回数值，调用方自行处理百分比语义

        # 2. 含小数点的复杂表述: "三点一四"
        if '点' in text:
            parts = text.split('点', 1)
            integer_part = cls.cn_to_arabic(parts[0])
            decimal_str = ''.join(
                str(all_num_chars.get(c, '')) for c in parts[1]
                if c in all_num_chars
            )
            if integer_part is not None and decimal_str:
                return float(f"{integer_part}.{decimal_str}")

        # 3. 纯中文数字转换
        # 策略: 分段处理 (亿/万/千/百/十)
        result = 0
        section = 0        # 当前段累计值
        unit_stack = []     # 单位栈

        for char in text:
            if char not in all_num_chars:
                continue  # 跳过非数字字符

            val = all_num_chars[char]

            if val >= 10:  # 单位 (十/百/千/万/亿)
                if section == 0:
                    section = 1  # "十亿"中的"十"=10, 前面省略了"一"
                if val >= 10000:  # 万或亿级别
                    section = (section * val)
                    result += section
                    section = 0
                else:
                    section = section * val
            else:  # 数字 (一~九)
                section = val

        result += section
        return result if result > 0 else None

    # ================================================================
    # 金额解析
    # ================================================================
    @classmethod
    def parse_amount(cls, text: str) -> Optional[float]:
        """
        解析金额字符串 → 数值(以元为单位)。

        示例:
          "10亿元" → 1000000000.0
          "500万元" → 5000000.0
          "3.5万" → 35000.0
          "十亿" → 1000000000.0
        """
        if not text:
            return None

        text = text.strip()

        # 策略1: 阿拉伯数字 + 单位
        match = re.match(r'(\d+(?:\.\d+)?)\s*([万亿千百]?)\s*元?', text)
        if match:
            num = float(match.group(1))
            unit = match.group(2)
            multiplier = {'亿': 1e8, '万': 1e4, '千': 1e3, '百': 1e2}.get(unit, 1)
            return num * multiplier

        # 策略2: 纯中文数字
        cn_match = re.match(r'([零〇一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億]+)\s*元?', text)
        if cn_match:
            val = cls.cn_to_arabic(cn_match.group(1))
            if val is not None:
                return float(val)

        return None

    # ================================================================
    # 基本运算
    # ================================================================
    @staticmethod
    def calculate_ratio(value: float, total: float) -> float:
        """计算比例 (0~1)"""
        if total == 0:
            return 0.0
        return value / total

    @staticmethod
    def check_threshold(value: float, threshold: float, operator: str = '>=') -> bool:
        """阈值判断"""
        ops = {
            '>=': lambda v, t: v >= t,
            '<=': lambda v, t: v <= t,
            '>': lambda v, t: v > t,
            '<': lambda v, t: v < t,
            '=': lambda v, t: abs(v - t) < 1e-9,
            '!=': lambda v, t: abs(v - t) >= 1e-9,
        }
        return ops.get(operator, lambda v, t: False)(value, threshold)

    @staticmethod
    def growth_rate(old_val: float, new_val: float) -> float:
        """同比增长率"""
        if old_val == 0:
            return float('inf') if new_val > 0 else 0.0
        return (new_val - old_val) / old_val

    # ================================================================
    # 文本数值提取
    # ================================================================
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """从文本中提取所有数值"""
        # 阿拉伯数字
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        # 百分比
        percentages = re.findall(r'(\d+(?:\.\d+)?)%', text)
        result = [float(n) for n in numbers]
        result.extend(float(p) / 100 for p in percentages)
        return result

    # ================================================================
    # 验算: 选项数值 vs 文档证据数值
    # ================================================================
    @classmethod
    def verify_option_amount(
        cls, option_text: str, evidence_text: str, tolerance: float = 0.01
    ) -> Tuple[bool, str]:
        """
        验证选项中的金额是否与文档证据一致。

        Args:
            option_text: 选项文本 (如 "发行规模不超过50亿元")
            evidence_text: 文档证据文本
            tolerance: 相对容差 (0.01 = 1%)

        Returns:
            (是否一致, 说明信息)
        """
        opt_amounts = []
        ev_amounts = []

        # 提取选项中的金额
        for match in re.finditer(
            r'(\d+(?:\.\d+)?)\s*([亿万千百]?)\s*元?', option_text
        ):
            num = float(match.group(1))
            unit = match.group(2) or ''
            multiplier = {'亿': 1e8, '万': 1e4, '千': 1e3, '百': 1e2}.get(unit, 1)
            opt_amounts.append(num * multiplier)

        # 提取证据中的金额
        for match in re.finditer(
            r'(\d+(?:\.\d+)?)\s*([亿万千百]?)\s*元?', evidence_text
        ):
            num = float(match.group(1))
            unit = match.group(2) or ''
            multiplier = {'亿': 1e8, '万': 1e4, '千': 1e3, '百': 1e2}.get(unit, 1)
            ev_amounts.append(num * multiplier)

        if not opt_amounts:
            return False, "选项未含数值"
        if not ev_amounts:
            return False, "证据未含数值"

        # 检查选项金额是否在证据中能找到匹配
        for oa in opt_amounts:
            found = False
            for ea in ev_amounts:
                if abs(oa - ea) / max(abs(ea), 1) <= tolerance:
                    found = True
                    break
            if not found:
                return False, f"金额不一致: 选项={oa}, 证据中无匹配值"
            # 中文数字补充验证
            cn_values = []
            for match in re.finditer(
                r'([零〇一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億]+)\s*元?',
                evidence_text
            ):
                val = cls.cn_to_arabic(match.group(1))
                if val is not None:
                    cn_values.append(float(val))
            for cv in cn_values:
                if abs(oa - cv) / max(abs(cv), 1) <= tolerance:
                    found = True
                    break

        return True, "金额验证通过"


# ================================================================
# 测试
# ================================================================
if __name__ == '__main__':
    calc = FinancialCalculator()

    # 中文数字测试
    tests = [
        ("十亿", 1000000000),
        ("三千五百万", 35000000),
        ("一百二十三", 123),
        ("三点一四", 3.14),
        ("百分之五", 5),
    ]
    print("中文数字解析:")
    for text, expected in tests:
        result = calc.cn_to_arabic(text)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {status}: '{text}' -> {result}")

    # 金额解析测试
    print("\n金额解析:")
    for text in ["10亿元", "500万元", "3.5万"]:
        result = calc.parse_amount(text)
        print(f"  '{text}' -> {result:,.0f}")

    # 验算测试
    print("\n验算测试:")
    ok, msg = calc.verify_option_amount(
        "发行规模不超过50亿元",
        "本次债券发行规模为人民币50亿元"
    )
    print(f"  50亿 vs 50亿: {ok} ({msg})")

    ok, msg = calc.verify_option_amount(
        "发行规模不超过30亿元",
        "本次债券发行规模为人民币50亿元"
    )
    print(f"  30亿 vs 50亿: {ok} ({msg})")

    print("\nP1-3: 数值计算模块完成")
