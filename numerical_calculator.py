# numerical_calculator.py
class FinancialCalculator:
    @staticmethod
    def calculate_ratio(value, total):
        """计算比例"""
        if total == 0:
            return 0
        return value / total
    
    @staticmethod
    def check_threshold(value, threshold, operator='>='):
        """阈值判断"""
        if operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
    
    @staticmethod
    def extract_numbers_from_text(text):
        """从文本中提取数字"""
        import re
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        return [float(n) for n in numbers if n]