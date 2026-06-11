# domain_parsers.py
import re
from typing import Dict, Any, List, Optional

class ContractParser:
    """金融合同专用解析器"""
    
    def __init__(self):
        # 合同关键信息匹配模式
        self.patterns = {
            # 发行主体/公司名称
            'issuer': r'(?:发行人|发行主体|公司名称)[：:]\s*([^\n]+)',
            'issuer_alt': r'([^，,]+(?:有限公司|股份有限公司|集团))',
            
            # 发行规模/金额
            'issue_size': r'(?:发行规模|发行金额|募集资金总额)[：:]\s*([^\n]+)',
            'amount': r'(\d+(?:\.\d+)?)\s*(?:亿|万)?\s*元',
            
            # 信用评级
            'credit_rating': r'(?:主体信用评级|债项评级)[：:]\s*([AAA|AA|A|BBB]+)',
            'rating_agency': r'(?:联合资信|中诚信|大公国际|新世纪)[^：:]*',
            
            # 中介机构
            'lead_underwriter': r'(?:主承销商|牵头主承销商)[：:]\s*([^\n]+)',
            'trustee': r'(?:受托管理人|债券受托管理人)[：:]\s*([^\n]+)',
            'lawyer': r'(?:律师事务所|法律顾问)[：:]\s*([^\n]+)',
            'auditor': r'(?:会计师事务所|审计机构)[：:]\s*([^\n]+)',
            
            # 债券期限
            'bond_term': r'(?:债券期限|发行期限)[：:]\s*([^\n]+)',
            
            # 利率
            'interest_rate': r'(?:票面利率|发行利率)[：:]\s*([^\n]+)',
            
            # 担保情况
            'guarantee': r'(?:担保|保证)[：:]\s*([^\n]+)',
        }
        
        # 条款编号模式
        self.article_pattern = r'第[一二三四五六七八九十百千万]+条'
        self.chapter_pattern = r'第[一二三四五六七八九十百千万]+章'
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析合同文本，提取关键信息"""
        result = {
            'issuer': None,
            'issue_size': None,
            'amounts': [],
            'credit_rating': None,
            'lead_underwriter': None,
            'trustee': None,
            'lawyer': None,
            'auditor': None,
            'bond_term': None,
            'interest_rate': None,
            'guarantee': None,
            'articles': [],
            'chapters': [],
        }
        
        # 提取各项信息
        for key, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and len(value) < 200:  # 过滤过长内容
                    result[key] = value
        
        # 提取所有金额
        amounts = re.findall(r'(\d+(?:\.\d+)?)\s*([亿万千百]?)\s*元?', text)
        for amount, unit in amounts:
            result['amounts'].append({
                'value': float(amount),
                'unit': unit if unit else '元',
                'text': f"{amount}{unit}元"
            })
        
        # 提取条款
        article_matches = re.finditer(r'(第[一二三四五六七八九十百千万]+条)\s*([^第]+?)(?=第[一二三四五六七八九十百千万]+条|$)', text, re.DOTALL)
        for match in article_matches:
            result['articles'].append({
                'number': match.group(1),
                'content': match.group(2).strip()[:500]  # 限制长度
            })
        
        # 提取章节
        chapter_matches = re.finditer(r'(第[一二三四五六七八九十百千万]+章)\s*([^第]+?)(?=第[一二三四五六七八九十百千万]+章|$)', text, re.DOTALL)
        for match in chapter_matches:
            result['chapters'].append({
                'number': match.group(1),
                'title': match.group(2).strip()[:200]
            })
        
        return result
    
    def compare_documents(self, doc1_text: str, doc2_text: str) -> Dict[str, Any]:
        """比较两份合同的差异"""
        parsed1 = self.parse(doc1_text)
        parsed2 = self.parse(doc2_text)
        
        comparison = {
            'issuer_match': parsed1['issuer'] == parsed2['issuer'],
            'issuer1': parsed1['issuer'],
            'issuer2': parsed2['issuer'],
            'amount_comparison': self._compare_amounts(parsed1['amounts'], parsed2['amounts']),
            'rating_match': parsed1['credit_rating'] == parsed2['credit_rating'],
            'rating1': parsed1['credit_rating'],
            'rating2': parsed2['credit_rating'],
            'trustee_match': parsed1['trustee'] == parsed2['trustee'],
            'trustee1': parsed1['trustee'],
            'trustee2': parsed2['trustee'],
        }
        
        return comparison
    
    def _compare_amounts(self, amounts1: List, amounts2: List) -> Dict:
        """比较金额大小"""
        total1 = sum(a['value'] for a in amounts1 if a['unit'] in ['亿', ''])
        total2 = sum(a['value'] for a in amounts2 if a['unit'] in ['亿', ''])
        
        if total1 > total2:
            return {'result': 'greater', 'diff': total1 - total2, 'total1': total1, 'total2': total2}
        elif total1 < total2:
            return {'result': 'less', 'diff': total2 - total1, 'total1': total1, 'total2': total2}
        else:
            return {'result': 'equal', 'total1': total1, 'total2': total2}


class InsuranceParser:
    """保险条款专用解析器"""
    
    def parse_policy(self, text: str) -> Dict[str, Any]:
        """提取保险责任、免责条款、现金价值等"""
        patterns = {
            'coverage': r'保险责任[：:]\s*(.*?)(?=责任免除|$)',
            'exclusion': r'责任免除[：:]\s*(.*?)(?=保险金额|$)',
            'premium': r'保险费[：:]\s*(.*?)(?=保险期间|$)',
            'cash_value': r'现金价值[：:]\s*(.*?)(?=\n\n|$)',
            'insured_amount': r'保险金额[：:]\s*(.*?)(?=保险费|$)',
            'beneficiary': r'(?:受益人|身故保险金受益人)[：:]\s*(.*?)(?=\n|$)',
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted[key] = match.group(1).strip()
        
        return extracted


class RegulationParser:
    """监管法规专用解析器"""
    
    def parse_regulation(self, text: str) -> List[Dict]:
        """提取法条编号、内容等"""
        articles = []
        pattern = r'(第[一二三四五六七八九十百千万]+条)\s*(.*?)(?=第[一二三四五六七八九十百千万]+条|$)'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            article_num = match.group(1)
            content = match.group(2).strip()
            if len(content) > 50:  # 过滤空条款
                articles.append({
                    'number': article_num,
                    'content': content[:800],
                    'keywords': self._extract_keywords(content)
                })
        
        return articles
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        # 提取"应当"、"可以"、"必须"等
        modal_words = re.findall(r'(应当|可以|必须|不得|禁止)', text)
        keywords.extend(modal_words)
        return keywords


# 测试代码
if __name__ == '__main__':
    print("=" * 50)
    print("测试领域解析器")
    print("=" * 50)
    
    # 测试合同解析器
    contract_parser = ContractParser()
    
    test_text = """
    发行人：广东省广晟控股集团有限公司
    发行规模：不超过50亿元
    主体信用评级：AAA
    主承销商：中信证券股份有限公司
    受托管理人：国信证券股份有限公司
    债券期限：3年
    票面利率：2.95%
    """
    
    result = contract_parser.parse(test_text)
    print("\n合同解析结果:")
    print(f"  发行人: {result['issuer']}")
    print(f"  发行规模: {result['issue_size']}")
    print(f"  信用评级: {result['credit_rating']}")
    print(f"  主承销商: {result['lead_underwriter']}")
    print(f"  受托管理人: {result['trustee']}")
    
    # 测试保险解析器
    insurance_parser = InsuranceParser()
    print("\n" + "=" * 50)
    print("测试保险解析器")
    print("=" * 50)
    print("保险解析器已就绪")
    
    # 测试法规解析器
    regulation_parser = RegulationParser()
    print("\n" + "=" * 50)
    print("测试法规解析器")
    print("=" * 50)
    print("法规解析器已就绪")