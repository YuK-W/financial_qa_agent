# pdf_parser.py
import pdfplumber
import re
from typing import Dict, Any, List, Optional

class FinancialPDFParser:
    def __init__(self):
        self.section_patterns = {
            'article': r'第[一二三四五六七八九十百千万]+条',
            'chapter': r'第[一二三四五六七八九十百千万]+章',
            'section': r'\d+\.\d+',
        }
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """解析PDF，提取结构化内容"""
        with pdfplumber.open(pdf_path) as pdf:
            full_text = []
            tables = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    paragraphs = self._split_paragraphs(text)
                    full_text.extend(paragraphs)
                
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
            
            # P3-2: 表格优化处理
            enhanced_tables = self._process_tables_enhanced(tables)

            return {
                'raw_text': '\n'.join(full_text),
                'paragraphs': full_text,
                'tables': tables,
                'enhanced_tables': enhanced_tables,        # P3-2: 增强表格文本
                'table_texts': [t['text'] for t in enhanced_tables],  # 表格文本列表
                'structure': self._build_structure(full_text),
                'metadata': self._extract_metadata(full_text)
            }
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """智能分割段落"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _build_structure(self, paragraphs: List[str]) -> List[Dict]:
        """构建文档结构（章节、条款）"""
        structure = []
        current_chapter = None
        
        for para in paragraphs:
            if re.match(r'第[一二三四五六七八九十百千万]+章', para):
                current_chapter = {'title': para, 'articles': []}
                structure.append(current_chapter)
            elif re.match(r'第[一二三四五六七八九十百千万]+条', para):
                if current_chapter:
                    current_chapter['articles'].append(para)
                else:
                    structure.append({'article': para})
        
        return structure
    
    def _extract_metadata(self, paragraphs: List[str]) -> Dict:
        """提取元数据（标题、日期）"""
        metadata = {}
        text_sample = '\n'.join(paragraphs[:20])
        
        # 提取标题
        for line in text_sample.split('\n')[:10]:
            if any(kw in line for kw in ['募集说明书', '公司章程', '保险条款']):
                metadata['title'] = line.strip()
                break
        
        # 提取日期
        date_pattern = r'20\d{2}年\d{1,2}月\d{1,2}日'
        dates = re.findall(date_pattern, text_sample)
        if dates:
            metadata['dates'] = dates
        
        return metadata

    # ================================================================
    # P3-2: 增强表格处理
    # ================================================================
    @staticmethod
    def _normalize_cell(cell) -> str:
        """
        规范化单元格内容。

        处理:
          - None → 空字符串
          - 多行文本 → 用空格合并（防止换行打散数值）
          - 前后空白清理
        """
        if cell is None:
            return ''
        text = str(cell).strip()
        # 合并内部换行（防止 "100\n亿\n元" 这种打散）
        text = re.sub(r'\n+', ' ', text)
        # 合并多余空格
        text = re.sub(r'\s{2,}', ' ', text)
        return text

    @staticmethod
    def _is_numeric_cell(text: str) -> bool:
        """判断单元格是否为数值型（数字+单位/百分比）"""
        if not text:
            return False
        # 匹配: 100, 100.5, 100亿, 100万元, 75%, 3,000,000
        return bool(re.match(r'^[\d,.\s]+(?:[亿万千百]?元?|%)?$', text))

    @staticmethod
    def _merge_split_values(row: List[str]) -> List[str]:
        """
        合并被 PDF 换行打散的数值单元格。

        场景: "100\n亿\n元" 分布在3行 → 应合并为 "100亿元"
        """
        merged = []
        i = 0
        while i < len(row):
            cell = row[i]
            # 检查后续单元格是否为当前数值的延续（纯单位）
            if FinancialPDFParser._is_numeric_cell(cell):
                j = i + 1
                while j < len(row) and row[j] and re.match(r'^[亿万千百元%]+$', row[j]):
                    cell += row[j]
                    j += 1
                merged.append(cell)
                i = j
            else:
                merged.append(cell)
                i += 1
        return merged

    def _process_tables_enhanced(self, tables: List) -> List[Dict]:
        """
        增强表格处理: 规范化 + 结构化文本输出。

        Returns:
            [{'headers': [...], 'rows': [[...], ...], 'text': '表格文本', 'numeric_data': {...}}, ...]
        """
        enhanced = []
        for table in tables:
            if not table or len(table) < 2:
                continue

            # 规范化所有单元格
            norm_table = []
            for row in table:
                norm_row = [self._normalize_cell(cell) for cell in row]
                # 过滤全空行
                if any(norm_row):
                    norm_table.append(norm_row)

            if len(norm_table) < 2:
                continue

            # 表头 + 数据行
            headers = norm_table[0]
            rows = norm_table[1:]

            # P3-2核心: 合并被换行打散的数值
            rows = [self._merge_split_values(row) for row in rows]

            # 构建结构化文本（表头+数据行的可读形式）
            text_lines = ["[表格]"]
            text_lines.append(" | ".join(h for h in headers if h))
            text_lines.append("-" * 40)
            for row in rows:
                # 对齐列数
                padded = row + [''] * (len(headers) - len(row))
                text_lines.append(" | ".join(
                    f"{h}: {v}" for h, v in zip(headers, padded) if v
                ))

            # 提取数值数据
            numeric_data = {}
            for row in rows:
                for i, cell in enumerate(row):
                    if i < len(headers) and self._is_numeric_cell(cell):
                        key = headers[i] if i < len(headers) else f"col_{i}"
                        if key not in numeric_data:
                            numeric_data[key] = []
                        numeric_data[key].append(cell)

            enhanced.append({
                'headers': headers,
                'rows': rows,
                'text': '\n'.join(text_lines),
                'numeric_data': numeric_data,
            })

        return enhanced

    # ================================================================
    # 工具: 表格文本供检索使用
    # ================================================================
    def get_table_texts(self, parsed: Dict) -> str:
        """将所有增强表格文本合并为一个字符串（供检索索引用）"""
        table_texts = parsed.get('table_texts', [])
        return '\n\n'.join(table_texts)