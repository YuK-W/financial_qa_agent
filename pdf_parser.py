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
            
            return {
                'raw_text': '\n'.join(full_text),
                'paragraphs': full_text,
                'tables': tables,
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