# chunking.py
import re
from typing import List, Dict, Any

class DocumentChunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将文档切分成适合检索的块"""
        chunks = []
        
        # 策略1：按段落切分
        for para in parsed_doc.get('paragraphs', []):
            if not para or len(para.strip()) < 10:
                continue
                
            if len(para) <= self.chunk_size:
                chunks.append({
                    'content': para,
                    'type': 'paragraph',
                    'metadata': self._extract_metadata_from_para(para)
                })
            else:
                # 长段落进一步切分
                sub_chunks = self._split_long_paragraph(para)
                chunks.extend(sub_chunks)
        
        # 策略2：按表格切分
        for table in parsed_doc.get('tables', []):
            if table:
                table_text = self._table_to_text(table)
                chunks.append({
                    'content': table_text,
                    'type': 'table',
                    'metadata': {'is_table': True}
                })
        
        return chunks
    
    def _split_long_paragraph(self, para: str) -> List[Dict[str, Any]]:
        """切分长段落，保持语义完整性"""
        # 按句子切分
        sentences = re.split(r'[。！？；]', para)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            if current_length + len(sent) > self.chunk_size:
                # 保存当前块
                chunk_text = '。'.join(current_chunk) + '。'
                chunks.append({
                    'content': chunk_text,
                    'type': 'paragraph_chunk',
                    'metadata': {'is_split': True}
                })
                # 保留重叠部分（最后2个句子）
                if len(current_chunk) > 2:
                    current_chunk = current_chunk[-2:]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sent)
            current_length += len(sent)
        
        # 最后一块
        if current_chunk:
            chunk_text = '。'.join(current_chunk) + '。'
            chunks.append({
                'content': chunk_text,
                'type': 'paragraph_chunk',
                'metadata': {'is_split': True}
            })
        
        return chunks
    
    def _extract_metadata_from_para(self, para: str) -> Dict[str, Any]:
        """从段落中提取元数据（条款编号等）"""
        metadata = {}
        
        # 提取条款编号（如：第三条、第十条）
        article_match = re.search(r'第[一二三四五六七八九十百千万]+条', para)
        if article_match:
            metadata['article'] = article_match.group()
        
        # 提取章节编号
        chapter_match = re.search(r'第[一二三四五六七八九十百千万]+章', para)
        if chapter_match:
            metadata['chapter'] = chapter_match.group()
        
        # 提取数字（金额、比例等）
        numbers = re.findall(r'\d+(?:\.\d+)?%?', para)
        if numbers:
            metadata['numbers'] = numbers[:5]  # 最多5个
        
        return metadata
    
    def _table_to_text(self, table: List[List]) -> str:
        """将表格转换为文本"""
        if not table or not table[0]:
            return ""
        
        # 提取表头
        headers = [str(cell).strip() if cell else '' for cell in table[0]]
        
        # 提取数据行
        rows = []
        for row in table[1:]:
            row_text = [str(cell).strip() if cell else '' for cell in row]
            rows.append(' | '.join(row_text))
        
        # 构建表格文本
        table_text = "表格内容：\n"
        table_text += " | ".join(headers) + "\n"
        for row in rows:
            table_text += row + "\n"
        
        return table_text


# 测试代码
if __name__ == '__main__':
    # 测试分块器
    chunker = DocumentChunker(chunk_size=200, overlap=30)
    
    # 模拟解析后的文档
    test_doc = {
        'paragraphs': [
            "这是一个短段落，不需要切分。",
            "这是一个很长的段落。" * 20 + "需要切分成多个小块。",
        ],
        'tables': [
            [['名称', '金额'], ['公司A', '100万'], ['公司B', '200万']]
        ]
    }
    
    chunks = chunker.chunk_document(test_doc)
    print(f"生成 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"块{i+1} [{chunk['type']}]: {chunk['content'][:50]}...")