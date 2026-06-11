# retrieval_system.py
import os
import re
import jieba
import pdfplumber
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi


class FinancialRetrievalSystem:
    """金融文档检索系统 - 不使用embedding"""
    
    def __init__(self):
        self.chunks = []           # 存储所有文档块
        self.keyword_index = defaultdict(list)   # 关键词 -> 块ID列表
        self.entity_index = defaultdict(list)    # 实体 -> 块ID列表
        self.bm25_index = None     # BM25索引
        self.tokenized_chunks = [] # 分词后的块内容
        
    def index_documents(self, docs_dir: str, doc_ids: List[str]) -> int:
        """
        索引多个文档
        docs_dir: 文档目录，如 './data/public_dataset_upload/raw/financial_contracts'
        doc_ids: 文档ID列表，如 ['text01', 'text02']
        """
        print(f"开始索引 {len(doc_ids)} 个文档...")
        
        all_chunks = []
        
        for doc_id in doc_ids:
            pdf_path = os.path.join(docs_dir, f"{doc_id}.pdf")
            if not os.path.exists(pdf_path):
                print(f"  警告: 未找到 {pdf_path}")
                continue
            
            print(f"  索引: {doc_id}")
            
            # 1. 解析PDF
            text = self._extract_pdf_text(pdf_path)
            
            # 2. 分块
            chunks = self._chunk_text(text, doc_id)
            
            # 3. 为每个块建立索引
            for chunk in chunks:
                chunk_id = f"{doc_id}_{len(all_chunks)}"
                chunk['id'] = chunk_id
                chunk['doc_id'] = doc_id
                all_chunks.append(chunk)
                
                # 关键词索引
                for kw in chunk['keywords']:
                    self.keyword_index[kw].append(chunk_id)
                
                # 实体索引
                for entity_type, entity_value in chunk['entities']:
                    self.entity_index[(entity_type, entity_value)].append(chunk_id)
        
        self.chunks = all_chunks
        
        # 4. 构建BM25索引
        self.tokenized_chunks = [self._tokenize(chunk['content']) for chunk in self.chunks]
        self.bm25_index = BM25Okapi(self.tokenized_chunks)
        
        print(f"索引完成: {len(self.chunks)} 个块, {len(self.keyword_index)} 个关键词")
        return len(self.chunks)
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """提取PDF文本"""
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text
    
    def _chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """将文本分块，每块约500字符"""
        chunks = []
        chunk_size = 500
        overlap = 50
        
        # 按段落分割
        paragraphs = text.split('\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 10:
                continue
            
            # 如果当前块加上新段落会超长，保存当前块
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(self._create_chunk(current_chunk, doc_id, chunk_index))
                chunk_index += 1
                # 保留重叠部分
                current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
            
            current_chunk += para + "\n"
        
        # 最后一块
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, doc_id, chunk_index))
        
        return chunks
    
    def _create_chunk(self, content: str, doc_id: str, index: int) -> Dict:
        """创建块对象"""
        return {
            'content': content,
            'doc_id': doc_id,
            'index': index,
            'keywords': self._extract_keywords(content),
            'entities': self._extract_entities(content)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（jieba分词 + 过滤）"""
        words = jieba.cut(text)
        keywords = set()
        for word in words:
            word = word.strip()
            if len(word) >= 2 and not word.isdigit():
                keywords.add(word)
        return list(keywords)
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """提取实体：金额、比例、日期"""
        entities = []
        
        # 金额：100亿、50亿元、8.5万
        amount_pattern = r'(\d+(?:\.\d+)?)\s*([万亿千百]?元?)'
        for match in re.finditer(amount_pattern, text):
            entities.append(('amount', match.group()))
        
        # 比例：75%、6.5%
        ratio_pattern = r'(\d+(?:\.\d+)?)%'
        for match in re.finditer(ratio_pattern, text):
            entities.append(('ratio', match.group()))
        
        # 日期：2024年、12月31日
        date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}年)'
        for match in re.finditer(date_pattern, text):
            entities.append(('date', match.group()))
        
        return entities
    
    def _tokenize(self, text: str) -> List[str]:
        """分词，用于BM25"""
        return list(jieba.cut(text))
    
    def retrieve(self, question: str, options: Dict[str, str], top_k: int = 10) -> List[Dict]:
        """
        多策略检索
        question: 问题
        options: 选项字典 {'A': 'xxx', 'B': 'xxx', ...}
        top_k: 返回结果数量
        """
        # 策略1: 关键词检索
        q_keywords = self._extract_keywords(question)
        keyword_results = self._search_by_keywords(q_keywords, top_k)
        
        # 策略2: BM25检索
        bm25_results = self._search_by_bm25(question, top_k)
        
        # 策略3: 实体检索
        q_entities = self._extract_entities(question)
        entity_results = self._search_by_entities(q_entities, top_k)
        
        # 策略4: 选项关键词检索
        option_results = set()
        for opt_text in options.values():
            opt_keywords = self._extract_keywords(opt_text)
            opt_results = self._search_by_keywords(opt_keywords, top_k // 2)
            option_results.update(opt_results)
        
        # 融合去重
        all_chunk_ids = set()
        all_chunk_ids.update(keyword_results)
        all_chunk_ids.update(bm25_results)
        all_chunk_ids.update(entity_results)
        all_chunk_ids.update(option_results)
        
        # 获取块内容并排序
        retrieved = [chunk for chunk in self.chunks if chunk['id'] in all_chunk_ids]
        retrieved = self._rerank(retrieved, question, options)
        
        return retrieved[:top_k]
    
    def _search_by_keywords(self, keywords: List[str], top_k: int) -> List[str]:
        """关键词倒排索引检索"""
        scores = defaultdict(int)
        for kw in keywords:
            for chunk_id in self.keyword_index.get(kw, []):
                scores[chunk_id] += 1
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:top_k]
    
    def _search_by_bm25(self, query: str, top_k: int) -> List[str]:
        """BM25全文检索"""
        if not self.bm25_index or not self.tokenized_chunks:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i]['id'] for i in top_indices if i < len(self.chunks)]
    
    def _search_by_entities(self, entities: List[Tuple[str, str]], top_k: int) -> List[str]:
        """实体检索"""
        chunk_ids = set()
        for entity_type, entity_value in entities:
            for chunk_id in self.entity_index.get((entity_type, entity_value), []):
                chunk_ids.add(chunk_id)
        return list(chunk_ids)[:top_k]
    
    def _rerank(self, chunks: List[Dict], question: str, options: Dict[str, str]) -> List[Dict]:
        """重排序"""
        q_keywords = set(self._extract_keywords(question))
        opt_keywords = set()
        for opt_text in options.values():
            opt_keywords.update(self._extract_keywords(opt_text))
        
        for chunk in chunks:
            chunk_keywords = set(chunk['keywords'])
            q_score = len(q_keywords & chunk_keywords) / max(len(q_keywords), 1)
            opt_score = len(opt_keywords & chunk_keywords) / max(len(opt_keywords), 1)
            chunk['relevance'] = q_score * 0.6 + opt_score * 0.4
        
        return sorted(chunks, key=lambda x: x.get('relevance', 0), reverse=True)
    
    def get_context(self, chunk_ids: List[str]) -> str:
        """根据块ID获取上下文文本"""
        id_to_chunk = {chunk['id']: chunk for chunk in self.chunks}
        contexts = []
        for chunk_id in chunk_ids:
            if chunk_id in id_to_chunk:
                contexts.append(f"[{id_to_chunk[chunk_id]['doc_id']}] {id_to_chunk[chunk_id]['content']}")
        return '\n\n'.join(contexts)


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("测试金融文档检索系统")
    print("=" * 60)
    
    # 创建检索系统
    retriever = FinancialRetrievalSystem()
    
    # 索引文档
    docs_dir = './data/public_dataset_upload/raw/financial_contracts'
    doc_ids = ['text01', 'text02']
    
    chunk_count = retriever.index_documents(docs_dir, doc_ids)
    print(f"\n共创建 {chunk_count} 个文档块")
    
    # 测试检索
    print("\n" + "=" * 60)
    print("测试检索")
    print("=" * 60)
    
    question = "发行人的名称是什么？"
    options = {
        'A': '广东省广晟控股集团有限公司',
        'B': '其他公司'
    }
    
    results = retriever.retrieve(question, options, top_k=5)
    print(f"检索到 {len(results)} 个相关块\n")
    
    for i, r in enumerate(results):
        print(f"--- 结果 {i+1} ---")
        print(f"文档: {r['doc_id']}")
        print(f"相关度: {r.get('relevance', 0):.3f}")
        print(f"内容: {r['content'][:150]}...")
        print()
    
    # 构建上下文
    context = retriever.get_context([r['id'] for r in results[:3]])
    print("=" * 60)
    print("构建的上下文（前500字符）")
    print("=" * 60)
    print(context[:500])