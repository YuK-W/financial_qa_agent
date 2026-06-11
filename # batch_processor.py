# batch_processor.py
class BatchProcessor:
    def __init__(self, agent):
        self.agent = agent
    
    def batch_answer(self, questions_batch):
        """批量处理多道题目，复用文档"""
        # 按文档分组
        doc_groups = {}
        for q in questions_batch:
            for doc_id in q.get('doc_ids', []):
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(q)
        
        # 预加载文档
        doc_contents = {}
        for doc_id in doc_groups:
            doc_contents[doc_id] = self.load_document(doc_id)
        
        # 批量处理
        results = []
        for q in questions_batch:
            result = self.agent.answer_question(q)
            results.append(result)
        
        return results