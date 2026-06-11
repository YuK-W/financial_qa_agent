# context_optimizer.py
class ContextOptimizer:
    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
    
    def optimize_context(self, evidences, question):
        """动态选择最相关的证据"""
        # 按相关度排序
        sorted_evidences = sorted(evidences, key=lambda x: x.get('relevance', 0), reverse=True)
        
        selected = []
        current_tokens = 0
        
        for ev in sorted_evidences:
            ev_tokens = self.estimate_tokens(ev['content'])
            if current_tokens + ev_tokens <= self.max_tokens:
                selected.append(ev)
                current_tokens += ev_tokens
            else:
                break
        
        return selected
    
    def estimate_tokens(self, text):
        """估算token数（中文约1.5字符/token）"""
        return len(text) // 1.5