# cache_manager.py
class TokenCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get_or_compute(self, key, compute_func):
        """缓存计算结果"""
        if key in self.cache:
            return self.cache[key]
        
        result = compute_func()
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result
        return result