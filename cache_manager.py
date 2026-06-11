# cache_manager.py
"""
Token 缓存管理器

使用 OrderedDict 实现真正的 LRU（最近最少使用）淘汰策略。
每次缓存命中时将条目移到末尾，淘汰时删除最久未访问的条目。
"""
from collections import OrderedDict
from typing import Any, Callable, Dict


class TokenCache:
    """
    LRU 缓存，用于存储文档解析结果、token 统计等计算结果。

    使用方式:
        cache = TokenCache(max_size=100)
        result = cache.get_or_compute("doc_text01", lambda: parse_pdf("text01.pdf"))
    """

    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: 最大缓存条目数，超过后按 LRU 淘汰
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0        # 缓存命中次数
        self.misses = 0      # 缓存未命中次数

    def get_or_compute(self, key: str, compute_func: Callable[[], Any]) -> Any:
        """
        获取缓存值，若不存在则通过 compute_func 计算后缓存。

        LRU 策略:
          - 命中时: 将该条目移到 OrderedDict 末尾（标记为最近使用）
          - 未命中时: 计算 → 写入 → 若超 max_size 则淘汰最久未使用的条目（头部）

        Args:
            key: 缓存键
            compute_func: 无参计算函数，仅在缓存未命中时调用

        Returns:
            缓存或计算得到的结果
        """
        # ---- 缓存命中: 移到末尾（最近使用） ----
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]

        # ---- 缓存未命中: 计算 + 写入 ----
        self.misses += 1
        try:
            result = compute_func()
        except Exception as e:
            print(f"  [TokenCache] 计算函数异常 [{key}]: {e}")
            raise

        # 若缓存已满，淘汰最久未使用的条目（OrderedDict 头部）
        if len(self.cache) >= self.max_size:
            evicted_key, _ = self.cache.popitem(last=False)
            print(f"  [TokenCache] LRU淘汰: {evicted_key}")

        self.cache[key] = result
        return result

    def get(self, key: str) -> Any:
        """
        直接获取缓存值（不触发计算）。

        Returns:
            缓存值，不存在返回 None
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """手动写入缓存"""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            evicted_key, _ = self.cache.popitem(last=False)
            print(f"  [TokenCache] LRU淘汰: {evicted_key}")
        self.cache[key] = value

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1%}",
        }

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)


# ================================================================
# 测试: 验证 LRU 行为正确
# ================================================================
if __name__ == '__main__':
    cache = TokenCache(max_size=3)

    # 写入 3 个条目
    cache.put("A", "数据A")
    cache.put("B", "数据B")
    cache.put("C", "数据C")
    print(f"初始: {list(cache.cache.keys())}")  # ['A', 'B', 'C']

    # 访问 A → A 移到末尾（最近使用）
    cache.get("A")
    print(f"访问A后: {list(cache.cache.keys())}")  # ['B', 'C', 'A']

    # 写入 D → 淘汰最久未使用（B 在头部）
    cache.put("D", "数据D")
    print(f"写入D后: {list(cache.cache.keys())}")  # ['C', 'A', 'D']
    print(f"B被淘汰: {'B' not in cache}")           # True

    # 验证统计
    print(f"\n统计: {cache.stats}")

    print("\nLRU 行为验证: PASS" if 'B' not in cache and list(cache.cache.keys()) == ['C', 'A', 'D'] else "FAIL")
