
"""
被测模块：core/cache_manager.py
被测类：
    1. LocalCacheBackend  本地内存缓存后端（支持 TTL 过期、容量淘汰）
    2. CacheManager       缓存管理器（本地/Redis 双后端 + 自动降级 + 击穿保护）

测试：
    - TTL 用例通过真实的 time.sleep(1.1) 触发过期（TTL 最小单位为秒，这是需要短暂等待的用例）
    - get_or_set 的"缓存穿透保护"（并发下 loader 只执行一次）是核心被测点，
                                 通过"loader 调用次数 == 1"来断言
    - Redis 降级用例传入一个肯定连不上的地址（端口 1），验证自动 fallback
================================================================================
"""
import time

from core.cache_manager import LocalCacheBackend, CacheManager

class TestLocalCacheBackend:
    """
    本地内存缓存测试
    ------------------------------------------------------------------------
    LocalCacheBackend ：
      - set(key, value, ttl)：写入并指定存活时间（秒）
      - get(key)：读取；若 key 不存在或已过期返回 None
      - delete(key)：删除
      - exists(key)：判断是否存在（未过期）
      - 容量超限时，按过期时间淘汰最旧的 20%+1 项
    """

    def test_set_get(self):
        """最基本的写入→读取闭环。"""
        c = LocalCacheBackend()
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_missing_returns_none(self):
        """读取不存在的 key 应返回 None（而非抛异常）。"""
        c = LocalCacheBackend()
        assert c.get("nope") is None

    def test_ttl_expiry(self):
        """TTL 过期后 get 应返回 None。
        注意：这是唯一使用真实 sleep 的用例
        TTL=1 秒， 等待 1.1 秒确保过期。"""
        c = LocalCacheBackend()
        c.set("k", "v", ttl=1)
        assert c.get("k") == "v"
        time.sleep(1.1)
        assert c.get("k") is None

    def test_delete(self):
        """delete 后 get 应返回 None。"""
        c = LocalCacheBackend()
        c.set("k", "v")
        c.delete("k")
        assert c.get("k") is None

    def test_exists(self):
        """exists 应准确反映 key 的状态。
        覆盖三种状态：不存在 → False；写入后 → True；删除后 → False。"""
        c = LocalCacheBackend()
        assert c.exists("k") is False
        c.set("k", "v")
        assert c.exists("k") is True
        c.delete("k")
        assert c.exists("k") is False

    def test_capacity_eviction(self):
        """超过容量 max_items 时，按过期时间淘汰最旧项。"""
        c = LocalCacheBackend(max_items=5)
        for i in range(10):
            c.set(f"k{i}", i, ttl=10)
        assert len(c._cache) <= 5

    def test_different_types(self):
        """缓存值支持任意类型（dict / list 等），验证不做类型限制。"""
        c = LocalCacheBackend()
        c.set("dict", {"a": 1})
        c.set("list", [1, 2, 3])
        assert c.get("dict") == {"a": 1}
        assert c.get("list") == [1, 2, 3]

class TestCacheManager:
    """
    缓存管理器测试
    ------------------------------------------------------------------------
    CacheManager 是统一入口：
      - 默认使用本地内存后端（backend_type == "local"）
      - 支持 Redis 分布式后端，Redis 不可用时自动降级本地（fallback_to_local）
      - get_or_set(key, loader)：key 未命中时调用 loader 计算并缓存，命中时直接返回缓存值 ；
                               用于防止"缓存穿透"（大量重复请求打到下游）
      - 内部用 per-key 锁保证并发下 loader 只执行一次（击穿保护）
    """

    def test_defaults_to_local(self):
        """默认后端是本地内存"""
        cm = CacheManager()
        assert cm.backend_type == "local"

    def test_get_or_set_loads_once(self):
        """get_or_set ：同一 key 多次调用，loader 只执行一次。
        通过计数器 calls 进行断言：第一次调用触发 loader（calls=1），
        后续调用直接命中缓存，loader 不再执行（仍为 1）。"""
        cm = CacheManager()
        calls = []
        def loader():
            calls.append(1)
            return "result"
        assert cm.get_or_set("k", loader) == "result"
        assert cm.get_or_set("k", loader) == "result"
        assert len(calls) == 1

    def test_get_or_set_without_lock(self):
        """use_lock=False 时（不加锁）也能正常缓存（关闭击穿保护）。"""
        cm = CacheManager()
        def loader():
            return "x"
        assert cm.get_or_set("k2", loader, use_lock=False) == "x"
        assert cm.get("k2") == "x"

    def test_get_or_set_ttl(self):
        """get_or_set 支持 TTL：缓存过期后再次调用会重新执行 loader。
        验证完整生命周期：写入 → 有效期内命中 → 过期 → 重新加载（calls 变 2）。"""
        cm = CacheManager()
        calls = []
        def loader():
            calls.append(1)
            return "v"
        cm.get_or_set("k3", loader, ttl=1)
        assert cm.get("k3") == "v"
        time.sleep(1.1)
        assert cm.get("k3") is None
        assert cm.get_or_set("k3", loader, ttl=1) == "v"
        assert len(calls) == 2

    def test_delete_removes_value_and_lock(self):
        """delete 不仅清除缓存值，还应清理对应的 per-key 锁，
        避免锁对象长期堆积导致内存泄漏。"""
        cm = CacheManager()
        cm.set("k4", "v")
        cm.delete("k4")
        assert cm.get("k4") is None
        assert "k4" not in cm._locks

    def test_fallback_to_local_when_redis_down(self):
        """Redis 不可用时自动降级到本地缓存。"""
        cm = CacheManager(redis_url="redis://127.0.0.1:1/0", fallback_to_local=True)
        assert cm.backend_type == "local"
