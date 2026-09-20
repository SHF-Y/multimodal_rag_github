
"""
缓存管理器：支持本地内存缓存和Redis分布式缓存两种后端
设计要点：
1. 统一接口，通过配置切换后端
2. Redis模式支持分布式部署、多实例共享
3. 本地模式作为降级方案（Redis不可用时自动回退）
4. 支持缓存击穿保护：热点key加互斥锁
"""
import json
import time
import threading
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class CacheBackend:
    """定义缓存后端基类"""
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int = 600):
        raise NotImplementedError

    def delete(self, key: str):
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

class LocalCacheBackend(CacheBackend):
    """本地内存缓存（带TTL和容量限制）"""
    def __init__(self, max_items: int = 1000):
        self._cache = {}
        self._lock = threading.Lock()
        self.max_items = max_items

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            if item["expire"] < time.time():
                del self._cache[key]
                return None
            return item["value"]

    def set(self, key: str, value: Any, ttl: int = 600):
        with self._lock:

            if len(self._cache) >= self.max_items:
                sorted_items = sorted(self._cache.items(), key=lambda x: x[1]["expire"])
                delete_count = int(self.max_items * 0.2) + 1
                for k, _ in sorted_items[:delete_count]:
                    del self._cache[k]
            self._cache[key] = {"value": value, "expire": time.time() + ttl}

    def delete(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

class RedisCacheBackend(CacheBackend):
    """Redis分布式缓存"""
    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 key_prefix: str = "rag:"):
        import redis
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

        self.key_prefix = key_prefix

        try:
            self.redis_client.ping()
            logger.info("Redis缓存连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.redis_client.get(self._make_key(key))

            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get失败: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 600):
        try:
            self.redis_client.setex(
                self._make_key(key),
                ttl,
                json.dumps(value, ensure_ascii=False)

            )
        except Exception as e:
            logger.error(f"Redis set失败: {e}")

    def delete(self, key: str):
        try:
            self.redis_client.delete(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis delete失败: {e}")

    def exists(self, key: str) -> bool:
        try:
            return self.redis_client.exists(self._make_key(key)) > 0
        except Exception:
            return False

class CacheManager:
    """
    缓存管理器：自动降级 + 缓存击穿保护
    - 优先使用Redis，连接失败自动降级到本地缓存
    - key获取时加互斥锁，防止缓存击穿
    """
    def __init__(self, redis_url: Optional[str] = None,
                 fallback_to_local: bool = True):
        self.backend: CacheBackend
        self._locks = {}
        self._locks_lock = threading.Lock()

        if redis_url:
            try:
                self.backend = RedisCacheBackend(redis_url)
                self.backend_type = "redis"
            except Exception as e:
                logger.warning(f"Redis不可用，降级到本地缓存: {e}")
                if fallback_to_local:
                    self.backend = LocalCacheBackend()
                    self.backend_type = "local"
                else:
                    raise
        else:
            self.backend = LocalCacheBackend()
            self.backend_type = "local"

        logger.info(f"缓存管理器初始化完成，后端: {self.backend_type}")

    def _get_key_lock(self, key: str) -> threading.Lock:
        """获取key对应的锁（用于防止缓存击穿）"""
        with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl: int = 600):
        self.backend.set(key, value, ttl)

    def get_or_set(self, key: str, loader_func, ttl: int = 600,
                   use_lock: bool = True) -> Any:
        """
        获取缓存，未命中时调用loader_func加载并写入缓存
        Args:
            key: 缓存键
            loader_func: 数据加载函数（无参数）
            ttl: 过期时间（秒）
            use_lock: 是否使用互斥锁防止缓存击穿
        """

        value = self.get(key)
        if value is not None:
            return value

        if use_lock:
            lock = self._get_key_lock(key)
            with lock:

                value = self.get(key)
                if value is not None:
                    return value

                value = loader_func()
                self.set(key, value, ttl)
                return value
        else:
            value = loader_func()
            self.set(key, value, ttl)
            return value

    def delete(self, key: str):
        self.backend.delete(key)
        with self._locks_lock:

            self._locks.pop(key, None)

    def clear_pattern(self, pattern: str):
        if self.backend_type == "redis":
            try:
                match = self.backend._make_key(pattern)
                keys = list(self.backend.redis_client.scan_iter(match))
                if keys:
                    self.backend.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear_pattern失败: {e}")
