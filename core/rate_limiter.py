"""
限流器与熔断器
1. 令牌桶限流：按IP/用户维度限制请求频率
2. 熔断器：监控LLM API错误率，超过阈值时熔断一段时间
"""
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TokenBucket:
    """
    令牌桶限流器（单桶）
    桶中初始有一定数量的令牌，每秒按固定速率补充令牌。
    每个请求需要消耗1个令牌才能通过，如果桶中令牌不足则拒绝。
    """
    def __init__(self, rate: float = 10, capacity: int = 20):
        """
        初始化令牌桶
            rate: 每秒生成令牌数（即允许的平均请求速率）
            capacity: 桶容量（允许的最大突发请求数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        """
        补充令牌（内部方法，需在持有锁的情况下调用）
        根据距离上次补充的时间，计算这段时间内新增的令牌数，但不能超过桶容量。
        """
        now = time.time()
        elapsed = now - self.last_refill

        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        尝试获取令牌（默认1个）
        返回 True 表示获取成功（请求允许通过），False 表示令牌不足（请求被限流）。
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def available(self) -> float:
        """
        查看当前可用令牌数（不消耗令牌）
        """
        with self.lock:
            self._refill()
            return self.tokens

class RateLimiter:
    """
    多维度限流器：
    - 全局限流：限制整个系统的总请求速率
    - 按IP限流：限制单个IP的请求速率（防止单IP刷接口）
    每个IP对应一个独立的TokenBucket实例，存储在字典中。
    检查流程采用“先预判后消耗”策略，避免IP被限流时仍消耗全局令牌。
    """
    def __init__(self, global_rate: float = 100, global_capacity: int = 200,
                 per_ip_rate: float = 10, per_ip_capacity: int = 20):
        """
        初始化限流器
            global_rate: 全局限流速率（令牌/秒）
            global_capacity: 全局桶容量
            per_ip_rate: 每个IP的限流速率
            per_ip_capacity: 每个IP桶容量
        """

        self.global_bucket = TokenBucket(global_rate, global_capacity)
        

        self.ip_buckets: Dict[str, TokenBucket] = {}
        

        self.per_ip_rate = per_ip_rate
        self.per_ip_capacity = per_ip_capacity
        

        self.lock = threading.Lock()
        

        self.max_buckets = 10000

    def _get_ip_bucket(self, ip: str) -> TokenBucket:
        """
        获取指定IP的令牌桶，如果不存在则创建。
        使用锁保护字典操作。
        当桶数量超过上限时，淘汰最久未使用的20%桶（根据last_refill排序）。
        """
        with self.lock:
            if ip not in self.ip_buckets:

                if len(self.ip_buckets) >= self.max_buckets:

                    sorted_ips = sorted(self.ip_buckets.items(), key=lambda x: x[1].last_refill)

                    delete_count = int(self.max_buckets * 0.2) + 1
                    for old_ip, _ in sorted_ips[:delete_count]:
                        del self.ip_buckets[old_ip]
                

                self.ip_buckets[ip] = TokenBucket(self.per_ip_rate, self.per_ip_capacity)
            return self.ip_buckets[ip]

    def check(self, ip: Optional[str] = None) -> tuple[bool, str]:
        """
        检查请求是否被允许。返回 (allowed, reason)：
        - allowed=True 表示允许请求
        - allowed=False 表示拒绝，reason为拒绝原因字符串
        先预判IP限流（只检查不消耗令牌），通过后再消耗全局令牌，
        避免IP被限流的请求白白消耗全局令牌，导致全局令牌被浪费。
        """

        if ip:
            bucket = self._get_ip_bucket(ip)
            with bucket.lock:
                bucket._refill()
                if bucket.tokens < 1:
                    return False, f"IP {ip} 请求频率超限，请稍后重试"

        if not self.global_bucket.acquire():
            return False, "全局请求频率超限，请稍后重试"

        if ip:
            bucket = self._get_ip_bucket(ip)
            bucket.acquire()

        return True, "ok"

    def cleanup_idle_buckets(self, idle_seconds: int = 300):
        """
        清理长时间空闲的IP桶（定期调用，防止内存泄漏）。
        空闲定义：距离上次补充令牌超过idle_seconds秒。
        """
        with self.lock:
            now = time.time()

            idle_ips = [
                ip for ip, bucket in self.ip_buckets.items()
                if now - bucket.last_refill > idle_seconds
            ]

            for ip in idle_ips:
                del self.ip_buckets[ip]

        logger.info(f"清理了 {len(idle_ips)} 个空闲IP限流桶")

class CircuitBreaker:
    """
    熔断器：三态（Closed -> Open -> Half-Open）
    - Closed（关闭）：正常状态，请求通过，统计连续失败次数。
    - Open（打开）：熔断状态，直接拒绝所有请求，等待一段时间后进入半开。
    - Half-Open（半开）：允许少量请求试探，若成功则关闭熔断器，若失败则重新打开。
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str = "default",
                 failure_threshold: int = 5,
                 recovery_timeout: int = 30,
                 half_open_max_requests: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_requests = 0
        
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        """
        检查是否允许请求通过。根据当前状态决定是否允许：
        - Closed：总是允许（True）
        - Open：若熔断时间已超过recovery_timeout，进入半开并允许(True)；否则拒绝(False)
        - Half-Open：允许的试探请求数未超过限制则允许(True)，否则拒绝(False)
        """
        with self.lock:
            if self.state == self.CLOSED:
                return True

            if self.state == self.OPEN:

                if time.time() - self.last_failure_time >= self.recovery_timeout:

                    self.state = self.HALF_OPEN
                    self.half_open_requests = 0
                    self.success_count = 0
                    logger.info(f"熔断器 {self.name} 进入半开状态")
                    return True
                return False

            if self.state == self.HALF_OPEN:

                if self.half_open_requests < self.half_open_max_requests:
                    self.half_open_requests += 1
                    return True
                return False

            return False

    def record_success(self):
        """
        记录一次请求成功。
        若在半开状态，累计成功次数，当成功次数达到half_open_max_requests时，认为服务恢复，关闭熔断器。
        若在关闭状态，重置失败计数（因为成功代表服务正常，连续失败计数应清零）。
        """
        with self.lock:
            if self.state == self.HALF_OPEN:
                self.success_count += 1

                if self.success_count >= self.half_open_max_requests:
                    self.state = self.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"熔断器 {self.name} 恢复关闭状态")
            elif self.state == self.CLOSED:

                self.failure_count = 0

    def record_failure(self):
        """
        记录一次请求失败。
        若在关闭状态，失败计数+1，达到阈值则打开熔断器。
        若在半开状态，立即回到打开状态（说明服务仍异常）。
        记录失败时间，用于计算恢复超时。
        """
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == self.HALF_OPEN:

                self.state = self.OPEN
                logger.warning(f"熔断器 {self.name} 半开状态失败，回到熔断状态")
            elif self.state == self.CLOSED:

                if self.failure_count >= self.failure_threshold:
                    self.state = self.OPEN
                    logger.warning(f"熔断器 {self.name} 触发熔断，连续失败 {self.failure_count} 次")

    def get_state(self) -> str:
        """获取当前熔断器状态"""
        with self.lock:
            return self.state