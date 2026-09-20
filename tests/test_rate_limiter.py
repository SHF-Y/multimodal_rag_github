
"""
被测模块：core/rate_limiter.py
被测类：
    1. TokenBucket   令牌桶算法（限流的核心数据结构）
    2. RateLimiter   多维限流器（全局桶 + 每 IP 桶 双层限流）
    3. CircuitBreaker 熔断器（Closed → Open → Half-Open 三态机）

测试思路：
    - 时间类用例不依赖真实时钟等待，而是直接操作内部字段（如 last_refill）
      来模拟"时间流逝"，保证测试快速且稳定（不 sleep、不 flaky）
    - 熔断器测试覆盖完整的生命周期：关闭 → 打开 → 半开 → 关闭 / 重开
"""
import time

from core.rate_limiter import TokenBucket, RateLimiter, CircuitBreaker

class TestTokenBucket:
    """
    令牌桶（TokenBucket）测试
    ------------------------------------------------------------------------
    令牌桶原理：桶里最多放 capacity 个令牌；每来一个请求消耗 1 个令牌；
        令牌按 rate（个/秒）的速度随时间自动补充，但不超过桶容量。
        作用是"平滑限流"：允许短时突发（桶内有积攒），但长期平均速率受限。
    """
    def test_initial_full(self):
        """刚创建的桶应该是"满桶"状态（有 capacity 个令牌可用）。
        这样系统启动后立即能放行请求，而不是要等令牌慢慢积累。"""
        tb = TokenBucket(rate=10, capacity=20)

        assert tb.available() == 20

    def test_acquire_consumes(self):
        """acquire() 成功时消耗一个令牌，令牌耗尽后 acquire() 返回 False。
        验证"每个请求消耗一个令牌、容量为 1 时只能放行一次"的最小场景。"""
        tb = TokenBucket(rate=10, capacity=1)

        assert tb.acquire() is True

        assert tb.acquire() is False

    def test_refill_math(self):
        """验证令牌按 rate 速率随时间自动补充。
        关键技巧：不真实等待，而是把内部时间戳 last_refill 手动"拨回 1 秒前"，
        再调用 available() 触发补充逻辑，从而验证：1 秒内应补充 rate=10 个令牌。"""
        tb = TokenBucket(rate=10, capacity=20)
        tb.tokens = 0
        tb.last_refill = time.time() - 1

        assert abs(tb.available() - 10) < 0.01

    def test_refill_capped_by_capacity(self):
        """补充令牌不能超过桶容量（容量是上限）。
        场景：速率 100/秒、容量只有 5，时间过去 10 秒本应补充 1000 个，
        但最多只能累积到 5 个——验证容量封顶逻辑，防止令牌无限积攒。"""
        tb = TokenBucket(rate=100, capacity=5)
        tb.tokens = 0
        tb.last_refill = time.time() - 10
        assert tb.available() == 5

class TestRateLimiter:
    """
    多维限流器（RateLimiter）测试组
    ------------------------------------------------------------------------
    RateLimiter 同时维护两个令牌桶：
      - 全局桶：限制整个服务总的请求速率（防止被打垮）
      -  IP 桶：限制单个来源 IP 的请求速率（防止被刷）
    check(ip) 会先查每 IP 桶、再查全局桶，返回 (是否放行, 拒绝原因)。
    """

    def test_allows_first_request(self):
        """首次请求在配额充足时应被放行，且 reason 为 "ok"。
        这是正常路径的测试：双桶都满，请求直接通过。"""
        rl = RateLimiter(global_rate=100, global_capacity=10,
                         per_ip_rate=10, per_ip_capacity=10)
        allowed, reason = rl.check("1.2.3.4")
        assert allowed is True
        assert reason == "ok"

    def test_per_ip_blocked(self):
        """同一 IP 连续请求超过其单 IP 配额后被拒绝，且拒绝原因包含 "IP"。
        验证 per-ip 限流生效：全局配额还很充足，但该 IP 自身超限。"""
        rl = RateLimiter(global_rate=100, global_capacity=100,
                         per_ip_rate=1, per_ip_capacity=1)

        rl.check("1.2.3.4")

        allowed, reason = rl.check("1.2.3.4")
        assert allowed is False
        assert "IP" in reason

    def test_global_blocked(self):
        """不同 IP 也受全局桶约束（全局限流是跨 IP 的）。
        场景：全局容量只有 1，第一个 IP 消耗掉后，第二个 IP 再请求就被全局拒绝。
        这验证：全局桶是共享资源，防的是"整个服务"超载。"""
        rl = RateLimiter(global_rate=1, global_capacity=1,
                         per_ip_rate=100, per_ip_capacity=100)
        rl.check("ip-a")
        allowed, reason = rl.check("ip-b")
        assert allowed is False
        assert "全局" in reason

    def test_different_ips_independent(self):
        """不同 IP 各自独立配额，互不挤占（per-ip 桶按 IP 隔离）。
        场景：每个 IP 容量 1，两个不同 IP 各请求一次都应放行——
        验证限流是按 IP 维度分别计数，而不是共享同一份。"""
        rl = RateLimiter(global_rate=100, global_capacity=100,
                         per_ip_rate=1, per_ip_capacity=1)
        assert rl.check("1.1.1.1")[0] is True
        assert rl.check("2.2.2.2")[0] is True

class TestCircuitBreaker:
    """
    熔断器（CircuitBreaker）测试
    ------------------------------------------------------------------------
    熔断器三态：
      - CLOSED（关闭）：正常放行所有请求；每次失败计数，达到阈值后转到 OPEN
      - OPEN（打开）：拒绝所有请求（快速失败），等待 recovery_timeout 秒后
                     允许一次试探请求转入 HALF_OPEN
      - HALF_OPEN（半开）：放行少量试探请求；成功则恢复 CLOSED，失败则回到 OPEN
    """

    def test_closed_allows_all(self):
        """熔断器初始处于 CLOSED 状态，所有请求放行。"""
        cb = CircuitBreaker()
        assert cb.allow_request() is True
        assert cb.get_state() == CircuitBreaker.CLOSED

    def test_open_after_threshold(self):
        """连续失败达到 failure_threshold 后熔断器进入 OPEN，请求被拒绝。
        验证"打开阈值"逻辑：3 次失败 → 熔断。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        for _ in range(3):
            cb.record_failure()
        assert cb.get_state() == CircuitBreaker.OPEN
        assert cb.allow_request() is False

    def test_recovery_enters_half_open(self):
        """OPEN 状态经过 recovery_timeout 冷却时间后( recovery_timeout=0 冷却时间为0)，
          下一次请求会转入 HALF_OPEN。
          验证状态机能从 OPEN 自动尝试恢复HALF_OPEN。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        assert cb.get_state() == CircuitBreaker.OPEN
        assert cb.allow_request() is True  

        assert cb.get_state() == CircuitBreaker.HALF_OPEN

    def test_half_open_success_closes(self):
        """半开状态下，试探请求全部成功（达到 half_open_max_requests）后，
        熔断器恢复 CLOSED，说明下游已恢复，可以全量放行。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0,
                            half_open_max_requests=2)
        cb.record_failure()
        cb.allow_request()
        cb.record_success()
        cb.record_success()
        assert cb.get_state() == CircuitBreaker.CLOSED

    def test_half_open_failure_reopens(self):
        """半开状态下试探请求失败，熔断器立刻回到 OPEN（说明下游仍故障）。
        防止半开后立即放洪。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        cb.allow_request()
        cb.record_failure()
        assert cb.get_state() == CircuitBreaker.OPEN

    def test_half_open_limits_probe_requests(self):
        """半开状态只放行有限(最多half_open_max_requests 次)试探请求，
        超过后拒绝，避免一次性把所有积压请求放进去。"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0,
                            half_open_max_requests=1)
        cb.record_failure()
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        assert cb.allow_request() is False

    def test_success_in_closed_resets_failure_count(self):
        """CLOSED 状态下某次请求成功，应清零失败计数（防止历史失败累积）。
        验证"成功重置失败计数"逻辑：即使之前失败过 2 次，成功后阈值重新从 0 计。"""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.get_state() == CircuitBreaker.CLOSED
