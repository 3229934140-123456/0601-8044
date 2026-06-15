import threading
import time
from typing import Deque, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Waiter:
    """等待节点——公平和非公平模式都用这个结构。"""
    permits: int
    cond: threading.Condition = field(default_factory=threading.Condition)
    granted: bool = False
    timed_out: bool = False


@dataclass
class _Stats:
    """可观测统计。所有读写都在 self._lock 保护下。"""
    total_success: int = 0
    total_timeout: int = 0
    total_wait_ns: int = 0  # 纳秒累计，计算平均值时再转 ms

    @property
    def total_ops(self) -> int:
        return self.total_success + self.total_timeout

    @property
    def avg_wait_ms(self) -> float:
        if self.total_ops == 0:
            return 0.0
        return (self.total_wait_ns / self.total_ops) / 1e6


class TimedSemaphore:
    """
    带超时的信号量（资源闸门）。

    核心能力：
    - 多许可原子获取：acquire(permits=n) 要么全拿要么全不拿
    - 公平排队模式：fair=True 时按进入顺序发放，避免饥饿
    - 严格超时边界：时间一到必失败，绝不"捡漏"
    - 非公平小请求绕过：大请求暂时凑不齐时，后面够资源的小请求先通过
    - 动态容量调整：set_capacity / adjust_capacity，不强行收回已借出许可
    - 可观测统计：累计成功/超时/平均等待时间

    默认：fair=False（非公平、吞吐更高，类似 Java Semaphore 默认）。
    """

    def __init__(self, value: int = 1, *, fair: bool = False):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._capacity = value
        self._value = value
        self._outstanding = 0          # 实际已借出的许可数（acquire-release 净额）
        self._fair = fair
        self._lock = threading.Lock()
        self._wait_queue: Deque[_Waiter] = deque()
        self._stats = _Stats()

    # ================================================================
    # 属性与观测
    # ================================================================
    @property
    def available(self) -> int:
        with self._lock:
            return self._value

    @property
    def capacity(self) -> int:
        """设计容量（总许可数上限，不考虑当前已借出的）。"""
        with self._lock:
            return self._capacity

    @property
    def borrowed(self) -> int:
        """当前实际已借出的许可数（不受 capacity 调整影响）。"""
        with self._lock:
            return self._outstanding

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._wait_queue)

    @property
    def is_fair(self) -> bool:
        return self._fair

    def get_stats(self) -> dict:
        """返回当前统计快照。"""
        with self._lock:
            return {
                "available": self._value,
                "capacity": self._capacity,
                "queue_length": len(self._wait_queue),
                "total_success": self._stats.total_success,
                "total_timeout": self._stats.total_timeout,
                "avg_wait_ms": self._stats.avg_wait_ms,
            }

    # ================================================================
    # 动态容量调整
    # ================================================================
    def adjust_capacity(self, delta: int) -> None:
        """
        相对调整容量：delta>0 扩容，delta<0 缩容。
        已借出的许可不会被强行收回。
        - 扩容：立即增加 delta 个可用许可，并尝试唤醒等待者
        - 缩容：尽量从 available 里扣；不够扣的先把 capacity 标小，后续 release 回来会在缩容时再扣
        """
        with self._lock:
            self._capacity += delta
            if delta > 0:
                self._value += delta
                if self._fair:
                    self._try_grant_fair()
                else:
                    self._wake_waiters()
            elif delta < 0:
                # 缩容：立即吞掉 available 中多余的
                shrink = min(self._value, -delta)
                self._value -= shrink
                # 不需要唤醒等待者——资源更少了

    def set_capacity(self, new_capacity: int) -> None:
        """设置绝对容量。"""
        if new_capacity < 0:
            raise ValueError("capacity must be >= 0")
        with self._lock:
            delta = new_capacity - self._capacity
        if delta != 0:
            self.adjust_capacity(delta)
        # 额外：缩容后若 available 超过新 capacity，把超出部分一次性吞掉
        #（防止 release 累积的多余许可）
        if new_capacity < 0:
            pass
        with self._lock:
            if self._value > self._capacity:
                self._value = self._capacity

    # ================================================================
    # 获取许可
    # ================================================================
    def acquire(self, permits: int = 1, timeout: Optional[float] = None) -> bool:
        """
        原子获取 permits 个许可。

        严格超时语义：
          一旦超时时间到，即使此刻刚好有许可释放，也返回 False。
          不会因为"刚好赶上末班车"而把一个已经超时的请求算成成功。

        Args:
            permits: 需要获取的许可数量，必须 >= 1
            timeout: 超时秒数。None 表示无限等待。

        Returns:
            True 表示成功获取 permits 个；False 表示超时，一个都没拿到。
        """
        if permits < 1:
            raise ValueError("permits must be >= 1")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0")

        t0_ns = time.monotonic_ns()
        ok = self._do_acquire(permits, timeout)
        wait_ns = time.monotonic_ns() - t0_ns

        # 更新统计和借出计数
        with self._lock:
            if ok:
                self._stats.total_success += 1
                self._outstanding += permits
            else:
                self._stats.total_timeout += 1
            self._stats.total_wait_ns += wait_ns

        return ok

    def _do_acquire(self, permits: int, timeout: Optional[float]) -> bool:
        """真实的 acquire 逻辑，不处理统计。"""
        if self._fair:
            return self._acquire_fair(permits, timeout)
        else:
            return self._acquire_unfair(permits, timeout)

    # ----------------------------------------------------------------
    # 非公平模式：小请求可跳过大请求
    # ----------------------------------------------------------------
    def _acquire_unfair(self, permits: int, timeout: Optional[float]) -> bool:
        waiter = _Waiter(permits=permits)

        with self._lock:
            # 快速路径：资源够，直接拿
            if self._value >= permits:
                self._value -= permits
                return True

            # 资源不够，登记到等待列表（非公平：顺序无关）
            self._wait_queue.append(waiter)

            # 顺便尝试：看看现有资源能不能先满足某个小请求（可能就是新入队的这个）
            # 不过因为我们刚确认 _value < permits，而其他等待者也在队列里，
            # 这次先不尝试，等 release 或超时再统一扫描。

        # 进入等待
        got_granted = False
        timed_out = False
        try:
            if timeout is None:
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out:
                        waiter.cond.wait()
                timed_out = False
            else:
                deadline = time.monotonic() + timeout
                remaining = timeout
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out and remaining > 0:
                        waiter.cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                timed_out = remaining <= 0

            # 严格超时边界
            if timed_out:
                with self._lock:
                    waiter.timed_out = True
                    # 临界：刚好被授予但要超时放弃——归还
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    self._remove_waiter(waiter)
                    # 腾出了资源/位置，扫描剩下的等待者看谁能满足
                    self._wake_waiters()
                return False

            got_granted = waiter.granted
            return got_granted
        finally:
            if not got_granted and not timed_out:
                # 异常路径
                with self._lock:
                    self._remove_waiter(waiter)
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    self._wake_waiters()

    # ----------------------------------------------------------------
    # 公平模式：FIFO，队首不满足就绝不满足后面的
    # ----------------------------------------------------------------
    def _acquire_fair(self, permits: int, timeout: Optional[float]) -> bool:
        waiter = _Waiter(permits=permits)

        with self._lock:
            self._wait_queue.append(waiter)
            # 尝试立刻满足（只有队首才有资格）
            self._try_grant_fair()
            if waiter.granted:
                return True

        got_granted = False
        timed_out = False
        try:
            if timeout is None:
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out:
                        waiter.cond.wait()
                timed_out = False
            else:
                deadline = time.monotonic() + timeout
                remaining = timeout
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out and remaining > 0:
                        waiter.cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                timed_out = remaining <= 0

            if timed_out:
                with self._lock:
                    waiter.timed_out = True
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    self._remove_waiter(waiter)
                    self._try_grant_fair()
                return False

            got_granted = waiter.granted
            return got_granted
        finally:
            if not got_granted and not timed_out:
                with self._lock:
                    self._remove_waiter(waiter)
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    self._try_grant_fair()

    # ================================================================
    # 内部：唤醒策略
    # ================================================================
    def _remove_waiter(self, waiter: _Waiter) -> None:
        """从等待队列移除。必须在持锁下调用。"""
        try:
            self._wait_queue.remove(waiter)
        except ValueError:
            pass

    def _wake_waiters(self) -> None:
        """
        非公平模式：扫描队列，**任意**满足条件的等待者都可以被唤醒。
        这就是"小请求绕过"机制——哪怕大请求排在前面，小请求只要够资源就先走。
        必须在持锁下调用。
        """
        if not self._wait_queue:
            return

        # 扫描列表多次（一次扫描可能有新空间腾出来）
        progress = True
        while progress:
            progress = False
            for waiter in list(self._wait_queue):  # 遍历副本，允许中途 remove
                if waiter.granted or waiter.timed_out:
                    continue
                if self._value >= waiter.permits:
                    self._value -= waiter.permits
                    waiter.granted = True
                    self._wait_queue.remove(waiter)
                    with waiter.cond:
                        waiter.cond.notify()
                    progress = True

    def _try_grant_fair(self) -> None:
        """
        公平模式：只有队首能被满足。严格 FIFO，大请求卡队首则后面都等。
        必须在持锁下调用。
        """
        while self._wait_queue and self._value >= self._wait_queue[0].permits:
            head = self._wait_queue[0]
            self._value -= head.permits
            head.granted = True
            self._wait_queue.popleft()
            with head.cond:
                head.cond.notify()

    # ================================================================
    # 释放许可
    # ================================================================
    def release(self, permits: int = 1) -> None:
        """
        释放 permits 个许可。

        注意：release 永远不会吞掉许可（避免 TimedSemaphore(0) 后 release(1) 发现许可消失的反直觉行为）。
        "吞掉超额许可"只在缩容操作 (set_capacity/adjust_capacity) 的瞬间发生。
        """
        if permits < 1:
            raise ValueError("permits must be >= 1")

        with self._lock:
            self._value += permits
            self._outstanding = max(0, self._outstanding - permits)
            if self._fair:
                self._try_grant_fair()
            else:
                self._wake_waiters()

    # ================================================================
    # 上下文管理器（单许可 & 多许可）
    # ================================================================
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Failed to acquire semaphore in __enter__")
        self._ctx_permits_held = 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        n = getattr(self, "_ctx_permits_held", 1)
        self.release(n)
        try:
            delattr(self, "_ctx_permits_held")
        except AttributeError:
            pass
        return False

    def acquire_multi(self, permits: int, timeout: Optional[float] = None) -> "_MultiCtx":
        """
        返回一个上下文管理器，进入时获取 permits 个许可，退出时归还 permits 个。
        异常退出也能正确归还。

        用法：
            with sem.acquire_multi(3, timeout=1.0) as ok:
                if ok:
                    do_work()
                else:
                    handle_timeout()
        """
        return _MultiCtx(self, permits, timeout)


class _MultiCtx:
    """acquire_multi 的上下文管理器实现。"""
    __slots__ = ("_sem", "_permits", "_timeout", "_acquired", "ok")

    def __init__(self, sem: TimedSemaphore, permits: int, timeout: Optional[float]):
        self._sem = sem
        self._permits = permits
        self._timeout = timeout
        self._acquired = False
        self.ok = False

    def __enter__(self) -> bool:
        self.ok = self._sem.acquire(permits=self._permits, timeout=self._timeout)
        self._acquired = self.ok
        return self.ok

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            self._sem.release(self._permits)
            self._acquired = False
        return False
