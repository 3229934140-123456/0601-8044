import threading
import time
from typing import Deque, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Waiter:
    """等待节点——公平和非公平模式都用这个结构。"""
    permits: int
    cond: threading.Condition = field(default_factory=threading.Condition)
    granted: bool = False
    timed_out: bool = False
    closed_rejected: bool = False   # 被闸门关闭拒绝
    enqueue_ts_ns: int = 0           # 入队时间（monotonic_ns），用于 inspect 等待时长


@dataclass
class _Stats:
    """可观测统计。所有读写都在 self._lock 保护下。"""
    total_success: int = 0
    total_timeout: int = 0
    total_closed_rejected: int = 0   # 闸门关闭被拒绝数
    total_wait_ns: int = 0           # 纳秒累计，计算平均值时再转 ms

    @property
    def total_ops(self) -> int:
        return self.total_success + self.total_timeout + self.total_closed_rejected

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
    - 可观测统计：累计成功/超时/关闭拒绝/平均等待时间
    - 等待视图：inspect_waiters() 看每个等待者所需许可、已等待时长、是否队首
    - 闸门开关：close()/open() 服务下线/恢复

    默认：fair=False（非公平、吞吐更高，类似 Java Semaphore 默认）。
    """

    def __init__(self, value: int = 1, *, fair: bool = False):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._capacity = value
        self._value = value
        self._outstanding = 0          # 实际已借出的许可数（acquire-release 净额）
        self._fair = fair
        self._closed = False            # 闸门是否关闭（用于服务下线）
        self._last_closed_rejected = False  # acquire 统计用标志位
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

    @property
    def is_closed(self) -> bool:
        """闸门是否已关闭（新请求将被立即拒绝）。"""
        with self._lock:
            return self._closed

    def get_stats(self) -> dict:
        """返回当前统计快照。"""
        with self._lock:
            return {
                "available": self._value,
                "capacity": self._capacity,
                "outstanding": self._outstanding,
                "queue_length": len(self._wait_queue),
                "total_success": self._stats.total_success,
                "total_timeout": self._stats.total_timeout,
                "total_closed_rejected": self._stats.total_closed_rejected,
                "avg_wait_ms": self._stats.avg_wait_ms,
                "is_fair": self._fair,
                "is_closed": self._closed,
            }

    def inspect_waiters(self) -> List[dict]:
        """
        批量异步等待视图——返回当前所有等待请求的快照。
        用于排障：判断是不是大请求堵住了资源池。

        每个元素包含：
          - permits:         请求的许可数
          - waited_ms:       已经等了多久（毫秒）
          - is_head:         是否公平队列的队首（仅 fair=True 时有意义）
          - granted:         是否已被授予但还没出 acquire
          - timed_out:       是否已标记超时
          - closed_rejected: 是否已被闸门关闭标记为拒绝

        注意：返回的是**快照**，与实际状态有时间差，只用于观测，不做判断依据。
        """
        now_ns = time.monotonic_ns()
        with self._lock:
            q_copy = list(self._wait_queue)
            out: List[dict] = []
            for i, w in enumerate(q_copy):
                waited_ms = 0.0
                if w.enqueue_ts_ns > 0:
                    waited_ms = (now_ns - w.enqueue_ts_ns) / 1e6
                out.append({
                    "permits": w.permits,
                    "waited_ms": round(waited_ms, 3),
                    "is_head": (i == 0),
                    "granted": w.granted,
                    "timed_out": w.timed_out,
                    "closed_rejected": w.closed_rejected,
                })
            return out

    # ================================================================
    # 闸门开关（服务下线/恢复）
    # ================================================================
    def close(self) -> None:
        """
        关闭闸门：
          1. 之后所有新的 acquire() 请求立即返回 False（不等待）
          2. 已在等待队列中的请求被全部唤醒，返回 False
          3. 已成功获取许可的任务仍然可以正常 release() 归还

        用于服务准备下线时快速"排水"。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            # 唤醒所有等待者——标记为"关闭拒绝"
            for waiter in list(self._wait_queue):
                if not waiter.granted and not waiter.timed_out:
                    waiter.closed_rejected = True
                    with waiter.cond:
                        waiter.cond.notify()
            # 清空等待队列（等各线程自己 remove 也可以，但清空更干净）
            self._wait_queue.clear()

    def open(self) -> None:
        """重新打开闸门（恢复服务）。关闭期间被拒绝的请求不会自动重试。"""
        with self._lock:
            self._closed = False

    # ================================================================
    # 动态容量调整
    # ================================================================
    def adjust_capacity(self, delta: int) -> None:
        """
        相对调整容量：delta>0 扩容，delta<0 缩容。
        已借出的许可不会被强行收回。
        - 扩容：立即增加 delta 个可用许可，并尝试唤醒等待者
        - 缩容：尽量从 available 里扣；后续 release 时若 available 超过 capacity，
          会自动把超出部分吞掉（确保对齐新容量）

        异常：如果最终 capacity < 0，抛出 ValueError，调整被回滚。
        """
        with self._lock:
            old_cap = self._capacity
            new_cap = old_cap + delta
            if new_cap < 0:
                raise ValueError(
                    f"adjust_capacity(delta={delta}) would result in capacity={new_cap} < 0, "
                    f"current borrowed={self._outstanding}. "
                    f"Rejecting to avoid negative capacity."
                )
            self._capacity = new_cap
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
        """
        设置绝对容量。如果 new_capacity < 0 立即抛出 ValueError。
        缩容时已借出的许可不会被强行收回，但后续归还的许可若超出新 capacity 会被自动吞掉。
        """
        if not isinstance(new_capacity, int):
            raise TypeError(f"capacity must be int, got {type(new_capacity).__name__}")
        if new_capacity < 0:
            raise ValueError(
                f"set_capacity(new={new_capacity}) rejected: capacity must be >= 0. "
                f"Note: outstanding borrowed={self._outstanding} will not be force-recalled."
            )
        with self._lock:
            delta = new_capacity - self._capacity
        if delta != 0:
            self.adjust_capacity(delta)
        # 额外规整：缩容后若 available 超过新 capacity，把超出部分一次性吞掉
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

        闸门关闭语义：
          如果 close() 过，立即返回 False（不计入超时统计，计入 closed_rejected）。

        Args:
            permits: 需要获取的许可数量，必须 >= 1
            timeout: 超时秒数。None 表示无限等待。

        Returns:
            True 表示成功获取 permits 个；False 表示失败（超时 / 关闭 / 参数错等）。
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
                # 区分是超时还是关闭拒绝：通过 "waiting for < 0.1ms" 这种启发式无法判断，
                # 所以在 _do_acquire 返回后用一个标志位
                if self._last_closed_rejected:
                    self._stats.total_closed_rejected += 1
                else:
                    self._stats.total_timeout += 1
            self._stats.total_wait_ns += wait_ns
            self._last_closed_rejected = False

        return ok

    def _do_acquire(self, permits: int, timeout: Optional[float]) -> bool:
        """真实的 acquire 逻辑，不处理统计。"""
        # 先在持锁下检查关闭——新请求直接拒绝
        with self._lock:
            self._last_closed_rejected = False
            if self._closed:
                self._last_closed_rejected = True
                return False

        if self._fair:
            return self._acquire_fair(permits, timeout)
        else:
            return self._acquire_unfair(permits, timeout)

    # ----------------------------------------------------------------
    # 非公平模式：小请求可跳过大请求
    # ----------------------------------------------------------------
    def _acquire_unfair(self, permits: int, timeout: Optional[float]) -> bool:
        waiter = _Waiter(permits=permits, enqueue_ts_ns=time.monotonic_ns())

        with self._lock:
            if self._closed:
                self._last_closed_rejected = True
                return False
            # 快速路径：资源够，直接拿
            if self._value >= permits:
                self._value -= permits
                return True

            # 资源不够，登记到等待列表
            self._wait_queue.append(waiter)

        # 进入等待
        got_granted = False
        timed_out = False
        closed_rej = False
        try:
            if timeout is None:
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out and not waiter.closed_rejected:
                        waiter.cond.wait()
                timed_out = False
            else:
                deadline = time.monotonic() + timeout
                remaining = timeout
                with waiter.cond:
                    while (not waiter.granted and not waiter.timed_out
                           and not waiter.closed_rejected and remaining > 0):
                        waiter.cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                timed_out = remaining <= 0 and not waiter.granted and not waiter.closed_rejected

            closed_rej = waiter.closed_rejected

            # 关闭拒绝：不归还，因为从未被授予
            if closed_rej:
                with self._lock:
                    self._last_closed_rejected = True
                    self._remove_waiter(waiter)
                return False

            # 严格超时边界
            if timed_out:
                with self._lock:
                    waiter.timed_out = True
                    # 临界：刚好被授予但要超时放弃——归还
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    self._remove_waiter(waiter)
                    self._wake_waiters()
                return False

            got_granted = waiter.granted
            return got_granted
        finally:
            if not got_granted and not timed_out and not closed_rej:
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
        waiter = _Waiter(permits=permits, enqueue_ts_ns=time.monotonic_ns())

        with self._lock:
            if self._closed:
                self._last_closed_rejected = True
                return False
            self._wait_queue.append(waiter)
            # 尝试立刻满足（只有队首才有资格）
            self._try_grant_fair()
            if waiter.granted:
                return True

        got_granted = False
        timed_out = False
        closed_rej = False
        try:
            if timeout is None:
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out and not waiter.closed_rejected:
                        waiter.cond.wait()
                timed_out = False
            else:
                deadline = time.monotonic() + timeout
                remaining = timeout
                with waiter.cond:
                    while (not waiter.granted and not waiter.timed_out
                           and not waiter.closed_rejected and remaining > 0):
                        waiter.cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                timed_out = remaining <= 0 and not waiter.granted and not waiter.closed_rejected

            closed_rej = waiter.closed_rejected

            if closed_rej:
                with self._lock:
                    self._last_closed_rejected = True
                    self._remove_waiter(waiter)
                return False

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
            if not got_granted and not timed_out and not closed_rej:
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

        progress = True
        while progress:
            progress = False
            for waiter in list(self._wait_queue):
                if waiter.granted or waiter.timed_out or waiter.closed_rejected:
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
            if head.closed_rejected:
                self._wait_queue.popleft()
                continue
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

        容量对齐：若当前 available + permits > capacity（因之前缩容后有未归还的借出），
        则归还后自动把超出部分吞掉，确保 available 不会超过 capacity。
        """
        if permits < 1:
            raise ValueError("permits must be >= 1")

        with self._lock:
            self._value += permits
            self._outstanding = max(0, self._outstanding - permits)
            # ---- 需求2a：缩容后归还自动对齐，不要超过 capacity ----
            if self._value > self._capacity:
                self._value = self._capacity
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
