import threading
import time
import enum
from typing import Deque, List, Optional, Tuple, Any, Union
from collections import deque
from dataclasses import dataclass, field


# ----------------------------------------------------------------
# 结果类型与枚举
# ----------------------------------------------------------------
class AcquireFailReason(enum.Enum):
    """acquire 失败原因。"""
    TIMEOUT = "timeout"           # 超时
    CLOSED = "closed"             # 闸门已关闭
    CANCELLED = "cancelled"       # 被 cancel_wait 取消


@dataclass(frozen=True)
class AcquireResult:
    """
    acquire 返回结果。向后兼容 bool：可以直接 `if sem.acquire():`。

    Attributes:
        ok:       是否成功
        reason:   失败原因（ok=True 时为 None）
        permits:  尝试获取的许可数
        waited_ms: 实际等待毫秒数
    """
    ok: bool
    permits: int
    waited_ms: float
    reason: Optional[AcquireFailReason] = None

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        if self.ok:
            return f"AcquireResult(OK, permits={self.permits}, waited={self.waited_ms:.2f}ms)"
        return (f"AcquireResult(FAIL, reason={self.reason.value}, "
                f"permits={self.permits}, waited={self.waited_ms:.2f}ms)")


# ----------------------------------------------------------------
# 内部数据结构
# ----------------------------------------------------------------
@dataclass
class _Waiter:
    """等待节点。"""
    permits: int
    request_id: Optional[str] = None
    name: Optional[str] = None
    cond: threading.Condition = field(default_factory=threading.Condition)
    granted: bool = False
    timed_out: bool = False
    closed_rejected: bool = False
    cancelled: bool = False
    enqueue_ts_ns: int = 0           # 入队时间（用于等待时长）


@dataclass
class _Event:
    """事件流水记录。"""
    ts_ns: int                        # 事件时间（monotonic_ns）
    type: str                         # acquire_success / acquire_fail / release /
                                      # adjust_capacity / set_capacity /
                                      # close / open / cancel_wait
    permits: int = 0
    request_id: Optional[str] = None
    name: Optional[str] = None
    details: str = ""                 # 失败原因 / delta / new_cap 等

    @property
    def ts_ms(self) -> float:
        return self.ts_ns / 1e6


@dataclass
class _Stats:
    """可观测统计。所有读写都在 self._lock 保护下。"""
    total_success: int = 0
    total_timeout: int = 0
    total_closed_rejected: int = 0
    total_cancelled: int = 0
    total_wait_ns: int = 0
    max_wait_ns: int = 0               # 单次最长等待

    @property
    def total_ops(self) -> int:
        return (self.total_success + self.total_timeout
                + self.total_closed_rejected + self.total_cancelled)

    @property
    def avg_wait_ms(self) -> float:
        if self.total_ops == 0:
            return 0.0
        return (self.total_wait_ns / self.total_ops) / 1e6

    @property
    def max_wait_ms(self) -> float:
        return self.max_wait_ns / 1e6


# =================================================================
# 主类
# =================================================================
class TimedSemaphore:
    """
    带超时的信号量（资源闸门）。

    核心能力：
    - 多许可原子获取：acquire(permits=n) 要么全拿要么全不拿
    - 公平/非公平双模式，非公平默认（吞吐高）
    - 严格超时边界：时间一到必失败，绝不"捡漏"
    - 命名请求：acquire(request_id="xxx", name="task-1")，可 cancel_wait
    - 事件流水：环形缓冲区记录最近 N 个关键事件，方便排障
    - 动态容量：set_capacity / adjust_capacity，不强行收回已借出
    - 闸门开关：close() 快速排水，open() 恢复服务
    - 可观测：get_stats() 含最长等待时间，get_event_log() 事件流水

    默认：fair=False（非公平、吞吐更高）。
    """

    DEFAULT_EVENT_LOG_SIZE = 1000

    def __init__(self, value: int = 1, *, fair: bool = False,
                 event_log_size: int = DEFAULT_EVENT_LOG_SIZE):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        if event_log_size < 0:
            raise ValueError("event_log_size must be >= 0")

        self._capacity = value
        self._value = value
        self._outstanding = 0
        self._fair = fair
        self._closed = False
        self._event_log_size = event_log_size
        self._last_fail_reason: Optional[AcquireFailReason] = None
        self._lock = threading.Lock()
        self._wait_queue: Deque[_Waiter] = deque()
        self._stats = _Stats()
        self._event_log: Deque[_Event] = deque(maxlen=event_log_size)

    # ================================================================
    # 属性与观测
    # ================================================================
    @property
    def available(self) -> int:
        with self._lock:
            return self._value

    @property
    def capacity(self) -> int:
        with self._lock:
            return self._capacity

    @property
    def borrowed(self) -> int:
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
        with self._lock:
            return self._closed

    def get_stats(self) -> dict:
        """返回当前统计快照（含最长等待时间）。"""
        with self._lock:
            return {
                "available": self._value,
                "capacity": self._capacity,
                "outstanding": self._outstanding,
                "queue_length": len(self._wait_queue),
                "total_success": self._stats.total_success,
                "total_timeout": self._stats.total_timeout,
                "total_closed_rejected": self._stats.total_closed_rejected,
                "total_cancelled": self._stats.total_cancelled,
                "avg_wait_ms": self._stats.avg_wait_ms,
                "max_wait_ms": self._stats.max_wait_ms,
                "is_fair": self._fair,
                "is_closed": self._closed,
            }

    def inspect_waiters(self) -> List[dict]:
        """
        批量等待视图：每个等待者的 request_id/name/permits/等待时长/是否队首。
        返回的是快照，仅用于观测。
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
                    "request_id": w.request_id,
                    "name": w.name,
                    "permits": w.permits,
                    "waited_ms": round(waited_ms, 3),
                    "is_head": (i == 0),
                    "granted": w.granted,
                    "timed_out": w.timed_out,
                    "closed_rejected": w.closed_rejected,
                    "cancelled": w.cancelled,
                })
            return out

    # ================================================================
    # 事件流水
    # ================================================================
    def _log_event(self, type_: str, permits: int = 0,
                   request_id: Optional[str] = None,
                   name: Optional[str] = None,
                   details: str = "") -> None:
        """打一条事件（必须持锁调用，或者是外部操作在持锁后）。"""
        if self._event_log_size == 0:
            return
        self._event_log.append(_Event(
            ts_ns=time.monotonic_ns(),
            type=type_,
            permits=permits,
            request_id=request_id,
            name=name,
            details=details,
        ))

    def get_event_log(self, limit: Optional[int] = None) -> List[dict]:
        """
        返回事件流水（最新的在前）。

        Args:
            limit: 最多返回多少条（None = 返回全部）
        """
        with self._lock:
            events = list(reversed(self._event_log))
            if limit is not None and limit > 0:
                events = events[:limit]
            return [
                {
                    "ts_ms": round(e.ts_ms, 3),
                    "type": e.type,
                    "permits": e.permits,
                    "request_id": e.request_id,
                    "name": e.name,
                    "details": e.details,
                }
                for e in events
            ]

    def get_event_summary(self) -> dict:
        """返回事件流水摘要（各类事件计数、最近失败原因 Top）。"""
        with self._lock:
            counts: dict = {}
            fail_reasons: dict = {}
            for e in self._event_log:
                counts[e.type] = counts.get(e.type, 0) + 1
                if e.type == "acquire_fail" and e.details:
                    fail_reasons[e.details] = fail_reasons.get(e.details, 0) + 1
            return {
                "total_events": len(self._event_log),
                "buffer_capacity": self._event_log_size,
                "counts": counts,
                "fail_reasons": fail_reasons,
            }

    # ================================================================
    # 闸门开关
    # ================================================================
    def close(self) -> None:
        """关闭闸门：新请求立即失败，等待者全部唤醒，已持有正常归还。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._log_event("close", details="gate closed")

            for waiter in list(self._wait_queue):
                if not waiter.granted and not waiter.timed_out and not waiter.cancelled:
                    waiter.closed_rejected = True
                    with waiter.cond:
                        waiter.cond.notify()
            self._wait_queue.clear()

    def open(self) -> None:
        """重新打开闸门。"""
        with self._lock:
            self._closed = False
            self._log_event("open", details="gate opened")

    # ================================================================
    # 取消等待（按 request_id）
    # ================================================================
    def cancel_wait(self, request_id: str) -> int:
        """
        按 request_id 取消正在等待的请求。
        所有匹配的等待者会被标记 cancelled、唤醒并返回失败（原因=CANCELLED）。
        不影响其他等待者。

        Args:
            request_id: 要取消的请求 ID

        Returns:
            实际取消了几个等待者（0 表示没找到匹配的等待者）
        """
        if not request_id:
            raise ValueError("request_id must be a non-empty string")

        cancelled_count = 0
        with self._lock:
            for waiter in list(self._wait_queue):
                if (waiter.request_id == request_id
                        and not waiter.granted
                        and not waiter.timed_out
                        and not waiter.closed_rejected
                        and not waiter.cancelled):
                    waiter.cancelled = True
                    cancelled_count += 1
                    with waiter.cond:
                        waiter.cond.notify()
            self._log_event(
                "cancel_wait",
                request_id=request_id,
                details=f"cancelled={cancelled_count}",
            )
        return cancelled_count

    # ================================================================
    # 动态容量调整
    # ================================================================
    def adjust_capacity(self, delta: int) -> None:
        """相对调整容量。delta>0 扩容，delta<0 缩容。最终 capacity<0 会抛 ValueError。"""
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
            self._log_event(
                "adjust_capacity",
                permits=abs(delta),
                details=f"old={old_cap}, new={new_cap}, delta={delta:+d}",
            )
            if delta > 0:
                self._value += delta
                if self._fair:
                    self._try_grant_fair()
                else:
                    self._wake_waiters()
            elif delta < 0:
                shrink = min(self._value, -delta)
                self._value -= shrink

    def set_capacity(self, new_capacity: int) -> None:
        """设置绝对容量。new_capacity<0 会抛 ValueError。"""
        if not isinstance(new_capacity, int):
            raise TypeError(f"capacity must be int, got {type(new_capacity).__name__}")
        if new_capacity < 0:
            raise ValueError(
                f"set_capacity(new={new_capacity}) rejected: capacity must be >= 0. "
                f"Note: outstanding borrowed={self._outstanding} will not be force-recalled."
            )
        with self._lock:
            old_cap = self._capacity
            delta = new_capacity - old_cap
        if delta != 0:
            self.adjust_capacity(delta)
        else:
            # 没变化也要打规整日志吗？不用，但要规整 available
            with self._lock:
                if self._value > self._capacity:
                    self._value = self._capacity
            return
        # 额外规整
        with self._lock:
            if self._value > self._capacity:
                self._value = self._capacity
            # 打 set_capacity 事件（adjust 里已经打过 adjust_capacity 事件）
            # 这里补充一个 set_capacity 标记便于流水查看
            self._log_event(
                "set_capacity",
                permits=new_capacity,
                details=f"final_capacity={new_capacity}",
            )

    # ================================================================
    # 获取许可
    # ================================================================
    def acquire(self, permits: int = 1, timeout: Optional[float] = None,
                request_id: Optional[str] = None,
                name: Optional[str] = None) -> AcquireResult:
        """
        原子获取 permits 个许可。

        Args:
            permits:    需要获取的许可数，>=1
            timeout:    超时秒数，None=无限等待
            request_id: 请求 ID，用于 cancel_wait(request_id) 取消排队
            name:       可读名称，显示在等待视图里便于排障

        Returns:
            AcquireResult：可直接当 bool 用，也可以看 .reason 失败原因、.waited_ms 等待时长
        """
        if permits < 1:
            raise ValueError("permits must be >= 1")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0")

        t0_ns = time.monotonic_ns()
        ok = self._do_acquire(permits, timeout, request_id, name)
        wait_ns = time.monotonic_ns() - t0_ns
        wait_ms = wait_ns / 1e6

        # 更新统计 & 事件日志
        with self._lock:
            fail_reason = self._last_fail_reason
            self._last_fail_reason = None

            if ok:
                self._stats.total_success += 1
                self._outstanding += permits
                self._log_event(
                    "acquire_success",
                    permits=permits,
                    request_id=request_id,
                    name=name,
                    details=f"waited={wait_ms:.2f}ms",
                )
            else:
                if fail_reason == AcquireFailReason.TIMEOUT:
                    self._stats.total_timeout += 1
                elif fail_reason == AcquireFailReason.CLOSED:
                    self._stats.total_closed_rejected += 1
                elif fail_reason == AcquireFailReason.CANCELLED:
                    self._stats.total_cancelled += 1
                else:
                    # 兜底，不该走到这里
                    self._stats.total_timeout += 1
                self._log_event(
                    "acquire_fail",
                    permits=permits,
                    request_id=request_id,
                    name=name,
                    details=fail_reason.value if fail_reason else "unknown",
                )

            self._stats.total_wait_ns += wait_ns
            if wait_ns > self._stats.max_wait_ns:
                self._stats.max_wait_ns = wait_ns

        return AcquireResult(
            ok=ok,
            permits=permits,
            waited_ms=wait_ms,
            reason=fail_reason if not ok else None,
        )

    def _do_acquire(self, permits: int, timeout: Optional[float],
                    request_id: Optional[str], name: Optional[str]) -> bool:
        """真实 acquire 逻辑，不处理统计。通过 self._last_fail_reason 返回失败原因。"""
        with self._lock:
            self._last_fail_reason = None
            if self._closed:
                self._last_fail_reason = AcquireFailReason.CLOSED
                return False

        if self._fair:
            return self._acquire_fair(permits, timeout, request_id, name)
        else:
            return self._acquire_unfair(permits, timeout, request_id, name)

    # ----------------------------------------------------------------
    # 非公平模式：小请求可跳过大请求
    # ----------------------------------------------------------------
    def _acquire_unfair(self, permits: int, timeout: Optional[float],
                        request_id: Optional[str], name: Optional[str]) -> bool:
        waiter = _Waiter(
            permits=permits,
            request_id=request_id,
            name=name,
            enqueue_ts_ns=time.monotonic_ns(),
        )

        with self._lock:
            if self._closed:
                self._last_fail_reason = AcquireFailReason.CLOSED
                return False
            if self._value >= permits:
                self._value -= permits
                return True
            self._wait_queue.append(waiter)

        got_granted = False
        timed_out = False
        closed_rej = False
        cancelled = False
        try:
            if timeout is None:
                with waiter.cond:
                    while (not waiter.granted and not waiter.timed_out
                           and not waiter.closed_rejected and not waiter.cancelled):
                        waiter.cond.wait()
            else:
                deadline = time.monotonic() + timeout
                remaining = timeout
                with waiter.cond:
                    while (not waiter.granted and not waiter.timed_out
                           and not waiter.closed_rejected and not waiter.cancelled
                           and remaining > 0):
                        waiter.cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                timed_out = (
                    remaining <= 0 and not waiter.granted
                    and not waiter.closed_rejected and not waiter.cancelled
                )

            closed_rej = waiter.closed_rejected
            cancelled = waiter.cancelled

            if closed_rej:
                with self._lock:
                    self._last_fail_reason = AcquireFailReason.CLOSED
                    self._remove_waiter(waiter)
                return False

            if cancelled:
                with self._lock:
                    self._last_fail_reason = AcquireFailReason.CANCELLED
                    self._remove_waiter(waiter)
                return False

            if timed_out:
                with self._lock:
                    waiter.timed_out = True
                    self._last_fail_reason = AcquireFailReason.TIMEOUT
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    self._remove_waiter(waiter)
                    self._wake_waiters()
                return False

            got_granted = waiter.granted
            return got_granted
        finally:
            if not got_granted and not timed_out and not closed_rej and not cancelled:
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
    def _acquire_fair(self, permits: int, timeout: Optional[float],
                      request_id: Optional[str], name: Optional[str]) -> bool:
        waiter = _Waiter(
            permits=permits,
            request_id=request_id,
            name=name,
            enqueue_ts_ns=time.monotonic_ns(),
        )

        with self._lock:
            if self._closed:
                self._last_fail_reason = AcquireFailReason.CLOSED
                return False
            self._wait_queue.append(waiter)
            self._try_grant_fair()
            if waiter.granted:
                return True

        got_granted = False
        timed_out = False
        closed_rej = False
        cancelled = False
        try:
            if timeout is None:
                with waiter.cond:
                    while (not waiter.granted and not waiter.timed_out
                           and not waiter.closed_rejected and not waiter.cancelled):
                        waiter.cond.wait()
            else:
                deadline = time.monotonic() + timeout
                remaining = timeout
                with waiter.cond:
                    while (not waiter.granted and not waiter.timed_out
                           and not waiter.closed_rejected and not waiter.cancelled
                           and remaining > 0):
                        waiter.cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                timed_out = (
                    remaining <= 0 and not waiter.granted
                    and not waiter.closed_rejected and not waiter.cancelled
                )

            closed_rej = waiter.closed_rejected
            cancelled = waiter.cancelled

            if closed_rej:
                with self._lock:
                    self._last_fail_reason = AcquireFailReason.CLOSED
                    self._remove_waiter(waiter)
                return False

            if cancelled:
                with self._lock:
                    self._last_fail_reason = AcquireFailReason.CANCELLED
                    self._remove_waiter(waiter)
                return False

            if timed_out:
                with self._lock:
                    waiter.timed_out = True
                    self._last_fail_reason = AcquireFailReason.TIMEOUT
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    self._remove_waiter(waiter)
                    self._try_grant_fair()
                return False

            got_granted = waiter.granted
            return got_granted
        finally:
            if not got_granted and not timed_out and not closed_rej and not cancelled:
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
        try:
            self._wait_queue.remove(waiter)
        except ValueError:
            pass

    def _wake_waiters(self) -> None:
        """非公平模式：扫描所有等待者，任意够资源的都授予（小请求绕过）。"""
        if not self._wait_queue:
            return
        progress = True
        while progress:
            progress = False
            for waiter in list(self._wait_queue):
                if (waiter.granted or waiter.timed_out
                        or waiter.closed_rejected or waiter.cancelled):
                    continue
                if self._value >= waiter.permits:
                    self._value -= waiter.permits
                    waiter.granted = True
                    self._wait_queue.remove(waiter)
                    with waiter.cond:
                        waiter.cond.notify()
                    progress = True

    def _try_grant_fair(self) -> None:
        """公平模式：只有队首能被满足。严格 FIFO。"""
        while self._wait_queue and self._value >= self._wait_queue[0].permits:
            head = self._wait_queue[0]
            if head.closed_rejected or head.cancelled:
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
        若 available + permits > capacity（因缩容），超出部分自动被吞掉。
        """
        if permits < 1:
            raise ValueError("permits must be >= 1")

        with self._lock:
            self._value += permits
            self._outstanding = max(0, self._outstanding - permits)
            if self._value > self._capacity:
                self._value = self._capacity
            self._log_event(
                "release",
                permits=permits,
                details=f"available_now={self._value}, capacity={self._capacity}",
            )
            if self._fair:
                self._try_grant_fair()
            else:
                self._wake_waiters()

    # ================================================================
    # 上下文管理器（单许可 & 多许可）
    # ================================================================
    def __enter__(self):
        # 单许可上下文，向后兼容
        res = self.acquire()
        if not res:
            raise RuntimeError(f"Failed to acquire semaphore in __enter__: {res}")
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

    def acquire_multi(self, permits: int, timeout: Optional[float] = None,
                      request_id: Optional[str] = None,
                      name: Optional[str] = None) -> "_MultiCtx":
        """返回多许可上下文管理器。"""
        return _MultiCtx(self, permits, timeout, request_id, name)


class _MultiCtx:
    """acquire_multi 的上下文管理器实现。"""
    __slots__ = ("_sem", "_permits", "_timeout", "_request_id", "_name",
                 "_acquired", "ok", "result")

    def __init__(self, sem: TimedSemaphore, permits: int,
                 timeout: Optional[float],
                 request_id: Optional[str], name: Optional[str]):
        self._sem = sem
        self._permits = permits
        self._timeout = timeout
        self._request_id = request_id
        self._name = name
        self._acquired = False
        self.ok = False
        self.result: Optional[AcquireResult] = None

    def __enter__(self) -> bool:
        self.result = self._sem.acquire(
            permits=self._permits, timeout=self._timeout,
            request_id=self._request_id, name=self._name,
        )
        self.ok = bool(self.result)
        self._acquired = self.ok
        return self.ok

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            self._sem.release(self._permits)
            self._acquired = False
        return False
