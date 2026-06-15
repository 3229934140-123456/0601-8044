import threading
import time
from typing import Deque, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Waiter:
    """公平模式下的排队节点，每个等待线程一个。"""
    permits: int
    cond: threading.Condition = field(default_factory=threading.Condition)
    granted: bool = False
    timed_out: bool = False


class TimedSemaphore:
    """
    带超时的信号量（资源闸门）。

    新增能力：
    - 多许可原子获取：acquire(permits=n) 要么全拿到要么全失败
    - 公平排队模式：fair=True 时按进入顺序发放许可，避免饥饿
    - 严格超时边界：超时一到立刻失败，即使刚好有释放也不"捡漏"

    默认行为：fair=False（非公平、吞吐更高，类似 Java Semaphore 默认）。
    """

    def __init__(self, value: int = 1, *, fair: bool = False):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._value = value
        self._fair = fair
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # 公平模式用的 FIFO 队列
        self._wait_queue: Deque[_Waiter] = deque()
        # 非公平模式下的等待者计数（仅用于唤醒数优化）
        self._unfair_waiters: int = 0

    @property
    def available(self) -> int:
        with self._lock:
            return self._value

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._wait_queue) if self._fair else self._unfair_waiters

    @property
    def is_fair(self) -> bool:
        return self._fair

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

        if self._fair:
            return self._acquire_fair(permits, timeout)
        else:
            return self._acquire_unfair(permits, timeout)

    # ------------------------------------------------------------------
    # 非公平模式（默认）：吞吐更高，允许插队
    # ------------------------------------------------------------------
    def _acquire_unfair(self, permits: int, timeout: Optional[float]) -> bool:
        with self._cond:
            # 快速路径
            if self._value >= permits:
                self._value -= permits
                return True

            self._unfair_waiters += 1
            try:
                if timeout is None:
                    while self._value < permits:
                        self._cond.wait()
                    self._value -= permits
                    return True
                else:
                    deadline = time.monotonic() + timeout
                    remaining = timeout
                    # 循环条件先看 remaining（严格超时），再看资源够不够
                    while remaining > 0 and self._value < permits:
                        self._cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                    # 严格超时边界：先判断是否超时，再判断是否有资源
                    if remaining <= 0:
                        return False
                    # 到这里说明 remaining > 0，再检查一次资源（防止虚假唤醒）
                    if self._value >= permits:
                        self._value -= permits
                        return True
                    return False
            finally:
                self._unfair_waiters -= 1
                # 自己走了但可能还剩资源，把剩下的人叫醒让他们判断
                if self._value > 0 and self._unfair_waiters > 0:
                    self._cond.notify()

    # ------------------------------------------------------------------
    # 公平模式：FIFO 排队，绝不插队
    # ------------------------------------------------------------------
    def _acquire_fair(self, permits: int, timeout: Optional[float]) -> bool:
        # 先登记到队列尾部（在锁外创建 cond，减少持锁时间）
        waiter = _Waiter(permits=permits)

        with self._lock:
            self._wait_queue.append(waiter)
            # 尝试立刻满足（只有队首才有资格被满足，避免插队）
            self._try_grant_fair()
            if waiter.granted:
                # _try_grant_fair 已经 popleft 了 waiter，直接返回即可
                return True

        # 不在队首或资源不够，进入带超时的等待
        got_granted = False
        timed_out = False
        try:
            if timeout is None:
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out:
                        waiter.cond.wait()
                # 无限等待不会超时
                timed_out = False
            else:
                deadline = time.monotonic() + timeout
                remaining = timeout
                with waiter.cond:
                    while not waiter.granted and not waiter.timed_out and remaining > 0:
                        waiter.cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                # 先判断是否超时——这是严格超时边界的核心
                timed_out = remaining <= 0

            # ---- 严格超时边界：超时了就放弃，即使刚好被授予 ----
            if timed_out:
                with self._lock:
                    waiter.timed_out = True
                    # 临界情况：刚好在超时的瞬间被 _try_grant_fair 授予了许可
                    # 必须把许可归还，不能占着
                    if waiter.granted:
                        self._value += waiter.permits
                        waiter.granted = False
                    if waiter in self._wait_queue:
                        self._wait_queue.remove(waiter)
                    # 自己走了，可能队首现在可以被满足了
                    self._try_grant_fair()
                return False

            # 没超时，看是否被授予
            got_granted = waiter.granted
            return got_granted
        finally:
            if not got_granted and not timed_out:
                # 异常退出路径，确保队列干净
                with self._lock:
                    if waiter in self._wait_queue:
                        self._wait_queue.remove(waiter)
                        if waiter.granted:
                            self._value += waiter.permits
                            waiter.granted = False
                    self._try_grant_fair()

    def _try_grant_fair(self) -> None:
        """
        公平模式下按队首顺序尝试发放许可。
        必须在持有 self._lock 的情况下调用。
        """
        while self._wait_queue and self._value >= self._wait_queue[0].permits:
            head = self._wait_queue[0]
            self._value -= head.permits
            head.granted = True
            self._wait_queue.popleft()
            with head.cond:
                head.cond.notify()

    # ------------------------------------------------------------------
    # 释放
    # ------------------------------------------------------------------
    def release(self, permits: int = 1) -> None:
        """
        释放 permits 个许可。

        许可永远先显式归还到池子，绝不依赖 notify 隐含传递。
        """
        if permits < 1:
            raise ValueError("permits must be >= 1")

        with self._lock:
            self._value += permits
            if self._fair:
                self._try_grant_fair()
            else:
                # 非公平模式：唤醒 min(permits, waiters) 个，让他们自己抢
                to_wake = min(permits, self._unfair_waiters)
                for _ in range(to_wake):
                    self._cond.notify()

    # ------------------------------------------------------------------
    # 上下文管理器（仅单许可，与普通 Semaphore 行为一致）
    # ------------------------------------------------------------------
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Failed to acquire semaphore in __enter__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
