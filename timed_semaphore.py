import threading
import time
from typing import Deque
from collections import deque


class TimedSemaphore:
    """
    带超时的信号量（资源闸门）。

    核心设计：
    1. 使用 Condition.wait(timeout) 进行非忙等的超时等待，线程在等待期间被操作系统挂起，不占用 CPU。
    2. 使用 waiters 计数器精确追踪"仍在队列中等待"的线程数，避免 release 时因等待者刚超时而导致许可丢失。
    """

    def __init__(self, value: int = 1):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._value = value
        self._waiters: int = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @property
    def available(self) -> int:
        with self._lock:
            return self._value

    @property
    def waiters(self) -> int:
        with self._lock:
            return self._waiters

    def acquire(self, timeout: float | None = None) -> bool:
        """
        获取一个许可。

        Args:
            timeout: 超时秒数。None 表示无限等待（与普通信号量一致）。

        Returns:
            True 表示成功获取，False 表示超时失败。
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0")

        with self._cond:
            # 快速路径：有空闲许可，直接拿走
            if self._value > 0:
                self._value -= 1
                return True

            # 慢速路径：没有许可，登记为等待者，进入超时等待
            self._waiters += 1
            try:
                if timeout is None:
                    # 无限等待，直到被 notify
                    while self._value <= 0:
                        self._cond.wait()
                    self._value -= 1
                    return True
                else:
                    deadline = time.monotonic() + timeout
                    remaining = timeout
                    while self._value <= 0 and remaining > 0:
                        self._cond.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                    # 醒来后必须再次检查：是拿到了许可，还是真的超时了
                    if self._value > 0:
                        self._value -= 1
                        return True
                    else:
                        # 真正超时了，没拿到许可
                        return False
            finally:
                self._waiters -= 1

    def release(self, n: int = 1) -> None:
        """
        释放 n 个许可。

        关键机制：
        1. 许可先显式归还到 _value 池子，**绝不**把许可"隐含地"寄托在某个 notify 上。
           这样即使被 notify 的线程刚好超时离开，许可仍在池子里，不会丢失。
        2. 只唤醒最多 min(n, waiters) 个等待者，让他们自己去抢池子里的许可。
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        with self._cond:
            self._value += n
            to_wake = min(n, self._waiters)
            for _ in range(to_wake):
                self._cond.notify()

    __enter__ = acquire

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
