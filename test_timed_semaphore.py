import threading
import time
from timed_semaphore import TimedSemaphore


def test_basic():
    print("=== test_basic ===")
    s = TimedSemaphore(2)
    assert s.acquire() is True
    assert s.acquire() is True
    assert s.available == 0
    s.release()
    assert s.available == 1
    s.release()
    assert s.available == 2
    print("PASS")


def test_timeout():
    print("=== test_timeout ===")
    s = TimedSemaphore(1)
    assert s.acquire() is True

    t0 = time.monotonic()
    result = s.acquire(timeout=0.2)
    elapsed = time.monotonic() - t0
    assert result is False, "should timeout"
    assert 0.18 <= elapsed <= 0.3, f"elapsed {elapsed} not near 0.2s"
    print("PASS (timeout works, no busy wait)")


def test_no_leak_when_release_and_timeout_concurrent():
    """
    关键测试：多个线程在极短超时后失败，同时 release 发生。
    验证最终 available 等于初始值（无泄漏），并且后续线程能拿到许可。
    """
    print("=== test_no_leak_when_release_and_timeout_concurrent ===")
    INITIAL = 3
    s = TimedSemaphore(INITIAL)

    # 先把所有许可拿走
    holders = []
    for _ in range(INITIAL):
        assert s.acquire() is True
        holders.append(True)
    assert s.available == 0

    # 启动 10 个线程，每个只等 20ms，几乎一定会超时
    results = [None] * 10
    start_barrier = threading.Barrier(10)

    def worker(i):
        start_barrier.wait()
        results[i] = s.acquire(timeout=0.02)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()

    # 让所有等待线程同时开始等，同时我们释放所有许可
    # 这里制造"等待者刚超时而 release 刚发生"的竞态窗口
    time.sleep(0.01)
    for _ in range(INITIAL):
        s.release()

    for t in threads:
        t.join()

    success_count = sum(1 for r in results if r is True)
    timeout_count = sum(1 for r in results if r is False)
    print(f"  success={success_count}, timeout={timeout_count}, available={s.available}")

    # 核心断言：总许可数守恒——成功拿到的 + 池子里剩余的 = 初始释放的 3 个
    assert success_count + s.available == INITIAL, (
        f"资源泄漏！成功{success_count} + 剩余{s.available} != {INITIAL}"
    )

    # 再验证：现在池子有空闲，新的请求一定能拿到
    for _ in range(s.available):
        assert s.acquire(timeout=0.1) is True
    assert s.available == 0
    print("PASS (no resource leak)")


def test_context_manager():
    print("=== test_context_manager ===")
    s = TimedSemaphore(1)
    with s:
        assert s.available == 0
    assert s.available == 1
    print("PASS")


def test_infinite_wait_wakes_up():
    print("=== test_infinite_wait_wakes_up ===")
    s = TimedSemaphore(0)
    acquired = threading.Event()

    def waiter():
        s.acquire()
        acquired.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)
    assert not acquired.is_set()
    s.release()
    acquired.wait(timeout=1)
    assert acquired.is_set()
    print("PASS")


if __name__ == "__main__":
    test_basic()
    test_timeout()
    for _ in range(20):  # 多跑几次增加捕获竞态的概率
        test_no_leak_when_release_and_timeout_concurrent()
    test_context_manager()
    test_infinite_wait_wakes_up()
    print("\nAll tests passed.")
