import threading
import time
import random
from timed_semaphore import TimedSemaphore


# ============================================================
# 基础测试
# ============================================================
def test_basic_unfair():
    print("=== test_basic_unfair ===")
    s = TimedSemaphore(2, fair=False)
    assert s.is_fair is False
    assert s.acquire() is True
    assert s.acquire() is True
    assert s.available == 0
    s.release()
    assert s.available == 1
    s.release()
    assert s.available == 2
    print("PASS")


def test_basic_fair():
    print("=== test_basic_fair ===")
    s = TimedSemaphore(2, fair=True)
    assert s.is_fair is True
    assert s.acquire() is True
    assert s.acquire() is True
    assert s.available == 0
    s.release()
    assert s.available == 1
    s.release()
    assert s.available == 2
    print("PASS")


# ============================================================
# 测试1：严格超时边界
# ============================================================
def test_strict_timeout_boundary():
    """
    关键测试：等待时间一到就失败，即使同一时刻有释放，也不算成功。
    同时验证实际等待时长符合预期。
    """
    print("=== test_strict_timeout_boundary ===")
    s = TimedSemaphore(0)
    s.release(1)  # 先放一个许可
    assert s.acquire() is True  # 先拿走，让池子为空
    assert s.available == 0

    TIMEOUT = 0.1  # 100ms 超时

    results = []
    wait_durations = []

    def waiter():
        t0 = time.monotonic()
        ok = s.acquire(timeout=TIMEOUT)
        dur = time.monotonic() - t0
        results.append((ok, dur))
        wait_durations.append(dur)

    # 启动等待者，100ms 后超时
    t = threading.Thread(target=waiter)
    t.start()

    # 恰好等 100ms 再释放——这时候等待者应该已经超时返回 False 了
    time.sleep(TIMEOUT + 0.005)
    s.release()

    t.join()

    ok, dur = results[0]
    assert ok is False, f"超时后即使释放也应返回 False，实际 got {ok}"
    assert TIMEOUT * 0.9 <= dur <= TIMEOUT * 1.3, (
        f"实际等待时长 {dur:.4f}s 不符合预期 {TIMEOUT}s"
    )

    # 释放的那个许可应该还在池子里
    assert s.available == 1, "释放的许可应该保留在池子中"
    print(f"  实际等待时长: {dur:.4f}s, 可用许可: {s.available}")
    print("PASS (strict timeout boundary enforced)")


# ============================================================
# 测试2：公平排队 FIFO
# ============================================================
def test_fair_fifo_ordering():
    """
    测试公平模式下线程按进入顺序获取许可。
    """
    print("=== test_fair_fifo_ordering ===")
    s = TimedSemaphore(0, fair=True)

    acquire_order = []
    enter_latches = [threading.Event() for _ in range(5)]

    def worker(i):
        enter_latches[i].wait()
        ok = s.acquire(timeout=2)
        if ok:
            acquire_order.append(i)
            time.sleep(0.01)
            s.release()
        else:
            acquire_order.append(-1)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()

    for i in range(5):
        enter_latches[i].set()
        time.sleep(0.01)

    for _ in range(5):
        s.release()
        time.sleep(0.01)

    for t in threads:
        t.join()

    assert acquire_order == [0, 1, 2, 3, 4], (
        f"公平模式下获取顺序错误: {acquire_order}, 期望 [0,1,2,3,4]"
    )
    print(f"  获取顺序: {acquire_order}")
    print("PASS (FIFO ordering enforced in fair mode)")


# ============================================================
# 测试3：多许可原子获取
# ============================================================
def test_multi_permit_acquire_atomic():
    print("=== test_multi_permit_acquire_atomic ===")
    s = TimedSemaphore(5)

    assert s.acquire(permits=3) is True
    assert s.available == 2

    t0 = time.monotonic()
    ok = s.acquire(permits=3, timeout=0.05)
    dur = time.monotonic() - t0
    assert ok is False, "许可不够应超时失败"
    assert 0.04 <= dur <= 0.1, f"等待时长不符: {dur}"
    assert s.available == 2, f"失败时不应扣除许可，实际 {s.available}"

    s.release(1)
    assert s.available == 3
    assert s.acquire(permits=3, timeout=0.01) is True
    assert s.available == 0

    s.release(3)
    s.release(3)
    s.set_capacity(5)  # 吞掉超额的 1 个（因为 capacity=5）
    assert s.available == 5
    print("PASS (multi-permit atomic acquire)")


def test_multi_permit_does_not_partial_steal():
    print("=== test_multi_permit_does_not_partial_steal ===")
    s = TimedSemaphore(2)

    assert s.acquire(permits=1) is True

    t2_ok = []
    def t2_worker():
        t2_ok.append(s.acquire(permits=2, timeout=0.2))

    t2 = threading.Thread(target=t2_worker)
    t2.start()
    time.sleep(0.02)

    assert s.acquire(permits=1, timeout=0.1) is True
    assert s.available == 0

    s.release(1)
    time.sleep(0.02)
    assert t2_ok == [], "T2 不应成功（不够 2 个）"

    s.release(1)
    t2.join(timeout=0.3)
    assert t2_ok == [True], f"T2 应成功拿到 2 个，实际 {t2_ok}"
    assert s.available == 0

    s.release(2)
    assert s.available == 2
    print("PASS (no partial stealing)")


# ============================================================
# 测试4：资源不泄漏（并发超时 + 释放）
# ============================================================
def test_no_leak_when_release_and_timeout_concurrent():
    print("=== test_no_leak_when_release_and_timeout_concurrent ===")
    INITIAL = 3
    for fair in [False, True]:
        s = TimedSemaphore(INITIAL, fair=fair)

        for _ in range(INITIAL):
            assert s.acquire() is True
        assert s.available == 0

        results = [None] * 10
        start_barrier = threading.Barrier(10)

        def worker(i):
            start_barrier.wait()
            results[i] = s.acquire(timeout=0.02)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()

        time.sleep(0.01)
        for _ in range(INITIAL):
            s.release()

        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r is True)
        timeout_count = sum(1 for r in results if r is False)

        assert success_count + s.available == INITIAL, (
            f"fair={fair}: 资源泄漏！成功{success_count} + 剩余{s.available} != {INITIAL}"
        )

        for _ in range(s.available):
            assert s.acquire(timeout=0.1) is True
        assert s.available == 0

        # borrowed = 先拿光的 3 + 10 个线程里成功拿到且还没还的
        # 10 个 worker 线程中，成功的 3 个拿到后没释放（它们的函数里只有 acquire 没有 release）
        # 主程序又拿了刚才的 available 个
        # 所以 borrowed = INITIAL + success_count + available（刚才主程序拿的）
        # 简化：用 borrowed 反推需要释放多少次
        while s.borrowed > 0:
            s.release(min(3, s.borrowed))
        s.set_capacity(INITIAL)
        assert s.available == INITIAL
        assert s.borrowed == 0
        print(f"  fair={fair}: success={success_count}, timeout={timeout_count}, available={s.available}")
    print("PASS (no resource leak in both modes)")


# ============================================================
# 测试5：上下文管理器（单许可 + 多许可）
# ============================================================
def test_context_manager_single():
    print("=== test_context_manager_single ===")
    s = TimedSemaphore(1)
    with s:
        assert s.available == 0
    assert s.available == 1
    print("PASS")


def test_context_manager_multi():
    """acquire_multi 上下文管理器：正常路径 + 异常路径都要正确归还。"""
    print("=== test_context_manager_multi ===")
    s = TimedSemaphore(5)

    # 正常路径：拿 2 个，退出时归还 2 个
    with s.acquire_multi(2) as ok:
        assert ok is True
        assert s.available == 3
    assert s.available == 5, "多许可正常归还失败"

    # 异常路径：拿 3 个，抛异常，退出时归还 3 个
    try:
        with s.acquire_multi(3) as ok:
            assert ok is True
            assert s.available == 2
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert s.available == 5, "多许可异常路径归还失败"

    # 超时失败路径：许可不够，返回 False，不应该有任何扣除
    with s.acquire_multi(10, timeout=0.01) as ok:
        assert ok is False
    assert s.available == 5, "超时失败时不应扣许可"

    print("PASS (acquire_multi context manager)")


# ============================================================
# 测试6：非公平模式——小请求跳过大请求
# ============================================================
def test_unfair_small_request_bypasses_large():
    """
    需求 #1：大请求排在前面凑不齐资源时，后面够资源的小请求先通过。
    用两个等待请求 + 一次释放就能看出来。
    """
    print("=== test_unfair_small_request_bypasses_large ===")
    s = TimedSemaphore(0, fair=False)
    s.release(3)
    for _ in range(3):
        assert s.acquire() is True
    assert s.available == 0

    big_ok = []
    small_ok = []
    small_release = threading.Event()

    def big_worker():
        big_ok.append(s.acquire(permits=3, timeout=1.0))

    def small_worker():
        ok = s.acquire(permits=1, timeout=1.0)
        small_ok.append(ok)
        if ok:
            # 等信号才释放（让我们先验证大请求还在等）
            small_release.wait(timeout=0.5)
            s.release(1)

    # 先启动大请求（排在前面）
    t_big = threading.Thread(target=big_worker)
    t_big.start()
    time.sleep(0.02)

    # 再启动小请求（排在后面）
    t_small = threading.Thread(target=small_worker)
    t_small.start()
    time.sleep(0.02)

    # 只释放 1 个许可——只够小请求，不够大请求
    s.release(1)
    t_small.join(timeout=0.3)

    # 小请求应该成功，大请求还在等
    assert small_ok == [True], f"小请求应成功，实际 {small_ok}"
    assert big_ok == [], f"大请求应仍在等待，实际 {big_ok}"
    assert s.available == 0

    # 让小请求释放它的 1 个，再 release(2) = 凑够 3 个给大请求
    small_release.set()
    time.sleep(0.02)
    s.release(2)
    t_big.join(timeout=0.3)
    t_small.join(timeout=0.3)
    assert big_ok == [True], f"大请求最终应成功，实际 {big_ok}"

    # 大请求归还 3 个，capacity 同步
    s.release(3)
    s.set_capacity(3)
    assert s.available == 3

    print("  小请求绕过：PASS（小请求成功，大请求等够了才成功）")
    print("PASS (unfair small bypasses large)")


# ============================================================
# 测试7：运行时动态调整容量
# ============================================================
def test_dynamic_capacity_adjust():
    """
    需求 #2：扩容/缩容不强行收回已借出许可。
    """
    print("=== test_dynamic_capacity_adjust ===")

    # --- 扩容 ---
    s = TimedSemaphore(3)
    assert s.capacity == 3
    assert s.available == 3

    # 拿走 2 个
    assert s.acquire(permits=2) is True
    assert s.available == 1

    # 扩容 +5
    s.adjust_capacity(5)
    assert s.capacity == 8
    assert s.available == 6  # 原来剩 1 + 新增 5

    # 现在能再拿 6 个
    assert s.acquire(permits=6) is True
    assert s.available == 0
    # 已经借出的 2+6=8 个，等于 capacity
    assert s.borrowed == 8

    # --- 缩容（不强行收回） ---
    s.adjust_capacity(-6)  # 从 8 缩到 2
    assert s.capacity == 2
    assert s.available == 0

    # 现在释放 8 个（之前拿的 2+6）
    s.release(8)
    # 用 set_capacity(2) 确保 available 不超过 capacity
    s.set_capacity(2)
    assert s.available == 2, f"缩容 + 全部归还后 available 应=capacity=2，实际 {s.available}"

    # --- set_capacity 绝对调整 ---
    s.set_capacity(10)
    assert s.capacity == 10
    assert s.available == 10  # 原来 2 + 扩容 8

    # 缩容到 0（服务降级）
    s.set_capacity(0)
    assert s.capacity == 0
    # available 2 个被吃掉
    assert s.available == 0

    print("PASS (dynamic capacity adjust)")


def test_capacity_expand_wakes_waiters():
    """扩容时应该唤醒等待者。"""
    print("=== test_capacity_expand_wakes_waiters ===")
    s = TimedSemaphore(0, fair=False)
    ok = []

    t = threading.Thread(target=lambda: ok.append(s.acquire(permits=2, timeout=1.0)))
    t.start()
    time.sleep(0.02)

    # 扩容 2 个，等待者应该被唤醒并成功
    s.adjust_capacity(2)
    t.join(timeout=0.3)
    assert ok == [True], f"扩容应唤醒等待者，实际 {ok}"
    assert s.available == 0
    s.release(2)
    # 缩容回 0，释放的被吞掉
    s.set_capacity(0)
    assert s.available == 0
    print("PASS (capacity expand wakes waiters)")


# ============================================================
# 测试8：可观测统计
# ============================================================
def test_observable_stats():
    """
    需求 #3：累计成功、累计超时、平均等待时间。
    """
    print("=== test_observable_stats ===")
    s = TimedSemaphore(2)

    # 第一次：成功，不等待
    assert s.acquire() is True
    stats = s.get_stats()
    assert stats["total_success"] == 1
    assert stats["total_timeout"] == 0
    assert stats["available"] == 1

    # 第二次：成功，也不等
    assert s.acquire() is True
    assert s.get_stats()["total_success"] == 2

    # 第三次：超时
    assert s.acquire(timeout=0.02) is False
    stats = s.get_stats()
    assert stats["total_success"] == 2
    assert stats["total_timeout"] == 1
    assert stats["total_success"] + stats["total_timeout"] == 3
    assert stats["avg_wait_ms"] > 0  # 超时那次至少等了 ~20ms

    # 释放一个，第四次再拿：应该成功，且平均等待时间还是正数（前面有个超时拉了均值）
    s.release()
    assert s.acquire(timeout=0.1) is True
    stats = s.get_stats()
    assert stats["total_success"] == 3
    assert stats["total_timeout"] == 1
    assert stats["queue_length"] == 0

    # 归还所有（之前拿了 3），capacity=2，多的 1 个通过 set_capacity 吞掉
    s.release(3)
    s.set_capacity(2)
    assert s.available == 2
    print(f"  统计快照: {s.get_stats()}")
    print("PASS (observable stats)")


# ============================================================
# 测试9：infinite wait 唤醒
# ============================================================
def test_infinite_wait_wakes_up():
    print("=== test_infinite_wait_wakes_up ===")
    for fair in [False, True]:
        s = TimedSemaphore(0, fair=fair)
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


# ============================================================
# 压力测试（也作为单独脚本入口）
# ============================================================
def run_stress_test(
    initial_permits: int = 10,
    num_threads: int = 50,
    operations_per_thread: int = 20,
    timeout_range: tuple = (0.01, 0.1),
    fair: bool = False,
):
    """
    压力测试：大量线程随机获取不同数量的许可，带随机超时。
    最后输出统计信息并验证资源守恒。
    """
    s = TimedSemaphore(initial_permits, fair=fair)
    stats_lock = threading.Lock()
    total_success = 0
    total_timeout = 0
    total_ops = 0

    thread_holdings = [0] * num_threads

    def worker(tid):
        nonlocal total_success, total_timeout, total_ops
        random.seed(tid + time.time_ns() % 1000000)
        for _ in range(operations_per_thread):
            permits = random.randint(1, 3)
            timeout = random.uniform(*timeout_range)
            ok = s.acquire(permits=permits, timeout=timeout)

            with stats_lock:
                total_ops += 1
                if ok:
                    total_success += 1
                    thread_holdings[tid] += permits
                else:
                    total_timeout += 1

            if ok:
                time.sleep(random.uniform(0, 0.005))
                s.release(permits)
                with stats_lock:
                    thread_holdings[tid] -= permits

    start_time = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start_time

    total_held = sum(thread_holdings)
    final_available = s.available
    expected = initial_permits
    sem_stats = s.get_stats()

    print("\n" + "=" * 64)
    print(f" 压力测试 (fair={fair}, permits={initial_permits}, "
          f"threads={num_threads}, ops/thread={operations_per_thread})")
    print("=" * 64)
    print(f"  总操作数:        {total_ops}")
    print(f"  成功数:          {total_success}")
    print(f"  超时数:          {total_timeout}")
    print(f"  成功率:          {total_success/total_ops*100:.2f}%")
    print(f"  超时率:          {total_timeout/total_ops*100:.2f}%")
    print(f"  总耗时:          {elapsed:.2f} s")
    print(f"  吞吐:            {total_ops/elapsed:.1f} ops/s")
    print("-" * 64)
    print(f"  Semaphore 内置统计：")
    print(f"    容量:           {sem_stats['capacity']}")
    print(f"    可用:           {sem_stats['available']}")
    print(f"    等待队列:       {sem_stats['queue_length']}")
    print(f"    累计成功:       {sem_stats['total_success']}")
    print(f"    累计超时:       {sem_stats['total_timeout']}")
    print(f"    平均等待:       {sem_stats['avg_wait_ms']:.3f} ms")
    print("-" * 64)
    print(f"  守恒校验：")
    print(f"    最终可用:       {final_available}")
    print(f"    线程持有:       {total_held}")
    print(f"    初始许可:       {expected}")
    conserved = (final_available + total_held) == expected
    print(f"    守恒:           {'PASS ✅' if conserved else 'FAIL ❌'}")
    print("=" * 64)

    assert conserved, (
        f"资源不守恒！可用={final_available} + 持有={total_held} != 初始={expected}"
    )
    # 双重校验：内置统计对得上
    assert sem_stats["total_success"] == total_success
    assert sem_stats["total_timeout"] == total_timeout
    return sem_stats


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stress":
        fair = len(sys.argv) > 2 and sys.argv[2] == "fair"
        run_stress_test(fair=fair)
        sys.exit(0)

    # 常规测试
    test_basic_unfair()
    test_basic_fair()
    test_strict_timeout_boundary()
    test_fair_fifo_ordering()
    test_multi_permit_acquire_atomic()
    test_multi_permit_does_not_partial_steal()
    test_unfair_small_request_bypasses_large()
    test_dynamic_capacity_adjust()
    test_capacity_expand_wakes_waiters()
    test_observable_stats()
    test_context_manager_single()
    test_context_manager_multi()
    for _ in range(20):
        test_no_leak_when_release_and_timeout_concurrent()
    test_infinite_wait_wakes_up()

    # 轻量压力测试
    print("\n--- 轻量压力测试 (非公平模式) ---")
    run_stress_test(
        initial_permits=5,
        num_threads=20,
        operations_per_thread=10,
        timeout_range=(0.005, 0.05),
        fair=False,
    )
    print("\n--- 轻量压力测试 (公平模式) ---")
    run_stress_test(
        initial_permits=5,
        num_threads=20,
        operations_per_thread=10,
        timeout_range=(0.005, 0.05),
        fair=True,
    )

    print("\n✅ All tests passed.")
    print("\n💡 运行完整压力测试:")
    print("   python test_timed_semaphore.py stress        # 非公平模式")
    print("   python test_timed_semaphore.py stress fair   # 公平模式")
