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
    用 ticketing 机制确保线程严格按 0,1,2,3,4 的顺序进入 acquire 排队。
    """
    print("=== test_fair_fifo_ordering ===")
    s = TimedSemaphore(0, fair=True)

    acquire_order = []
    # 5 个门闩，每个线程必须等前一个线程"已入队"后才能开始
    enter_latches = [threading.Event() for _ in range(5)]
    entered_latches = [threading.Event() for _ in range(5)]

    def worker(i):
        # 等待"我可以入队了"的信号
        enter_latches[i].wait()
        # 调用 acquire（会按调用顺序排队）
        ok = s.acquire(timeout=2)
        if ok:
            acquire_order.append(i)
            time.sleep(0.01)  # 持一小会儿，确保下一个人真的要等
            s.release()
        else:
            acquire_order.append(-1)  # 标记超时

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()

    # 依次放行每个线程进入 acquire，确保排队顺序就是 0,1,2,3,4
    for i in range(5):
        enter_latches[i].set()
        # 等一小会儿，确保线程 i 真的进入了 semaphore 的等待队列
        time.sleep(0.01)

    # 现在 5 个线程都在排队，依次释放 5 个许可
    for _ in range(5):
        s.release()
        time.sleep(0.01)

    for t in threads:
        t.join()

    # 公平模式下，拿到许可的顺序应该就是 0,1,2,3,4
    assert acquire_order == [0, 1, 2, 3, 4], (
        f"公平模式下获取顺序错误: {acquire_order}, 期望 [0,1,2,3,4]"
    )
    print(f"  获取顺序: {acquire_order}")
    print("PASS (FIFO ordering enforced in fair mode)")


# ============================================================
# 测试3：多许可原子获取
# ============================================================
def test_multi_permit_acquire_atomic():
    """
    一次获取多个许可：要么全拿到要么全失败，绝不部分扣除。
    """
    print("=== test_multi_permit_acquire_atomic ===")
    s = TimedSemaphore(5)

    # 拿 3 个
    assert s.acquire(permits=3) is True
    assert s.available == 2

    # 再拿 3 个——不够，等 50ms 应该失败
    t0 = time.monotonic()
    ok = s.acquire(permits=3, timeout=0.05)
    dur = time.monotonic() - t0
    assert ok is False, f"许可不够应超时失败"
    assert 0.04 <= dur <= 0.1, f"等待时长不符: {dur}"
    # 失败时可用许可应还是 2（没被部分扣除）
    assert s.available == 2, f"失败时不应扣除许可，实际 {s.available}"

    # 释放 1 个，变成 3，这时候再拿 3 个应该成功
    s.release(1)
    assert s.available == 3
    assert s.acquire(permits=3, timeout=0.01) is True
    assert s.available == 0

    # 释放 3 + 之前的 3 = 6
    s.release(3)
    s.release(3)
    assert s.available == 6
    print("PASS (multi-permit atomic acquire)")


def test_multi_permit_does_not_partial_steal():
    """
    验证多许可请求不会从单许可请求身上"抢"部分资源。
    非公平模式下虽然允许插队，但原子性仍然保证。
    """
    print("=== test_multi_permit_does_not_partial_steal ===")
    s = TimedSemaphore(2)

    # T1 拿 1 个，剩 1
    assert s.acquire(permits=1) is True

    # T2 要拿 2 个，不够，开始等
    t2_ok = []
    def t2_worker():
        t2_ok.append(s.acquire(permits=2, timeout=0.2))

    t2 = threading.Thread(target=t2_worker)
    t2.start()
    time.sleep(0.02)  # 确保 T2 已经开始等

    # T3 拿 1 个——应该能拿到（非公平插队），但 T2 还是要等够 2 个
    assert s.acquire(permits=1, timeout=0.1) is True
    assert s.available == 0

    # T1 释放 1 个，T2 还是不够（1 < 2）
    s.release(1)
    time.sleep(0.02)
    assert t2_ok == [], "T2 不应成功（不够 2 个）"

    # T3 也释放 1 个，现在有 2 个了，T2 应该能拿到
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

        # 先拿光
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

        # 守恒：成功拿到的 + 池子里剩的 = 释放的 3 个
        assert success_count + s.available == INITIAL, (
            f"fair={fair}: 资源泄漏！成功{success_count} + 剩余{s.available} != {INITIAL}"
        )

        # 验证剩余的许可能被新请求拿到
        for _ in range(s.available):
            assert s.acquire(timeout=0.1) is True
        assert s.available == 0

        # 全部释放回去
        s.release(INITIAL)
        assert s.available == INITIAL
        print(f"  fair={fair}: success={success_count}, timeout={timeout_count}, available={s.available}")
    print("PASS (no resource leak in both modes)")


# ============================================================
# 测试5：上下文管理器
# ============================================================
def test_context_manager():
    print("=== test_context_manager ===")
    s = TimedSemaphore(1)
    with s:
        assert s.available == 0
    assert s.available == 1
    print("PASS")


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
    total_wait_time = 0.0
    total_ops = 0

    # 每个线程的总持有量（用于最后验证守恒）
    thread_holdings = [0] * num_threads

    def worker(tid):
        nonlocal total_success, total_timeout, total_wait_time, total_ops
        random.seed(tid + time.time_ns() % 1000000)
        for _ in range(operations_per_thread):
            permits = random.randint(1, 3)
            timeout = random.uniform(*timeout_range)
            t0 = time.monotonic()
            ok = s.acquire(permits=permits, timeout=timeout)
            dur = time.monotonic() - t0

            with stats_lock:
                total_ops += 1
                total_wait_time += dur
                if ok:
                    total_success += 1
                    thread_holdings[tid] += permits
                else:
                    total_timeout += 1

            if ok:
                # 持一小会儿再释放
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

    # 资源守恒验证
    total_held = sum(thread_holdings)
    final_available = s.available
    expected = initial_permits

    print("\n" + "=" * 60)
    print(f"压力测试结果 (fair={fair}, permits={initial_permits}, threads={num_threads}, ops/thread={operations_per_thread})")
    print("=" * 60)
    print(f"  总操作数:        {total_ops}")
    print(f"  成功数:          {total_success}")
    print(f"  超时数:          {total_timeout}")
    print(f"  成功率:          {total_success/total_ops*100:.2f}%")
    print(f"  超时率:          {total_timeout/total_ops*100:.2f}%")
    print(f"  平均等待时间:    {total_wait_time/total_ops*1000:.3f} ms")
    print(f"  总耗时:          {elapsed:.2f} s")
    print(f"  吞吐:            {total_ops/elapsed:.1f} ops/s")
    print("-" * 60)
    print(f"  最终可用许可:    {final_available}")
    print(f"  线程持有合计:    {total_held}")
    print(f"  初始许可:        {expected}")
    print(f"  守恒验证:        {'PASS' if (final_available + total_held) == expected else 'FAIL'}")
    print("=" * 60)

    assert final_available + total_held == expected, (
        f"资源不守恒！可用={final_available} + 持有={total_held} != 初始={expected}"
    )
    return {
        "success": total_success,
        "timeout": total_timeout,
        "avg_wait_ms": total_wait_time / total_ops * 1000,
        "final_available": final_available,
    }


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
    for _ in range(20):
        test_no_leak_when_release_and_timeout_concurrent()
    test_context_manager()
    test_infinite_wait_wakes_up()

    # 跑一轮轻量压力测试
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
