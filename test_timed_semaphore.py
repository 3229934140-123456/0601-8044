import threading
import time
import random
import sys
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
    print("=== test_strict_timeout_boundary ===")
    s = TimedSemaphore(1)
    assert s.acquire() is True  # 拿光，让池子空

    TIMEOUT = 0.1
    results = []
    wait_durations = []

    def waiter():
        t0 = time.monotonic()
        ok = s.acquire(timeout=TIMEOUT)
        dur = time.monotonic() - t0
        results.append(ok)
        wait_durations.append(dur)

    # 启动等待线程
    t = threading.Thread(target=waiter)
    t.start()
    # 等它进入等待，然后在它超时刻附近释放许可
    time.sleep(TIMEOUT * 0.6)
    s.release()  # 此时释放——虽然有了许可，但 waiter 在剩下 ~40ms 内会等到超时吗？
    # 不，这个测试要的是"时间到必失败"，所以我们把释放放在刚好超时刻之后
    # 先等线程结束
    t.join(timeout=TIMEOUT + 0.5)

    # 结果分析：因为 release 发生在 TIMEOUT*0.6（60ms 处），waiter 应该拿到
    # 所以我们需要另一种更严格的测试设计：
    # ---- 严格超时测试（真·时间到必失败） ----
    s2 = TimedSemaphore(1)
    s2.acquire()  # 拿光

    t0 = time.monotonic()
    ok2 = s2.acquire(timeout=0.08)  # 80ms 超时
    dur = time.monotonic() - t0

    # 因为全程没人 release，必失败
    assert ok2 is False, f"应超时失败，实际 ok={ok2}"
    assert 0.07 < dur < 0.2, f"等待时长 {dur:.3f}s 不符合预期 (0.07-0.2s)"

    # 关键：超时后，s2 的许可没有少（那个 acquire 没拿）
    s2.release()  # 现在归还那 1 个 + 没有其他的
    # capacity=1, release(1)=0+1=1 → aligned=1
    assert s2.available == 1, f"池子应恢复 1，实际 {s2.available}"

    print(f"  严格超时：ok=False, 等待 {dur*1000:.1f}ms ✓")
    print("PASS (strict timeout boundary)")


# ============================================================
# 测试2：公平 FIFO
# ============================================================
def test_fair_fifo_ordering():
    print("=== test_fair_fifo_ordering ===")
    s = TimedSemaphore(0, fair=True)
    # 先放 5 个拿光，让后续全部排队
    s.set_capacity(5)
    for _ in range(5):
        s.acquire()
    assert s.available == 0

    N = 4
    order = []
    start_events = [threading.Event() for _ in range(N)]

    def worker(i):
        start_events[i].set()
        if s.acquire(timeout=2.0):
            order.append(i)
            time.sleep(0.008)  # 持有一段时间，保证 FIFO 顺序可观察
            s.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    # 按顺序启动，每个启动后确认入队再启下一个
    for i, th in enumerate(threads):
        th.start()
        start_events[i].wait(timeout=0.5)
        time.sleep(0.015)  # 确保排在等待队列（队尾）

    # 依次释放许可，每次 1 个，共 4 次
    for _ in range(N):
        s.release()
        time.sleep(0.03)

    for th in threads:
        th.join(timeout=0.5)

    assert order == [0, 1, 2, 3], f"公平模式获取顺序应为 [0,1,2,3]，实际 {order}"
    # 主程序拿了 5 个没释放，加上 4 个 worker 已经释放 4 个 = av = min(4, 5)
    # 主程序再释放 5 个 → av = min(4+5, 5) = 5
    s.release(5)
    assert s.available == 5, f"最终 available={s.available}（应对齐 capacity=5）"
    print(f"  获取顺序: {order} ✓")
    print("PASS (fair FIFO)")


# ============================================================
# 测试3：多许可原子获取
# ============================================================
def test_multi_permit_acquire_atomic():
    print("=== test_multi_permit_acquire_atomic ===")
    s = TimedSemaphore(5)
    assert s.acquire(permits=3) is True
    assert s.available == 2
    # 再要 3 个不够，失败，不部分扣除
    t0 = time.monotonic()
    ok = s.acquire(permits=3, timeout=0.05)
    dur = time.monotonic() - t0
    assert ok is False, f"应失败，实际 ok={ok}"
    assert 0.04 < dur < 0.2, f"等待时长不符 {dur:.3f}"
    assert s.available == 2, f"不应部分扣除，available={s.available}"
    # 归还 3 个
    s.release(3)  # 2+3=5, capacity=5 → 对齐=5
    assert s.available == 5
    print("PASS (multi-permit atomic)")


def test_multi_permit_does_not_partial_steal():
    """多许可失败时不扣池子中任何一个。"""
    print("=== test_multi_permit_does_not_partial_steal ===")
    s = TimedSemaphore(5)
    assert s.acquire(permits=4)
    assert s.available == 1
    # 再来一个线程要 2 个（不够，超时失败）
    r = []
    def w(): r.append(s.acquire(permits=2, timeout=0.06))
    t = threading.Thread(target=w); t.start(); t.join(timeout=0.3)
    assert r == [False]
    assert s.available == 1, f"部分扣除了！av={s.available}"
    s.release(4)  # 1+4=5
    assert s.available == 5
    print("PASS (no partial steal)")


# ============================================================
# 测试4：释放+超时并发不泄漏
# ============================================================
def test_no_leak_when_release_and_timeout_concurrent():
    """
    并发竞态：10 个线程同时超时 + 3 个许可同时释放，验证没有许可丢失。
    """
    print("=== test_no_leak_when_release_and_timeout_concurrent ===")
    INITIAL = 3
    for fair in [False, True]:
        for rd in range(5):  # 每模式跑 5 轮
            s = TimedSemaphore(INITIAL, fair=fair)
            N = 10
            # 先拿光
            for _ in range(INITIAL):
                assert s.acquire() is True
            # 启动 N 个等待线程（带短超时）
            results = [None] * N
            barrier = threading.Barrier(N)
            def worker(i):
                barrier.wait()
                results[i] = s.acquire(timeout=0.02)
            ts = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
            for th in ts: th.start()
            time.sleep(0.01)  # 稍等，让他们开始等
            # 释放 INITIAL 个
            for _ in range(INITIAL): s.release()
            for th in ts: th.join(timeout=0.3)

            success_count = sum(1 for r in results if r is True)
            timeout_count = sum(1 for r in results if r is False)
            assert success_count + timeout_count == N

            # 守恒：成功数 + 池子剩余 = INITIAL
            assert success_count + s.available == INITIAL, (
                f"fair={fair} rd={rd}: 泄漏！success={success_count} + av={s.available} != {INITIAL}"
            )

            # 清理：先拿完所有剩余
            for _ in range(s.available):
                assert s.acquire(timeout=0.1) is True
            # 现在 borrowed 应该是 INITIAL + 所有成功拿到的（初始拿光 INITIAL 个 + 10 线程里成功拿到没还的）
            while s.borrowed > 0:
                s.release(min(INITIAL, s.borrowed))
            # available 因为 capacity 对齐自动 = INITIAL
            assert s.available == INITIAL

        print(f"  fair={fair}: {5} 轮无泄漏 ✓")
    print("PASS (no leak concurrent)")


# ============================================================
# 测试5：上下文管理器（单许可 + 多许可）
# ============================================================
def test_context_manager_single():
    print("=== test_context_manager_single ===")
    s = TimedSemaphore(1)
    with s:
        assert s.available == 0
    assert s.available == 1 and s.borrowed == 0
    print("PASS")


def test_context_manager_multi():
    print("=== test_context_manager_multi ===")
    s = TimedSemaphore(5)
    # 正常路径
    with s.acquire_multi(2) as ok:
        assert ok and s.available == 3
    assert s.available == 5, f"正常路径归还失败 av={s.available}"
    # 异常路径
    try:
        with s.acquire_multi(3) as ok:
            assert ok and s.available == 2
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert s.available == 5, f"异常路径归还失败 av={s.available}"
    # 超时失败
    with s.acquire_multi(10, timeout=0.01) as ok:
        assert ok is False
    assert s.available == 5
    print("PASS (acquire_multi normal/exception/timeout)")


# ============================================================
# 测试6：非公平模式——小请求跳过大请求
# ============================================================
def test_unfair_small_request_bypasses_large():
    """
    需求 #1（之前的）：大请求在前凑不齐时，小请求能先过。
    用 2 等待请求 + 1 次释放验证。
    """
    print("=== test_unfair_small_request_bypasses_large ===")
    s = TimedSemaphore(3, fair=False)
    for _ in range(3): s.acquire()  # 拿光，av=0
    assert s.available == 0

    big_ok, small_ok = [], []
    small_release = threading.Event()

    def big_worker():
        big_ok.append(s.acquire(permits=3, timeout=0.8))

    def small_worker():
        ok = s.acquire(permits=1, timeout=0.8)
        small_ok.append(ok)
        if ok:
            small_release.wait(timeout=0.5)
            s.release(1)

    t_big = threading.Thread(target=big_worker); t_big.start(); time.sleep(0.03)
    t_small = threading.Thread(target=small_worker); t_small.start(); time.sleep(0.03)

    # ---- 只释放 1 个，只够小请求 ----
    s.release(1)
    t_small.join(timeout=0.3)
    assert small_ok == [True], f"小请求应成功，实际 {small_ok}"
    assert big_ok == [], f"大请求应仍等待，实际 {big_ok}"
    assert s.available == 0

    # 关键：这就是"修好了"——大请求排前但没资源时让后面够的先过，不让许可空着

    # 让小请求释放 1 个，再 release(2) = 凑够 3 个给大请求
    small_release.set()
    time.sleep(0.03)
    s.release(2)
    t_big.join(timeout=0.3)
    t_small.join(timeout=0.3)
    assert big_ok == [True], f"大请求最终应成功，实际 {big_ok}"

    # 大请求归还 3 个：capacity=3，av = min(已对齐?, 3)
    s.release(3)  # 0+3=3 → cap=3 → av=3
    assert s.available == 3
    assert s.borrowed == 0
    print("  小请求绕过：大在前没资源，小在后够资源先走 ✓")
    print("PASS (unfair small bypasses large)")


# ============================================================
# 测试7：运行时动态调整容量（含 release 对齐）
# ============================================================
def test_dynamic_capacity_adjust():
    """
    需求 #2：扩容/缩容不强行收回，缩容后 release 自动对齐。
    """
    print("=== test_dynamic_capacity_adjust ===")
    s = TimedSemaphore(3)

    # --- 扩容：直接加 5 个可用 ---
    s.adjust_capacity(+5)
    assert s.capacity == 8 and s.available == 8

    # 拿 8 个（2 + 6）
    assert s.acquire(2) is True
    assert s.acquire(6) is True
    assert s.available == 0 and s.borrowed == 8

    # --- 缩容：从 8 → 2 ---
    s.adjust_capacity(-6)
    assert s.capacity == 2 and s.available == 0

    # --- 关键：release 8 个（全部归还），但 capacity=2，超额自动吞掉 ---
    s.release(8)
    assert s.available == 2, (
        f"缩容后全部归还，available 应=新 capacity=2，实际 {s.available}"
    )
    assert s.borrowed == 0

    # --- set_capacity 绝对调整：扩容到 10 ---
    s.set_capacity(10)
    assert s.capacity == 10 and s.available == 10, (
        f"从 2→10 扩容，加 8 个，av={s.available}"
    )

    # --- 缩容到 0：available=10，扣光 10 个 ---
    s.set_capacity(0)
    assert s.capacity == 0 and s.available == 0
    assert s.borrowed == 0

    # --- 再回到 3 ---
    s.set_capacity(3)
    assert s.capacity == 3 and s.available == 3

    print("  release 自动对齐：缩容后归还不会超过新 capacity ✓")
    print("PASS (dynamic capacity adjust)")


def test_capacity_expand_wakes_waiters():
    print("=== test_capacity_expand_wakes_waiters ===")
    s = TimedSemaphore(0)
    result = []

    def waiter():
        result.append(s.acquire(permits=2, timeout=1.0))

    t = threading.Thread(target=waiter); t.start(); time.sleep(0.03)
    assert s.queue_length == 1
    # 扩容：加 2 个容量
    s.adjust_capacity(+2)
    t.join(timeout=0.3)
    assert result == [True], f"扩容后应唤醒等待者，实际 {result}"
    # 归还：capacity=2，release(2)=2
    s.release(2)
    assert s.available == 2
    print("PASS (capacity expand wakes waiters)")


# ============================================================
# 测试8：容量负数/非法参数校验（需求 #2b）
# ============================================================
def test_negative_capacity_validation():
    print("=== test_negative_capacity_validation ===")
    s = TimedSemaphore(5)
    # 拿 3 个
    s.acquire(3)
    assert s.borrowed == 3

    # 1. set_capacity(-1) 直接拒绝
    try:
        s.set_capacity(-1)
        assert False, "应抛出 ValueError"
    except ValueError as e:
        msg = str(e)
        assert "capacity must be >= 0" in msg
        assert "outstanding borrowed" in msg
        print(f"  set_capacity(-1) 错误信息：{msg}")

    # 2. adjust_capacity 导致负数也要拒绝
    try:
        s.adjust_capacity(-10)  # 5-10=-5
        assert False, "应抛出 ValueError"
    except ValueError as e:
        msg = str(e)
        assert "would result in capacity=" in msg and "< 0" in msg
        print(f"  adjust_capacity(-10) 错误信息：{msg}")

    # 3. 拒绝后容量保持不变
    assert s.capacity == 5 and s.borrowed == 3

    # 4. 类型错误也拒绝
    try:
        s.set_capacity("5")
        assert False, "应抛出 TypeError"
    except TypeError as e:
        assert "must be int" in str(e)

    # 5. 合法缩容还是可以的（到 0 允许）
    s.set_capacity(0)
    assert s.capacity == 0 and s.borrowed == 3
    # 归还后 available 自动到 0
    s.release(3)
    assert s.available == 0

    print("PASS (negative capacity validation)")


# ============================================================
# 测试9：release 对齐 capacity（需求 #2a 的专门验证）
# ============================================================
def test_release_aligns_to_capacity():
    print("=== test_release_aligns_to_capacity ===")
    s = TimedSemaphore(10)
    # 拿光 10 个
    for _ in range(10): s.acquire()
    assert s.borrowed == 10 and s.available == 0

    # 缩容到 2
    s.set_capacity(2)
    assert s.capacity == 2 and s.available == 0

    # 归还 5 个：av 应该 = min(0+5, 2) = 2
    s.release(5)
    assert s.available == 2, f"缩容后归还 5，av 应封顶为 2，实际 {s.available}"
    assert s.borrowed == 5  # 10-5 还没还

    # 再归还 5 个：av 还是 2（已经到顶）
    s.release(5)
    assert s.available == 2
    assert s.borrowed == 0

    print("  release(5) 缩容到 2 后 av 封顶=2；再 release(5) 仍封顶=2 ✓")
    print("PASS (release auto align)")


# ============================================================
# 测试10：等待视图 inspect_waiters（需求 #1）
# ============================================================
def test_inspect_waiters_view():
    print("=== test_inspect_waiters_view ===")
    s = TimedSemaphore(0, fair=True)  # 用公平模式，is_head 有意义
    s.set_capacity(5)
    for _ in range(5): s.acquire()
    assert s.available == 0

    # 启动 3 个不同大小的等待请求
    r1, r2, r3 = [], [], []
    e1, e2, e3 = threading.Event(), threading.Event(), threading.Event()

    def w1(): e1.set(); r1.append(s.acquire(permits=5, timeout=0.5))
    def w2(): e2.set(); r2.append(s.acquire(permits=2, timeout=0.5))
    def w3(): e3.set(); r3.append(s.acquire(permits=1, timeout=0.5))

    t1 = threading.Thread(target=w1); t1.start(); e1.wait(); time.sleep(0.02)
    t2 = threading.Thread(target=w2); t2.start(); e2.wait(); time.sleep(0.02)
    t3 = threading.Thread(target=w3); t3.start(); e3.wait(); time.sleep(0.02)
    time.sleep(0.01)

    view = s.inspect_waiters()
    assert len(view) == 3, f"视图长度应为 3，实际 {len(view)}: {view}"

    # 公平模式按顺序：permits 5 → 2 → 1
    assert view[0]["permits"] == 5 and view[0]["is_head"] is True, f"head={view[0]}"
    assert view[1]["permits"] == 2 and view[1]["is_head"] is False
    assert view[2]["permits"] == 1 and view[2]["is_head"] is False

    # 每一项 waited_ms 都是递增正数
    for i, v in enumerate(view):
        assert v["waited_ms"] >= 0, f"#{i} waited_ms={v}"
        assert v["granted"] is False
        assert v["timed_out"] is False
        assert v["closed_rejected"] is False

    print(f"  等待视图（公平模式，3 项）：")
    for i, v in enumerate(view):
        print(f"    #{i}: permits={v['permits']:<2d}  waited={v['waited_ms']:.2f}ms"
              f"  head={v['is_head']}")

    # ---- 公平模式：释放 3 个，队首 5 不够 → 2 和 1 仍然等着（严格 FIFO） ----
    s.release(3)
    time.sleep(0.05)
    view2 = s.inspect_waiters()
    assert len(view2) == 3, (
        f"公平模式：队首 5 不满足，后面的 2 和 1 也都不能拿（大请求堵住），实际 {view2}"
    )
    # r1/r2/r3 都还没拿到
    assert r1 == [] and r2 == [] and r3 == [], (
        f"公平模式：队首堵住应没人成功，实际 r1={r1} r2={r2} r3={r3}"
    )
    print(f"  公平模式释放 3 个后：队首 5 仍然堵住，3 个请求全在等（符合严格 FIFO）✓")

    # ---- 现在把闸门换成非公平模式（重开一个验证 small bypass） ----
    print(f"  额外验证：切非公平模式，同样 3 个请求等待 + release(3)")
    s2 = TimedSemaphore(0, fair=False)
    s2.set_capacity(5)
    for _ in range(5): s2.acquire()  # 拿光
    rr1, rr2, rr3 = [], [], []
    ee1, ee2, ee3 = threading.Event(), threading.Event(), threading.Event()
    sr_ev = threading.Event()
    def ww1():
        ee1.set()
        ok = s2.acquire(5, timeout=0.5); rr1.append(ok)
    def ww2():
        ee2.set()
        ok = s2.acquire(2, timeout=0.5); rr2.append(ok)
        if ok:
            sr_ev.wait(timeout=0.4)
            s2.release(2)
    def ww3():
        ee3.set()
        ok = s2.acquire(1, timeout=0.5); rr3.append(ok)
        if ok:
            sr_ev.wait(timeout=0.4)
            s2.release(1)

    tt1 = threading.Thread(target=ww1); tt1.start(); ee1.wait(); time.sleep(0.02)
    tt2 = threading.Thread(target=ww2); tt2.start(); ee2.wait(); time.sleep(0.02)
    tt3 = threading.Thread(target=ww3); tt3.start(); ee3.wait(); time.sleep(0.02)
    # 只 release(3)：只够 2+1 过，队首 5 还不够
    s2.release(3)
    time.sleep(0.05)
    v = s2.inspect_waiters()
    assert len(v) == 1 and v[0]["permits"] == 5, f"非公平：小请求应绕过队首，实际 {v}"
    assert rr1 == [] and rr2 == [True] and rr3 == [True], (
        f"非公平：2+1 应成功，5 还在等，实际 {rr1} {rr2} {rr3}"
    )
    print(f"  非公平模式：小请求（2+1）绕过队首 5，小的先过 ✓")
    sr_ev.set()
    time.sleep(0.02)
    s2.release(5)  # 再补够大请求的
    tt1.join(timeout=0.3); tt2.join(timeout=0.3); tt3.join(timeout=0.3)
    assert rr1 == [True]
    s2.release(8)  # 5+2+1 = 8 全归还，capacity=5
    assert s2.available == 5

    # ---- 清理公平模式：s 池子 ----
    s.release(5)  # 补够给队首 5
    t1.join(timeout=0.3); t2.join(timeout=0.3); t3.join(timeout=0.3)
    assert r1 == [True]
    s.release(8)  # 5+2+1
    assert s.available == 5

    print("PASS (inspect waiters view)")


# ============================================================
# 测试11：可观测统计（含 closed_rejected 新字段）
# ============================================================
def test_observable_stats():
    print("=== test_observable_stats ===")
    s = TimedSemaphore(2)
    # 拿 2 + 2 尝试 1 次超时
    assert s.acquire() is True
    assert s.acquire() is True
    assert s.acquire(timeout=0.01) is False  # timeout
    # 释放 1，再拿 1（成功）
    s.release()
    assert s.acquire(timeout=0.1) is True    # success #3
    stats = s.get_stats()
    assert stats["total_success"] == 3
    assert stats["total_timeout"] == 1
    assert stats["total_closed_rejected"] == 0
    assert stats["avg_wait_ms"] > 0
    assert stats["available"] == 0 and stats["capacity"] == 2
    # 关闭闸门后再来一次 → closed_rejected
    s.close()
    assert s.is_closed
    assert s.acquire(timeout=0.1) is False   # closed
    stats2 = s.get_stats()
    assert stats2["total_closed_rejected"] == 1
    # 开启，归还
    s.open()
    # 现在 borrowed = 3（之前拿了 2，释放了 1 后拿了 1 = 仍持有 2？让我算清楚：）
    # 初始 av=2：acquire, acquire → borrowed=2, av=0
    # release → borrowed=1, av=1
    # acquire(timeout=0.1) 成功 → borrowed=2, av=0
    # 再加上 关闭测试失败，没拿。所以 borrowed = 2
    s.release(2)
    assert s.available == 2 and s.borrowed == 0
    stats3 = s.get_stats()
    print(f"  最终统计：success={stats3['total_success']}, "
          f"timeout={stats3['total_timeout']}, "
          f"closed_rej={stats3['total_closed_rejected']}, "
          f"avg_wait={stats3['avg_wait_ms']:.2f}ms")
    print("PASS (observable stats)")


# ============================================================
# 测试12：闸门 close/open（需求 #3）
# ============================================================
def test_close_gate_behavior():
    print("=== test_close_gate_behavior ===")
    s = TimedSemaphore(0, fair=False)
    s.set_capacity(5)
    for _ in range(5): s.acquire()  # 拿光，av=0

    # 启动 3 个等 1 的，1 个等 2 的
    r = [[] for _ in range(4)]
    events = [threading.Event() for _ in range(4)]
    def worker(i, n):
        events[i].set()
        r[i].append(s.acquire(permits=n, timeout=3.0))

    # 3 个要 1，1 个要 2
    ts = []
    ts.append(threading.Thread(target=worker, args=(0, 1))); ts[-1].start(); events[0].wait(); time.sleep(0.01)
    ts.append(threading.Thread(target=worker, args=(1, 1))); ts[-1].start(); events[1].wait(); time.sleep(0.01)
    ts.append(threading.Thread(target=worker, args=(2, 1))); ts[-1].start(); events[2].wait(); time.sleep(0.01)
    ts.append(threading.Thread(target=worker, args=(3, 2))); ts[-1].start(); events[3].wait(); time.sleep(0.01)
    time.sleep(0.02)

    assert s.queue_length == 4, f"应有 4 人在等，实际 {s.queue_length}"

    # --- 关闭闸门 ---
    s.close()
    assert s.is_closed

    # --- 等待者全部被唤醒并失败 ---
    for i, th in enumerate(ts):
        th.join(timeout=0.3)
        assert r[i] == [False], f"等待者 #{i} 应因关闭而失败，实际 {r[i]}"

    # --- 新请求立即失败，不等待 ---
    t0 = time.monotonic()
    ok = s.acquire(timeout=5.0)
    dur = time.monotonic() - t0
    assert ok is False
    assert dur < 0.05, f"新请求应立即失败，等了 {dur*1000:.1f}ms"

    # --- 已拿到的 5 个依然可以正常归还 ---
    s.release(5)
    assert s.available == 5, "关闭期间归还也应有效（但 capacity=5，av=5）"

    # --- 开启闸门后新请求可成功 ---
    s.open()
    assert not s.is_closed
    assert s.acquire(permits=3) is True
    s.release(3)
    assert s.available == 5

    # --- 统计应看到 4 + 1 个 closed_rejected ---
    stats = s.get_stats()
    assert stats["total_closed_rejected"] == 5, (
        f"应有 4 等待+1 新请求 = 5 个关闭拒绝，实际 {stats['total_closed_rejected']}"
    )

    print(f"  关闭后：4 个等待者全部失败，新请求立即返回，已持有可正常归还 ✓")
    print(f"  closed_rejected 统计={stats['total_closed_rejected']} ✓")
    print("PASS (close gate behavior)")


# ============================================================
# 压力测试（基础版）
# ============================================================
def run_stress_test(permits: int, n_threads: int, ops_per_thread: int,
                    work_ms_range: tuple, fair: bool):
    title = f"压力测试 (fair={fair}, permits={permits}, threads={n_threads}, ops/thread={ops_per_thread})"
    print("=" * 64)
    print(f" {title} ")
    print("=" * 64)

    sem = TimedSemaphore(permits, fair=fair)
    work_min, work_max = work_ms_range
    start_barrier = threading.Barrier(n_threads)
    total_ops = n_threads * ops_per_thread
    external_success = [0]
    external_timeout = [0]
    external_closed = [0]
    lock = threading.Lock()
    t_start = time.monotonic()

    def worker(idx):
        my_ok = my_to = my_cr = 0
        start_barrier.wait()
        for _ in range(ops_per_thread):
            p = random.choice([1, 1, 2, 1])  # 多拿单许可
            ok = sem.acquire(permits=p, timeout=0.15)
            if ok:
                my_ok += 1
                time.sleep(random.uniform(work_min, work_max))
                sem.release(p)
            else:
                if sem.is_closed:
                    my_cr += 1
                else:
                    my_to += 1
        with lock:
            external_success[0] += my_ok
            external_timeout[0] += my_to
            external_closed[0] += my_cr

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads: t.start()
    for t in threads: t.join()
    t_end = time.monotonic()

    dur = t_end - t_start
    ok_cnt = external_success[0]
    to_cnt = external_timeout[0]
    cr_cnt = external_closed[0]
    st = sem.get_stats()

    print(f"  总操作数:        {total_ops}")
    print(f"  成功数:          {ok_cnt}")
    print(f"  超时数:          {to_cnt}")
    if cr_cnt > 0:
        print(f"  关闭拒绝数:      {cr_cnt}")
    print(f"  成功率:          {100*ok_cnt/total_ops:.2f}%")
    if to_cnt > 0:
        print(f"  超时率:          {100*to_cnt/total_ops:.2f}%")
    print(f"  总耗时:          {dur:.2f} s")
    print(f"  吞吐:            {total_ops/dur:.1f} ops/s")
    print("-" * 64)
    print(f" Semaphore 内置统计：")
    print(f"   容量:           {st['capacity']}")
    print(f"   可用:           {st['available']}")
    print(f"   等待队列:       {st['queue_length']}")
    print(f"   累计成功:       {st['total_success']}")
    print(f"   累计超时:       {st['total_timeout']}")
    if st['total_closed_rejected'] > 0:
        print(f"   累计关闭拒绝:   {st['total_closed_rejected']}")
    print(f"   平均等待:       {st['avg_wait_ms']:.3f} ms")
    print("-" * 64)

    # 守恒校验
    init = permits
    final_av = st['available']
    borrowed = st['outstanding']
    conserved = (final_av + borrowed == init)
    print(f"  守恒校验：")
    print(f"    最终可用:       {final_av}")
    print(f"    线程持有:       {borrowed}")
    print(f"    初始许可:       {init}")
    print(f"    守恒:           {'PASS ✅' if conserved else 'FAIL ❌'}")
    print("=" * 64)

    # 外部统计 vs 内部统计双重校验
    assert st['total_success'] == ok_cnt, (
        f"内置成功 {st['total_success']} != 外部统计 {ok_cnt}"
    )
    assert abs(st['total_timeout'] + st['total_closed_rejected'] - to_cnt - cr_cnt) <= 0, (
        f"内置失败 {st['total_timeout']}+{st['total_closed_rejected']} != 外部 {to_cnt}+{cr_cnt}"
    )
    assert conserved, f"资源不守恒！av={final_av} + borrowed={borrowed} != {init}"


# ============================================================
# 压力测试（动态场景：随机扩容/缩容/关闭/恢复）——需求 #4
# ============================================================
def run_stress_test_dynamic(permits: int, n_threads: int, ops_per_thread: int,
                            work_ms_range: tuple, fair: bool):
    title = (f"动态压力测试 (fair={fair}, permits={permits}, "
             f"threads={n_threads}, ops/thread={ops_per_thread})")
    print("=" * 64)
    print(f" {title} ")
    print("=" * 64)

    sem = TimedSemaphore(permits, fair=fair)
    work_min, work_max = work_ms_range
    start_barrier = threading.Barrier(n_threads + 1)  # +1 给动态控制线程
    stop_controller = threading.Event()
    history = []     # 记录动态操作
    lock = threading.Lock()

    ext_success = [0]; ext_timeout = [0]; ext_closed = [0]
    t_start = time.monotonic()

    def worker(idx):
        my_ok = my_to = my_cr = 0
        start_barrier.wait()
        for _ in range(ops_per_thread):
            p = random.choice([1, 1, 1, 2, 3])
            ok = sem.acquire(permits=p, timeout=0.2)
            if ok:
                my_ok += 1
                time.sleep(random.uniform(work_min, work_max))
                try:
                    sem.release(p)
                except Exception:
                    pass
            else:
                if sem.is_closed:
                    my_cr += 1
                else:
                    my_to += 1
        with lock:
            ext_success[0] += my_ok
            ext_timeout[0] += my_to
            ext_closed[0] += my_cr

    def controller():
        """动态控制线程：随机扩容 / 缩容 / 关闭 / 恢复。"""
        start_barrier.wait()
        op_count = 0
        while not stop_controller.is_set():
            time.sleep(random.uniform(0.02, 0.06))
            op = random.choices(
                ["expand", "shrink", "close", "open", "noop"],
                weights=[3, 3, 1, 1, 2]
            )[0]
            try:
                if op == "expand":
                    d = random.choice([1, 1, 2, 3])
                    sem.adjust_capacity(+d)
                    history.append(f"+{d}")
                elif op == "shrink":
                    d = random.choice([1, 1, 2])
                    # 避免缩到负数
                    try:
                        sem.adjust_capacity(-d)
                        history.append(f"-{d}")
                    except ValueError:
                        history.append("-X(cap floor)")
                elif op == "close":
                    if not sem.is_closed:
                        sem.close()
                        history.append("CLOSE")
                elif op == "open":
                    if sem.is_closed:
                        sem.open()
                        history.append("OPEN")
                else:
                    history.append(".")
                op_count += 1
            except Exception as e:
                history.append(f"ERR({e})")
        # 测试结束：确保闸门开启
        if sem.is_closed: sem.open()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    ctrl_th = threading.Thread(target=controller, daemon=True)
    for t in threads: t.start()
    ctrl_th.start()

    for t in threads: t.join()
    stop_controller.set()
    ctrl_th.join(timeout=1.0)
    t_end = time.monotonic()

    dur = t_end - t_start
    total_ops = n_threads * ops_per_thread
    ok_cnt = ext_success[0]
    to_cnt = ext_timeout[0]
    cr_cnt = ext_closed[0]
    st = sem.get_stats()

    print(f"  总操作数:        {total_ops}")
    print(f"  成功数:          {ok_cnt}")
    print(f"  超时数:          {to_cnt}")
    print(f"  关闭拒绝数:      {cr_cnt}")
    print(f"  成功率:          {100*ok_cnt/total_ops:.2f}%")
    print(f"  总耗时:          {dur:.2f} s")
    print(f"  吞吐:            {total_ops/dur:.1f} ops/s")
    print("-" * 64)
    print(f"  动态操作数:      {len(history)} (示例: {''.join(history[:12])}...)")
    print("-" * 64)
    print(f" Semaphore 内置统计：")
    print(f"   容量:           {st['capacity']}")
    print(f"   可用:           {st['available']}")
    print(f"   等待队列:       {st['queue_length']}")
    print(f"   累计成功:       {st['total_success']}")
    print(f"   累计超时:       {st['total_timeout']}")
    print(f"   累计关闭拒绝:   {st['total_closed_rejected']}")
    print(f"   平均等待:       {st['avg_wait_ms']:.3f} ms")
    print("-" * 64)

    # 守恒：动态场景下需要用 capacity 历史来算 —— 但因为动态调整很复杂，
    # 我们用更简单的断言：
    #   1. available <= capacity（release 自动对齐保证）
    #   2. outstanding >= 0
    #   3. 最后把 outstanding 全部归还后，available == capacity（最终守恒）
    init = permits
    final_av = st['available']
    final_borrowed = st['outstanding']
    final_cap = st['capacity']
    assert final_av <= final_cap, f"av={final_av} > cap={final_cap}（没对齐）"
    assert final_borrowed >= 0

    # 归还剩下所有借出的
    while sem.borrowed > 0:
        sem.release(min(permits, sem.borrowed))
    after_av = sem.available
    conserved = (after_av == final_cap)

    print(f"  守恒校验（动态容量）：")
    print(f"    最终容量:       {final_cap}")
    print(f"    归还后可用:     {after_av}")
    print(f"    (归还前可用:    {final_av}, 未归还: {final_borrowed})")
    print(f"    守恒:           {'PASS ✅' if conserved else 'FAIL ❌'}")
    print("=" * 64)

    assert conserved, f"最终不守恒：after_av={after_av} != cap={final_cap}"


# ============================================================
# CLI 入口
# ============================================================
def run_all_unit_tests():
    tests = [
        test_basic_unfair,
        test_basic_fair,
        test_strict_timeout_boundary,
        test_fair_fifo_ordering,
        test_multi_permit_acquire_atomic,
        test_multi_permit_does_not_partial_steal,
        test_no_leak_when_release_and_timeout_concurrent,
        test_context_manager_single,
        test_context_manager_multi,
        test_unfair_small_request_bypasses_large,
        test_dynamic_capacity_adjust,
        test_capacity_expand_wakes_waiters,
        test_negative_capacity_validation,
        test_release_aligns_to_capacity,
        test_inspect_waiters_view,
        test_observable_stats,
        test_close_gate_behavior,
    ]
    fail = 0
    for fn in tests:
        try:
            fn()
            print()
        except Exception as e:
            fail += 1
            print(f"FAIL [{fn.__name__}]: {e}")
            import traceback
            traceback.print_exc()
            print()
    if fail == 0:
        print(f"==== 所有 {len(tests)} 个单元测试全部通过 ====\n")
    else:
        print(f"==== {fail}/{len(tests)} 个测试失败 ====\n")
    return fail == 0


def main():
    args = sys.argv[1:]
    if not args:
        # 默认：跑所有单元测试 + 一个基础压力测试（非公平+公平）
        ok = run_all_unit_tests()
        if not ok:
            sys.exit(1)
        print("\n================ 基础压力测试（非公平） ================\n")
        run_stress_test(5, 10, 8, (0.005, 0.04), fair=False)
        print("\n================ 基础压力测试（公平） ================\n")
        run_stress_test(5, 10, 8, (0.005, 0.04), fair=True)
        print("\n================ 动态压力测试（非公平） ================\n")
        run_stress_test_dynamic(5, 10, 8, (0.005, 0.04), fair=False)
        print("\n================ 动态压力测试（公平） ================\n")
        run_stress_test_dynamic(5, 10, 8, (0.005, 0.04), fair=True)
        return

    mode = args[0].lower()
    if mode in ("stress",):
        fair = len(args) >= 2 and args[1].lower() == "fair"
        # 大一点
        run_stress_test(10, 20, 15, (0.005, 0.05), fair=fair)
        print()
        run_stress_test_dynamic(10, 20, 15, (0.005, 0.05), fair=fair)
    elif mode == "unit":
        sys.exit(0 if run_all_unit_tests() else 1)
    else:
        print(f"用法: python {sys.argv[0]} [stress [fair] | unit]")
        sys.exit(1)


if __name__ == "__main__":
    main()
