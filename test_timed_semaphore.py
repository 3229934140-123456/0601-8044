import threading
import time
import random
import sys
from timed_semaphore import (
    TimedSemaphore, AcquireResult, AcquireFailReason,
)


# ============================================================
# 基础测试
# ============================================================
def test_basic_unfair():
    print("=== test_basic_unfair ===")
    s = TimedSemaphore(2, fair=False)
    assert s.is_fair is False
    res = s.acquire()
    assert isinstance(res, AcquireResult) and bool(res) and res.ok
    res2 = s.acquire()
    assert bool(res2) and s.available == 0
    s.release()
    assert s.available == 1
    s.release()
    assert s.available == 2
    print("PASS")


def test_basic_fair():
    print("=== test_basic_fair ===")
    s = TimedSemaphore(2, fair=True)
    assert s.is_fair is True
    assert bool(s.acquire()) and bool(s.acquire())
    assert s.available == 0
    s.release()
    s.release()
    assert s.available == 2
    print("PASS")


# ============================================================
# 测试1：返回值带失败原因
# ============================================================
def test_acquire_result_carries_reason():
    print("=== test_acquire_result_carries_reason ===")
    s = TimedSemaphore(1)
    # 成功：ok=True, reason=None
    r1 = s.acquire()
    assert r1.ok is True and r1.reason is None
    assert isinstance(r1, AcquireResult)
    assert bool(r1) is True

    # 超时
    t0 = time.monotonic()
    r2 = s.acquire(timeout=0.06)
    dur = time.monotonic() - t0
    assert r2.ok is False
    assert r2.reason == AcquireFailReason.TIMEOUT
    assert 0.05 < dur < 0.2
    assert r2.waited_ms > 50

    # 关闭
    s.close()
    r3 = s.acquire(timeout=5.0)
    assert r3.ok is False
    assert r3.reason == AcquireFailReason.CLOSED
    assert r3.waited_ms < 10, "关闭后应立即返回"
    s.open()

    # 取消
    s2 = TimedSemaphore(0, fair=False)
    s2.set_capacity(3)
    for _ in range(3): s2.acquire()
    res_list = []
    ev = threading.Event()
    def w():
        ev.set()
        res = s2.acquire(timeout=2.0, request_id="req-42", name="batch-job")
        res_list.append(res)
    t = threading.Thread(target=w); t.start(); ev.wait(); time.sleep(0.02)
    assert s2.queue_length == 1
    # 取消
    n = s2.cancel_wait("req-42")
    assert n == 1
    t.join(timeout=0.3)
    assert len(res_list) == 1
    r4 = res_list[0]
    assert r4.ok is False
    assert r4.reason == AcquireFailReason.CANCELLED
    assert r4.permits == 1

    s2.release(3)
    print(f"  超时={r2.reason.value}, 关闭={r3.reason.value}, 取消={r4.reason.value} ✓")
    print("PASS (acquire result carries reason)")


# ============================================================
# 测试2：严格超时边界（返回值验证）
# ============================================================
def test_strict_timeout_boundary():
    print("=== test_strict_timeout_boundary ===")
    s = TimedSemaphore(1)
    s.acquire()  # 拿光
    t0 = time.monotonic()
    res = s.acquire(timeout=0.08)
    dur = time.monotonic() - t0
    assert res.ok is False
    assert res.reason == AcquireFailReason.TIMEOUT
    assert 0.07 < dur < 0.2
    s.release()
    assert s.available == 1
    print(f"  严格超时：ok=False, reason=timeout, 等待 {dur*1000:.1f}ms ✓")
    print("PASS (strict timeout boundary)")


# ============================================================
# 测试3：公平 FIFO
# ============================================================
def test_fair_fifo_ordering():
    print("=== test_fair_fifo_ordering ===")
    s = TimedSemaphore(0, fair=True)
    s.set_capacity(5)
    for _ in range(5): s.acquire()  # 主程序拿 5 个
    assert s.available == 0

    N = 4
    order = []
    evs = [threading.Event() for _ in range(N)]

    def worker(i):
        evs[i].set()
        res = s.acquire(timeout=2.0)
        if res:
            order.append(i)
            time.sleep(0.008)
            s.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for i, th in enumerate(threads):
        th.start(); evs[i].wait(timeout=0.5); time.sleep(0.015)

    for _ in range(N):
        s.release()
        time.sleep(0.03)

    for th in threads: th.join(timeout=0.5)
    assert order == [0, 1, 2, 3], f"获取顺序 {order}"
    s.release(5)  # 主程序归还
    assert s.available == 5
    print(f"  获取顺序: {order} ✓")
    print("PASS (fair FIFO)")


# ============================================================
# 测试4：多许可原子
# ============================================================
def test_multi_permit_atomic():
    print("=== test_multi_permit_atomic ===")
    s = TimedSemaphore(5)
    r1 = s.acquire(permits=3)
    assert r1 and s.available == 2
    t0 = time.monotonic()
    r2 = s.acquire(permits=3, timeout=0.05)
    dur = time.monotonic() - t0
    assert not r2 and r2.reason == AcquireFailReason.TIMEOUT
    assert 0.04 < dur < 0.2 and s.available == 2
    s.release(3)
    assert s.available == 5
    print("PASS (multi-permit atomic)")


def test_multi_permit_no_partial_steal():
    print("=== test_multi_permit_no_partial_steal ===")
    s = TimedSemaphore(5)
    assert s.acquire(4) and s.available == 1
    r = []
    def w(): r.append(s.acquire(2, timeout=0.06))
    t = threading.Thread(target=w); t.start(); t.join(timeout=0.3)
    assert len(r) == 1 and not r[0]
    assert s.available == 1
    s.release(4)
    assert s.available == 5
    print("PASS (no partial steal)")


# ============================================================
# 测试5：释放+超时并发不泄漏
# ============================================================
def test_no_leak_concurrent():
    print("=== test_no_leak_concurrent ===")
    INITIAL = 3
    for fair in [False, True]:
        for rd in range(5):
            s = TimedSemaphore(INITIAL, fair=fair)
            N = 10
            for _ in range(INITIAL): s.acquire()
            results = [None]*N
            b = threading.Barrier(N)
            def wk(i):
                b.wait()
                results[i] = s.acquire(timeout=0.02)
            ts = [threading.Thread(target=wk, args=(i,)) for i in range(N)]
            for th in ts: th.start()
            time.sleep(0.01)
            for _ in range(INITIAL): s.release()
            for th in ts: th.join(timeout=0.3)

            sc = sum(1 for r in results if r)
            tc = sum(1 for r in results if not r)
            assert sc + s.available == INITIAL, (
                f"fair={fair} rd={rd}: {sc}+{s.available}!={INITIAL}"
            )
            # 清理
            for _ in range(s.available):
                assert s.acquire(timeout=0.1)
            while s.borrowed > 0:
                s.release(min(INITIAL, s.borrowed))
            assert s.available == INITIAL
        print(f"  fair={fair}: 5 轮无泄漏 ✓")
    print("PASS (no leak concurrent)")


# ============================================================
# 测试6：上下文管理器
# ============================================================
def test_ctx_single():
    print("=== test_ctx_single ===")
    s = TimedSemaphore(1)
    with s:
        assert s.available == 0
    assert s.available == 1 and s.borrowed == 0
    print("PASS")


def test_ctx_multi():
    print("=== test_ctx_multi ===")
    s = TimedSemaphore(5)
    with s.acquire_multi(2) as ok:
        assert ok and s.available == 3
    assert s.available == 5
    try:
        with s.acquire_multi(3) as ok:
            assert ok and s.available == 2
            raise RuntimeError()
    except RuntimeError:
        pass
    assert s.available == 5
    with s.acquire_multi(10, timeout=0.01) as ok:
        assert ok is False
    assert s.available == 5
    print("PASS (ctx multi normal/exc/timeout)")


# ============================================================
# 测试7：非公平小请求绕过
# ============================================================
def test_unfair_small_bypass_large():
    print("=== test_unfair_small_bypass_large ===")
    s = TimedSemaphore(3, fair=False)
    for _ in range(3): s.acquire()
    assert s.available == 0

    big, small = [], []
    sr = threading.Event()

    def bw(): big.append(s.acquire(3, timeout=0.8))
    def sw():
        ok = s.acquire(1, timeout=0.8)
        small.append(ok)
        if ok:
            sr.wait(timeout=0.5)
            s.release(1)

    tb = threading.Thread(target=bw); tb.start(); time.sleep(0.03)
    ts = threading.Thread(target=sw); ts.start(); time.sleep(0.03)

    # 只 release 1 个
    s.release(1)
    ts.join(timeout=0.3)
    assert bool(small[0]) is True and big == [] and s.available == 0
    sr.set(); time.sleep(0.02)
    s.release(2)
    tb.join(timeout=0.3); ts.join(timeout=0.3)
    assert bool(big[0]) is True
    s.release(3)
    assert s.available == 3 and s.borrowed == 0
    print("  小请求绕过：大在前没资源，小在后够资源先走 ✓")
    print("PASS (unfair small bypass)")


# ============================================================
# 测试8：动态容量调整 + release 对齐
# ============================================================
def test_dynamic_capacity_adjust():
    print("=== test_dynamic_capacity_adjust ===")
    s = TimedSemaphore(3)
    s.adjust_capacity(+5)
    assert s.capacity == 8 and s.available == 8
    s.acquire(2); s.acquire(6)
    assert s.available == 0 and s.borrowed == 8
    s.adjust_capacity(-6)
    assert s.capacity == 2 and s.available == 0
    s.release(8)  # 全部归还
    assert s.available == 2  # 自动对齐
    assert s.borrowed == 0
    s.set_capacity(10)
    assert s.capacity == 10 and s.available == 10
    s.set_capacity(0)
    assert s.capacity == 0 and s.available == 0 and s.borrowed == 0
    s.set_capacity(3)
    assert s.capacity == 3 and s.available == 3
    print("  release 自动对齐 ✓")
    print("PASS (dynamic capacity)")


def test_capacity_expand_wakes_waiters():
    print("=== test_capacity_expand_wakes_waiters ===")
    s = TimedSemaphore(0)
    r = []
    def w(): r.append(s.acquire(2, timeout=1.0))
    t = threading.Thread(target=w); t.start(); time.sleep(0.03)
    s.adjust_capacity(+2)
    t.join(timeout=0.3)
    assert r and r[0]
    s.release(2)
    assert s.available == 2
    print("PASS")


# ============================================================
# 测试9：负数容量校验
# ============================================================
def test_negative_capacity_validation():
    print("=== test_negative_capacity_validation ===")
    s = TimedSemaphore(5)
    s.acquire(3)
    try:
        s.set_capacity(-1); assert False
    except ValueError as e:
        assert ">= 0" in str(e) and "outstanding borrowed" in str(e)
    try:
        s.adjust_capacity(-10); assert False
    except ValueError as e:
        assert "would result in capacity" in str(e) and "< 0" in str(e)
    assert s.capacity == 5 and s.borrowed == 3
    try:
        s.set_capacity("5"); assert False
    except TypeError:
        pass
    s.set_capacity(0)
    assert s.capacity == 0 and s.borrowed == 3
    s.release(3)
    assert s.available == 0
    print("PASS (negative validation)")


# ============================================================
# 测试10：release 对齐
# ============================================================
def test_release_aligns_capacity():
    print("=== test_release_aligns_capacity ===")
    s = TimedSemaphore(10)
    for _ in range(10): s.acquire()
    s.set_capacity(2)
    assert s.capacity == 2 and s.available == 0
    s.release(5)
    assert s.available == 2 and s.borrowed == 5
    s.release(5)
    assert s.available == 2 and s.borrowed == 0
    print("  release(5) cap=2 → av=2; release(5) → av=2 ✓")
    print("PASS")


# ============================================================
# 测试11：等待视图（含 request_id / name）
# ============================================================
def test_inspect_waiters_with_ids():
    print("=== test_inspect_waiters_with_ids ===")
    # ---- 公平模式：队首大请求堵住 ----
    s = TimedSemaphore(0, fair=True)
    s.set_capacity(5)
    for _ in range(5): s.acquire()

    r1, r2, r3 = [], [], []
    e1, e2, e3 = threading.Event(), threading.Event(), threading.Event()

    def w1(): e1.set(); r1.append(s.acquire(5, timeout=0.5, request_id="big-1", name="bulk-loader"))
    def w2(): e2.set(); r2.append(s.acquire(2, timeout=0.5, request_id="med-1", name="reconcile"))
    def w3(): e3.set(); r3.append(s.acquire(1, timeout=0.5, request_id="sml-1", name="health-check"))

    t1 = threading.Thread(target=w1); t1.start(); e1.wait(); time.sleep(0.02)
    t2 = threading.Thread(target=w2); t2.start(); e2.wait(); time.sleep(0.02)
    t3 = threading.Thread(target=w3); t3.start(); e3.wait(); time.sleep(0.02)

    view = s.inspect_waiters()
    assert len(view) == 3
    assert view[0]["request_id"] == "big-1" and view[0]["name"] == "bulk-loader"
    assert view[0]["is_head"] is True and view[0]["permits"] == 5
    assert view[1]["request_id"] == "med-1" and view[1]["permits"] == 2
    assert view[2]["request_id"] == "sml-1" and view[2]["permits"] == 1

    # 公平模式：release(3) 不够队首 5，全继续等
    s.release(3)
    time.sleep(0.05)
    view2 = s.inspect_waiters()
    assert len(view2) == 3
    assert r1 == [] and r2 == [] and r3 == []
    print("  公平模式：队首堵住，request_id/name 显示正确 ✓")

    # ---- 非公平模式：小请求绕过 ----
    s2 = TimedSemaphore(0, fair=False)
    s2.set_capacity(5)
    for _ in range(5): s2.acquire()
    rr1, rr2, rr3 = [], [], []
    ee1, ee2, ee3 = threading.Event(), threading.Event(), threading.Event()
    sr_ev = threading.Event()
    def ww1(): ee1.set(); rr1.append(s2.acquire(5, timeout=0.5, request_id="BIG"))
    def ww2():
        ee2.set(); ok = s2.acquire(2, timeout=0.5, request_id="MED")
        rr2.append(ok)
        if ok: sr_ev.wait(timeout=0.4); s2.release(2)
    def ww3():
        ee3.set(); ok = s2.acquire(1, timeout=0.5, request_id="SML")
        rr3.append(ok)
        if ok: sr_ev.wait(timeout=0.4); s2.release(1)

    tt1 = threading.Thread(target=ww1); tt1.start(); ee1.wait(); time.sleep(0.02)
    tt2 = threading.Thread(target=ww2); tt2.start(); ee2.wait(); time.sleep(0.02)
    tt3 = threading.Thread(target=ww3); tt3.start(); ee3.wait(); time.sleep(0.02)

    s2.release(3)
    time.sleep(0.05)
    v = s2.inspect_waiters()
    assert len(v) == 1 and v[0]["request_id"] == "BIG"
    assert rr1 == [] and bool(rr2[0]) is True and bool(rr3[0]) is True
    print("  非公平模式：MED+SML 绕过 BIG 先走 ✓")

    # 清理
    sr_ev.set(); time.sleep(0.03)
    s2.release(5)
    tt1.join(timeout=0.3); tt2.join(timeout=0.3); tt3.join(timeout=0.3)
    assert bool(rr1[0]) is True
    s2.release(8)
    assert s2.available == 5

    # 清理公平模式 s
    s.release(5)
    t1.join(timeout=0.3); t2.join(timeout=0.3); t3.join(timeout=0.3)
    assert bool(r1[0]) is True
    s.release(8)
    assert s.available == 5

    print("PASS (inspect waiters with ids)")


# ============================================================
# 测试12：取消等待 cancel_wait
# ============================================================
def test_cancel_wait_by_id():
    print("=== test_cancel_wait_by_id ===")
    s = TimedSemaphore(0, fair=False)
    s.set_capacity(5)
    for _ in range(5): s.acquire()

    # 启动 4 个等待者：2 个 request_id="group-a"，1 个 "group-b"，1 个无 ID
    results = []
    evs = [threading.Event() for _ in range(4)]
    def wk(i, rid, name):
        evs[i].set()
        res = s.acquire(timeout=5.0, request_id=rid, name=name)
        results.append((i, res))

    ids = ["group-a", "group-a", "group-b", None]
    names = ["A-1", "A-2", "B-1", "anon"]
    threads = [threading.Thread(target=wk, args=(i, ids[i], names[i])) for i in range(4)]
    for i, th in enumerate(threads):
        th.start(); evs[i].wait(timeout=0.5); time.sleep(0.015)

    assert s.queue_length == 4
    view = s.inspect_waiters()
    assert [v["request_id"] for v in view] == ["group-a", "group-a", "group-b", None]

    # ---- 取消 group-a 的 2 个 ----
    n = s.cancel_wait("group-a")
    assert n == 2, f"应取消 2 个，实际 {n}"
    # 等那 2 个线程醒来返回
    time.sleep(0.05)
    # 剩下 group-b 和 无 ID 的共 2 个
    assert s.queue_length == 2

    # ---- 取消不存在的 ID ----
    n2 = s.cancel_wait("not-exist")
    assert n2 == 0

    # ---- 取消 group-b ----
    n3 = s.cancel_wait("group-b")
    assert n3 == 1
    time.sleep(0.05)
    assert s.queue_length == 1

    # ---- 无 request_id 的取消不了（因为 cancel_wait 只按精确字符串匹配） ----
    # 传空串会抛 ValueError（不允许），直接关闭闸门清掉最后一个
    s.close()
    time.sleep(0.05)
    assert s.queue_length == 0

    # ---- 验证所有结果 ----
    for th in threads: th.join(timeout=0.5)
    assert len(results) == 4
    cancelled_reasons = 0
    closed_reasons = 0
    for i, res in results:
        assert res.ok is False
        if res.reason == AcquireFailReason.CANCELLED:
            cancelled_reasons += 1
        elif res.reason == AcquireFailReason.CLOSED:
            closed_reasons += 1
    assert cancelled_reasons == 3, f"应有 3 个 cancelled，实际 {cancelled_reasons}"
    assert closed_reasons == 1, f"应有 1 个 closed，实际 {closed_reasons}"

    # ---- 取消空串应抛 ValueError ----
    try:
        s.cancel_wait("")
        assert False
    except ValueError:
        pass
    try:
        s.cancel_wait(None)  # type: ignore
        assert False
    except ValueError:
        pass

    s.open()
    s.release(5)
    s.set_capacity(5)
    assert s.available == 5

    print(f"  cancel('group-a')={n}, cancel('group-b')={n3}, 3 CANCELLED + 1 CLOSED ✓")
    print("PASS (cancel_wait by id)")


# ============================================================
# 测试13：事件流水
# ============================================================
def test_event_log_and_summary():
    print("=== test_event_log_and_summary ===")
    s = TimedSemaphore(2, event_log_size=50)  # 小容量便于测试
    # 做一系列操作
    r1 = s.acquire(name="op1")
    r2 = s.acquire(name="op2")
    r3 = s.acquire(timeout=0.01, name="op3")  # 超时
    s.release()
    s.adjust_capacity(+3)  # 2→5
    s.set_capacity(3)      # 5→3
    s.close()
    r4 = s.acquire(timeout=5.0, name="op4")  # 关闭被拒
    s.open()
    s.cancel_wait("no-such-id")  # 0 个取消
    s.release(3)

    # 查看日志
    log = s.get_event_log()
    assert len(log) > 0
    types = [e["type"] for e in log]
    assert "acquire_success" in types and "acquire_fail" in types
    assert "release" in types and "adjust_capacity" in types
    assert "set_capacity" in types and "close" in types and "open" in types
    assert "cancel_wait" in types

    # 查看摘要
    summary = s.get_event_summary()
    assert summary["total_events"] == len(log)
    assert summary["buffer_capacity"] == 50
    assert "counts" in summary and "fail_reasons" in summary

    fail_reasons = summary["fail_reasons"]
    assert "timeout" in fail_reasons and "closed" in fail_reasons

    # limit 测试
    log10 = s.get_event_log(limit=10)
    assert len(log10) == min(10, len(log))

    # 打印示例日志（前 10 条最新的）
    print("  最近 10 条事件：")
    for e in log[:10]:
        print(f"    {e['ts_ms']:>8.2f}ms  {e['type']:<20}  permits={e['permits']:<2d}"
              f"  name={e['name'] or '-':<8}  details={e['details']}")

    print("  事件摘要：", summary)
    print("PASS (event log)")


# ============================================================
# 测试14：可观测统计（含 max_wait_ms / total_cancelled）
# ============================================================
def test_observable_stats_extended():
    print("=== test_observable_stats_extended ===")
    s = TimedSemaphore(2, event_log_size=0)  # 关掉流水
    # 成功 3，超时 1，关闭 1，取消 1
    assert s.acquire() and s.acquire()
    r1 = s.acquire(timeout=0.01)  # timeout
    s.release(); s.acquire(timeout=0.1)  # success #3
    s.close()
    r2 = s.acquire(timeout=5.0)  # closed
    s.open()

    # 启动一个取消的
    s2 = TimedSemaphore(0, event_log_size=0); s2.set_capacity(1); s2.acquire()
    ev = threading.Event()
    res_c = []
    def w():
        ev.set()
        res_c.append(s2.acquire(timeout=5.0, request_id="x"))
    t = threading.Thread(target=w); t.start(); ev.wait(); time.sleep(0.02)
    s2.cancel_wait("x")
    t.join(timeout=0.3)
    assert res_c and not res_c[0]
    assert res_c[0].reason == AcquireFailReason.CANCELLED

    # 验证统计
    st = s.get_stats()
    assert st["total_success"] == 3
    assert st["total_timeout"] == 1
    assert st["total_closed_rejected"] == 1
    assert st["total_cancelled"] == 0  # s 自己没有取消
    assert st["max_wait_ms"] >= 0
    assert st["avg_wait_ms"] >= 0

    st2 = s2.get_stats()
    assert st2["total_cancelled"] == 1

    # 释放 + 验证对齐
    s.release(2)
    s.set_capacity(2)
    assert s.available == 2 and s.borrowed == 0
    s2.release(1)
    s2.set_capacity(1)
    assert s2.available == 1

    print(f"  统计：success={st['total_success']}, timeout={st['total_timeout']}, "
          f"closed={st['total_closed_rejected']}, cancelled={st2['total_cancelled']}, "
          f"max_wait={max(st['max_wait_ms'], st2['max_wait_ms']):.2f}ms ✓")
    print("PASS (extended stats)")


# ============================================================
# 测试15：闸门关闭
# ============================================================
def test_close_gate():
    print("=== test_close_gate ===")
    s = TimedSemaphore(0, fair=False)
    s.set_capacity(5)
    for _ in range(5): s.acquire()

    # 4 个等待者
    r = [[] for _ in range(4)]
    evs = [threading.Event() for _ in range(4)]
    def wk(i, n):
        evs[i].set()
        r[i].append(s.acquire(n, timeout=3.0))
    ts = []
    for i, n in enumerate([1, 1, 1, 2]):
        ts.append(threading.Thread(target=wk, args=(i, n)))
        ts[-1].start(); evs[i].wait(); time.sleep(0.01)

    assert s.queue_length == 4
    s.close()
    assert s.is_closed
    # 等待者都被唤醒
    for i, th in enumerate(ts):
        th.join(timeout=0.3)
        assert len(r[i]) == 1 and bool(r[i][0]) is False, (
            f"#{i} 应因关闭失败，实际 {r[i]}"
        )
        assert r[i][0].reason == AcquireFailReason.CLOSED

    # 新请求立即失败
    t0 = time.monotonic()
    res = s.acquire(timeout=5.0)
    dur = time.monotonic() - t0
    assert not res and res.reason == AcquireFailReason.CLOSED
    assert dur < 0.05

    # 已拿的正常归还
    s.release(5)
    assert s.available == 5

    s.open()
    assert not s.is_closed
    assert s.acquire(3)
    s.release(3)
    assert s.available == 5

    st = s.get_stats()
    assert st["total_closed_rejected"] == 5  # 4 等待 + 1 新请求

    print(f"  closed_rejected={st['total_closed_rejected']} ✓")
    print("PASS (close gate)")


# ============================================================
# 测试16：统计口径对齐（内部 vs 外部）
# ============================================================
def test_stats_alignment():
    """
    很多线程同时失败时，内部四类统计（success+timeout+closed+cancelled）
    必须 = 外部看到的操作数。
    """
    print("=== test_stats_alignment ===")
    N_THREADS = 30
    OPS = 10
    s = TimedSemaphore(3, fair=False)

    start_barrier = threading.Barrier(N_THREADS + 1)  # +1 控制线程
    stop_flag = threading.Event()
    ext_ok = [0]; ext_to = [0]; ext_cl = [0]; ext_cn = [0]
    lock = threading.Lock()

    def worker():
        my_ok = my_to = my_cl = my_cn = 0
        start_barrier.wait()
        for i in range(OPS):
            res = s.acquire(
                timeout=0.03,
                request_id=f"req-{threading.current_thread().name}-{i}",
            )
            if res:
                my_ok += 1
                time.sleep(random.uniform(0.002, 0.008))
                s.release()
            elif res.reason == AcquireFailReason.TIMEOUT:
                my_to += 1
            elif res.reason == AcquireFailReason.CLOSED:
                my_cl += 1
            elif res.reason == AcquireFailReason.CANCELLED:
                my_cn += 1
            # 随机触发一些关闭/取消
            if i == 3 and random.random() < 0.15:
                s.close()
                time.sleep(0.005)
                s.open()
        with lock:
            ext_ok[0] += my_ok
            ext_to[0] += my_to
            ext_cl[0] += my_cl
            ext_cn[0] += my_cn

    threads = [threading.Thread(target=worker, name=f"t{i}") for i in range(N_THREADS)]
    for t in threads: t.start()
    start_barrier.wait()

    # 控制线程：随机做一些 cancel
    def controller():
        start_barrier.wait()
        for _ in range(15):
            time.sleep(random.uniform(0.01, 0.04))
            # 随机取消一个
            if s.queue_length > 0:
                view = s.inspect_waiters()
                ids = [v["request_id"] for v in view if v["request_id"]]
                if ids:
                    rid = random.choice(ids)
                    s.cancel_wait(rid)

    ctrl_th = threading.Thread(target=controller, daemon=True)
    ctrl_th.start()

    for t in threads: t.join()
    stop_flag.set()
    ctrl_th.join(timeout=1.0)

    # 等待 outstanding 归零（所有线程都 release 了）
    deadline = time.monotonic() + 2.0
    while s.borrowed > 0 and time.monotonic() < deadline:
        time.sleep(0.02)

    st = s.get_stats()

    # ---- 口径对齐 ----
    total_ext = ext_ok[0] + ext_to[0] + ext_cl[0] + ext_cn[0]
    total_int = (st["total_success"] + st["total_timeout"]
                 + st["total_closed_rejected"] + st["total_cancelled"])
    print(f"  外部: ok={ext_ok[0]}, to={ext_to[0]}, cl={ext_cl[0]}, cn={ext_cn[0]}, total={total_ext}")
    print(f"  内部: ok={st['total_success']}, to={st['total_timeout']}, "
          f"cl={st['total_closed_rejected']}, cn={st['total_cancelled']}, total={total_int}")
    print(f"  max_wait={st['max_wait_ms']:.2f}ms, avg={st['avg_wait_ms']:.2f}ms")

    assert total_ext == total_int, f"口径对齐失败: ext={total_ext} != int={total_int}"
    assert st["total_success"] == ext_ok[0], (
        f"success 对齐失败: {st['total_success']} != {ext_ok[0]}"
    )
    # 超时/关闭/取消是或的关系，每个线程只记一类，总数对齐即可
    assert (st["total_timeout"] + st["total_closed_rejected"] + st["total_cancelled"]
            == ext_to[0] + ext_cl[0] + ext_cn[0])

    # ---- 守恒 ----
    while s.borrowed > 0:
        s.release(min(3, s.borrowed))
    assert s.available == s.capacity == 3
    print("  口径对齐 & 守恒 ✅")
    print("PASS (stats alignment)")


# ============================================================
# 压力测试：基础版
# ============================================================
def run_stress_test(permits: int, n_threads: int, ops_per_thread: int,
                    work_ms_range: tuple, fair: bool):
    title = f"基础压力 (fair={fair}, permits={permits}, threads={n_threads}, ops/thread={ops_per_thread})"
    print("=" * 64)
    print(f" {title} ")
    print("=" * 64)

    sem = TimedSemaphore(permits, fair=fair)
    w_min, w_max = work_ms_range
    start_barrier = threading.Barrier(n_threads)
    total_ops = n_threads * ops_per_thread

    ext_ok = [0]; ext_to = [0]; ext_cl = [0]; ext_cn = [0]
    lock = threading.Lock()
    t0 = time.monotonic()

    def worker(i):
        my_ok = my_to = my_cl = my_cn = 0
        start_barrier.wait()
        for _ in range(ops_per_thread):
            p = random.choice([1, 1, 1, 2, 3])
            res = sem.acquire(permits=p, timeout=0.15,
                              request_id=f"t{i}-{_}", name=f"thread-{i}")
            if res:
                my_ok += 1
                time.sleep(random.uniform(w_min, w_max))
                sem.release(p)
            elif res.reason == AcquireFailReason.TIMEOUT:
                my_to += 1
            elif res.reason == AcquireFailReason.CLOSED:
                my_cl += 1
            elif res.reason == AcquireFailReason.CANCELLED:
                my_cn += 1
        with lock:
            ext_ok[0] += my_ok; ext_to[0] += my_to
            ext_cl[0] += my_cl; ext_cn[0] += my_cn

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads: t.start()
    for t in threads: t.join()
    t1 = time.monotonic()

    dur = t1 - t0
    st = sem.get_stats()
    total_ok = ext_ok[0]; total_to = ext_to[0]; total_cl = ext_cl[0]; total_cn = ext_cn[0]

    print(f"  总操作数:        {total_ops}")
    print(f"  成功数:          {total_ok}")
    print(f"  超时数:          {total_to}")
    if total_cl > 0: print(f"  关闭拒绝:        {total_cl}")
    if total_cn > 0: print(f"  取消数:          {total_cn}")
    print(f"  成功率:          {100*total_ok/total_ops:.2f}%")
    if total_to > 0: print(f"  超时率:          {100*total_to/total_ops:.2f}%")
    print(f"  总耗时:          {dur:.2f} s")
    print(f"  吞吐:            {total_ops/dur:.1f} ops/s")
    print("-" * 64)
    print(f" Semaphore 内置统计：")
    print(f"   容量:           {st['capacity']}")
    print(f"   可用:           {st['available']}")
    print(f"   等待队列:       {st['queue_length']}")
    print(f"   累计成功:       {st['total_success']}")
    print(f"   累计超时:       {st['total_timeout']}")
    print(f"   累计关闭拒绝:   {st['total_closed_rejected']}")
    print(f"   累计取消:       {st['total_cancelled']}")
    print(f"   平均等待:       {st['avg_wait_ms']:.3f} ms")
    print(f"   最长等待:       {st['max_wait_ms']:.3f} ms")
    print("-" * 64)

    # 守恒
    init = permits
    final_av = st["available"]
    borrowed = st["outstanding"]
    conserved = (final_av + borrowed == init)
    print(f"  守恒校验：")
    print(f"    最终可用:       {final_av}")
    print(f"    线程持有:       {borrowed}")
    print(f"    初始许可:       {init}")
    print(f"    守恒:           {'PASS ✅' if conserved else 'FAIL ❌'}")
    print("-" * 64)

    # 事件流水摘要
    summary = sem.get_event_summary()
    print(f"  事件流水摘要：")
    print(f"    总事件数:       {summary['total_events']} / {summary['buffer_capacity']}")
    print(f"    事件类型:       {summary['counts']}")
    if summary["fail_reasons"]:
        print(f"    失败原因:       {summary['fail_reasons']}")
    print("=" * 64)

    # 校验
    assert st["total_success"] == total_ok
    assert (st["total_timeout"] + st["total_closed_rejected"] + st["total_cancelled"]
            == total_to + total_cl + total_cn)
    assert conserved


# ============================================================
# 压力测试：动态场景（扩容/缩容/关闭/取消排队 混合）
# ============================================================
def run_stress_test_dynamic(permits: int, n_threads: int, ops_per_thread: int,
                            work_ms_range: tuple, fair: bool):
    title = (f"动态压力 (fair={fair}, permits={permits}, "
             f"threads={n_threads}, ops/thread={ops_per_thread})")
    print("=" * 64)
    print(f" {title} ")
    print("=" * 64)

    sem = TimedSemaphore(permits, fair=fair)
    w_min, w_max = work_ms_range
    start_barrier = threading.Barrier(n_threads + 1)
    stop_ctrl = threading.Event()
    total_ops = n_threads * ops_per_thread
    history = []

    ext_ok = [0]; ext_to = [0]; ext_cl = [0]; ext_cn = [0]
    lock = threading.Lock()
    t0 = time.monotonic()

    def worker(i):
        my_ok = my_to = my_cl = my_cn = 0
        start_barrier.wait()
        for opn in range(ops_per_thread):
            p = random.choice([1, 1, 1, 2, 3])
            rid = f"t{i}-op{opn}"
            res = sem.acquire(permits=p, timeout=0.2,
                              request_id=rid, name=f"task-{i}")
            if res:
                my_ok += 1
                time.sleep(random.uniform(w_min, w_max))
                try:
                    sem.release(p)
                except Exception:
                    pass
            elif res.reason == AcquireFailReason.TIMEOUT:
                my_to += 1
            elif res.reason == AcquireFailReason.CLOSED:
                my_cl += 1
            elif res.reason == AcquireFailReason.CANCELLED:
                my_cn += 1
        with lock:
            ext_ok[0] += my_ok; ext_to[0] += my_to
            ext_cl[0] += my_cl; ext_cn[0] += my_cn

    def controller():
        start_barrier.wait()
        while not stop_ctrl.is_set():
            time.sleep(random.uniform(0.02, 0.06))
            op = random.choices(
                ["expand", "shrink", "close", "open", "cancel", "noop"],
                weights=[3, 3, 1, 1, 2, 2]
            )[0]
            try:
                if op == "expand":
                    d = random.choice([1, 1, 2, 3])
                    sem.adjust_capacity(+d); history.append(f"+{d}")
                elif op == "shrink":
                    d = random.choice([1, 1, 2])
                    try:
                        sem.adjust_capacity(-d); history.append(f"-{d}")
                    except ValueError:
                        history.append("-X")
                elif op == "close":
                    if not sem.is_closed:
                        sem.close(); history.append("C")
                elif op == "open":
                    if sem.is_closed:
                        sem.open(); history.append("O")
                elif op == "cancel":
                    view = sem.inspect_waiters()
                    ids = [v["request_id"] for v in view if v["request_id"]]
                    if ids:
                        rid = random.choice(ids)
                        n_c = sem.cancel_wait(rid)
                        history.append(f"X{n_c}")
                    else:
                        history.append(".")
                else:
                    history.append(".")
            except Exception as e:
                history.append(f"E({e})")
        if sem.is_closed:
            sem.open()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    ctrl_th = threading.Thread(target=controller, daemon=True)
    for t in threads: t.start()
    ctrl_th.start()
    for t in threads: t.join()
    stop_ctrl.set()
    ctrl_th.join(timeout=1.0)
    t1 = time.monotonic()

    dur = t1 - t0
    st = sem.get_stats()
    total_ok = ext_ok[0]; total_to = ext_to[0]
    total_cl = ext_cl[0]; total_cn = ext_cn[0]

    print(f"  总操作数:        {total_ops}")
    print(f"  成功数:          {total_ok}")
    print(f"  超时数:          {total_to}")
    print(f"  关闭拒绝:        {total_cl}")
    print(f"  取消排队:        {total_cn}")
    print(f"  成功率:          {100*total_ok/total_ops:.2f}%")
    print(f"  总耗时:          {dur:.2f} s")
    print(f"  吞吐:            {total_ops/dur:.1f} ops/s")
    print("-" * 64)
    print(f"  动态操作:        {len(history)} 次 (示例: {''.join(history[:16])}...)")
    print("-" * 64)
    print(f" Semaphore 内置统计：")
    print(f"   容量:           {st['capacity']}")
    print(f"   可用:           {st['available']}")
    print(f"   等待队列:       {st['queue_length']}")
    print(f"   累计成功:       {st['total_success']}")
    print(f"   累计超时:       {st['total_timeout']}")
    print(f"   累计关闭拒绝:   {st['total_closed_rejected']}")
    print(f"   累计取消:       {st['total_cancelled']}")
    print(f"   平均等待:       {st['avg_wait_ms']:.3f} ms")
    print(f"   最长等待:       {st['max_wait_ms']:.3f} ms")
    print("-" * 64)

    # 守恒：归还所有借出后 av == capacity
    final_av = st["available"]
    final_bo = st["outstanding"]
    final_cap = st["capacity"]
    assert final_av <= final_cap, f"av={final_av} > cap={final_cap}"
    while sem.borrowed > 0:
        sem.release(min(permits, sem.borrowed))
    after_av = sem.available
    conserved = (after_av == final_cap)

    print(f"  守恒校验（动态容量）：")
    print(f"    最终容量:       {final_cap}")
    print(f"    归还后可用:     {after_av}")
    print(f"    (归还前可用:    {final_av}, 未归还: {final_bo})")
    print(f"    守恒:           {'PASS ✅' if conserved else 'FAIL ❌'}")
    print("-" * 64)

    # 事件流水摘要
    summary = sem.get_event_summary()
    print(f"  事件流水摘要：")
    print(f"    总事件数:       {summary['total_events']} / {summary['buffer_capacity']}")
    print(f"    事件类型:       {summary['counts']}")
    if summary["fail_reasons"]:
        print(f"    失败原因:       {summary['fail_reasons']}")
    print("=" * 64)

    # 口径对齐
    assert st["total_success"] == total_ok
    assert (st["total_timeout"] + st["total_closed_rejected"] + st["total_cancelled"]
            == total_to + total_cl + total_cn)
    assert conserved


# ============================================================
# CLI
# ============================================================
def run_all_unit_tests():
    tests = [
        test_basic_unfair, test_basic_fair,
        test_acquire_result_carries_reason, test_strict_timeout_boundary,
        test_fair_fifo_ordering,
        test_multi_permit_atomic, test_multi_permit_no_partial_steal,
        test_no_leak_concurrent,
        test_ctx_single, test_ctx_multi,
        test_unfair_small_bypass_large,
        test_dynamic_capacity_adjust, test_capacity_expand_wakes_waiters,
        test_negative_capacity_validation, test_release_aligns_capacity,
        test_inspect_waiters_with_ids,
        test_cancel_wait_by_id,
        test_event_log_and_summary,
        test_observable_stats_extended,
        test_close_gate,
        test_stats_alignment,
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
        print(f"==== {fail}/{len(tests)} 个失败 ====\n")
    return fail == 0


def main():
    args = sys.argv[1:]
    if not args:
        ok = run_all_unit_tests()
        if not ok: sys.exit(1)
        print("\n================ 基础压力（非公平） ================\n")
        run_stress_test(5, 10, 8, (0.005, 0.04), fair=False)
        print("\n================ 基础压力（公平） ================\n")
        run_stress_test(5, 10, 8, (0.005, 0.04), fair=True)
        print("\n================ 动态压力（非公平） ================\n")
        run_stress_test_dynamic(5, 10, 8, (0.005, 0.04), fair=False)
        print("\n================ 动态压力（公平） ================\n")
        run_stress_test_dynamic(5, 10, 8, (0.005, 0.04), fair=True)
        return

    mode = args[0].lower()
    if mode == "stress":
        fair = len(args) >= 2 and args[1].lower() == "fair"
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
