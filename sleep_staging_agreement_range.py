#!/usr/bin/env python3
"""Achievable range of inter-scorer sleep-staging agreement from EDF annotations.

Two technicians score the same recording.  Starting from the nominal
hypnogram read from an EDF+ annotation file, every segment boundary may be
shifted by each technician within an allowed time window.  The stage order
is fixed and every segment has a minimum duration.  Because the minimum
duration constraints couple neighbouring boundaries, the achievable
minimum/maximum agreement is computed with a whole-night dynamic program
instead of picking the most favourable boundary independently per
interval (which would piece together scorings that cannot actually exist).

The denominator of the agreement rate is the time scored by both
technicians; unscored time ("Sleep stage ?") and explicit wake
("Sleep stage W") are reported separately.

Exactness note: with minimum segment duration L and windows W_a, W_b the
desynchronisation regions of two neighbouring boundaries cannot overlap
because L >= W_a + W_b (checked at runtime).  The agreement of a schedule
pair is then exactly the sum of the per-segment synchronised stretches,
which is what the dynamic program optimises.
"""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_right
from pathlib import Path

import numpy as np

from edfio import read_edf

UNSCORED = "?"
WAKE = "W"


def load_segments(path):
    """Return contiguous [(stage, start, end)] merged from EDF annotations."""
    edf = read_edf(path)
    annotations = sorted(edf.annotations, key=lambda a: a.onset)
    segments = []
    for ann in annotations:
        stage = ann.text.strip().removeprefix("Sleep stage ")
        start = ann.onset
        end = ann.onset + (ann.duration or 0.0)
        if segments and segments[-1][0] == stage and segments[-1][2] == start:
            segments[-1] = (stage, segments[-1][1], end)
        elif segments and segments[-1][2] < start:  # gap -> unscored
            segments.append((UNSCORED, segments[-1][2], start))
            segments.append((stage, start, end))
        elif segments and segments[-1][2] > start:
            raise ValueError(f"overlapping annotations at {start}")
        else:
            segments.append((stage, start, end))
    return segments


def _prefix_opt(values, mode):
    """2D prefix optimum of `values`; returns (optimum, argmax/argmin)."""
    ga, gb = values.shape
    pref = np.empty_like(values)
    pref_arg = np.zeros((ga, gb, 2), dtype=np.int64)
    take_max = mode == "max"
    for ia in range(ga):
        for ib in range(gb):
            best_val = values[ia, ib]
            best_arg = (ia, ib)
            if ia > 0:
                cand = pref[ia - 1, ib]
                if (take_max and cand > best_val) or (not take_max and cand < best_val):
                    best_val, best_arg = cand, tuple(pref_arg[ia - 1, ib])
            if ib > 0:
                cand = pref[ia, ib - 1]
                if (take_max and cand > best_val) or (not take_max and cand < best_val):
                    best_val, best_arg = cand, tuple(pref_arg[ia, ib - 1])
            pref[ia, ib] = best_val
            pref_arg[ia, ib] = best_arg
    return pref, pref_arg


def optimize_agreement(stages, times, window_a, window_b, min_dur, mode):
    """Whole-night DP for min/max agreement; returns (value, bounds_a, bounds_b).

    stages: stage label per segment (n_seg).
    times:  boundary times t_0..t_n_seg in integer time units.
    Offsets are integer units within [-window, window] per technician.
    """
    n_seg = len(stages)
    n_bnd = n_seg - 1
    off_a = list(range(-window_a, window_a + 1))
    off_b = list(range(-window_b, window_b + 1))
    ga, gb = len(off_a), len(off_b)
    invalid = -math.inf if mode == "max" else math.inf
    scored = [stage != UNSCORED for stage in stages]

    # dp[ia, ib]: optimal agreement of stretches left of and including the
    # region at the current boundary, ending with offsets (off_a[ia], off_b[ib]).
    dp = np.full((ga, gb), invalid)
    for ia, oa in enumerate(off_a):
        for ib, ob in enumerate(off_b):
            if (
                times[1] + oa - times[0] >= min_dur
                and times[1] + ob - times[0] >= min_dur
            ):
                dp[ia, ib] = (
                    (times[1] + min(oa, ob) - times[0]) if scored[0] else 0.0
                )

    backpointers = []
    for j in range(1, n_bnd):
        wj = 1.0 if scored[j] else 0.0
        # transition j -> j+1 adds stretch of segment j:
        #   wj * (min(pos' ) - max(pos))  =  wj*min' term  +  (dp - wj*max term)
        move = np.full((ga, gb), invalid)
        for ia, oa in enumerate(off_a):
            for ib, ob in enumerate(off_b):
                if dp[ia, ib] != invalid:
                    move[ia, ib] = dp[ia, ib] - wj * (times[j] + max(oa, ob))
        pref, pref_arg = _prefix_opt(move, mode)
        # min duration of segment j:  off' >= off + (min_dur - gap)
        d = min_dur - (times[j + 1] - times[j])
        new_dp = np.full((ga, gb), invalid)
        new_arg = np.zeros((ga, gb, 2), dtype=np.int64)
        for ia, oa in enumerate(off_a):
            xa = min(ia - d, ga - 1)
            if xa < 0:
                continue
            for ib, ob in enumerate(off_b):
                xb = min(ib - d, gb - 1)
                if xb < 0 or pref[xa, xb] == invalid:
                    continue
                new_dp[ia, ib] = wj * (times[j + 1] + min(oa, ob)) + pref[xa, xb]
                new_arg[ia, ib] = pref_arg[xa, xb]
        dp = new_dp
        backpointers.append(new_arg)

    w_last = 1.0 if scored[-1] else 0.0
    best_val, best_pos = invalid, None
    for ia, oa in enumerate(off_a):
        for ib, ob in enumerate(off_b):
            if dp[ia, ib] == invalid:
                continue
            if (
                times[-1] - (times[-2] + oa) < min_dur
                or times[-1] - (times[-2] + ob) < min_dur
            ):
                continue
            val = dp[ia, ib] + w_last * (times[-1] - times[-2] - max(oa, ob))
            if (mode == "max" and val > best_val) or (mode == "min" and val < best_val):
                best_val, best_pos = val, (ia, ib)
    if best_pos is None:
        raise RuntimeError("boundary constraints are infeasible")

    offsets = [None] * n_bnd
    pos = best_pos
    offsets[n_bnd - 1] = pos
    for j in range(n_bnd - 1, 0, -1):
        pos = tuple(backpointers[j - 1][pos])
        offsets[j - 1] = pos
    bounds_a = [times[j] + off_a[offsets[j - 1][0]] for j in range(1, n_bnd + 1)]
    bounds_b = [times[j] + off_b[offsets[j - 1][1]] for j in range(1, n_bnd + 1)]
    return best_val, bounds_a, bounds_b


def check_constraints(times, bounds, window, min_dur, label):
    for j, pos in enumerate(bounds, start=1):
        assert abs(pos - times[j]) <= window + 1e-9, (label, j, pos)
    prev = times[0]
    for pos in bounds:
        assert pos - prev >= min_dur - 1e-9, (label, prev, pos)
        prev = pos
    assert times[-1] - prev >= min_dur - 1e-9, (label, prev, times[-1])


def evaluate_schedule(stages, t_start, t_end, bounds_a, bounds_b):
    """Exact agreement metrics of a schedule pair, sweeping all boundaries."""
    events = sorted({t_start, t_end, *bounds_a, *bounds_b})
    metrics = {
        "both_unscored": 0.0,
        "one_unscored": 0.0,
        "wake_agree": 0.0,
        "sleep_agree": 0.0,
        "disagree": 0.0,
    }
    per_stage_agree = {}
    for left, right in zip(events, events[1:]):
        if right <= left:
            continue
        stage_a = stages[bisect_right(bounds_a, left)]
        stage_b = stages[bisect_right(bounds_b, left)]
        length = right - left
        if stage_a == UNSCORED and stage_b == UNSCORED:
            metrics["both_unscored"] += length
        elif stage_a == UNSCORED or stage_b == UNSCORED:
            metrics["one_unscored"] += length
        elif stage_a == stage_b == WAKE:
            metrics["wake_agree"] += length
        elif stage_a == stage_b:
            metrics["sleep_agree"] += length
            per_stage_agree[stage_a] = per_stage_agree.get(stage_a, 0.0) + length
        else:
            metrics["disagree"] += length
    metrics["agreement"] = metrics["wake_agree"] + metrics["sleep_agree"]
    metrics["scored_both"] = metrics["agreement"] + metrics["disagree"]
    metrics["per_stage_agree"] = per_stage_agree
    return metrics


def write_schedule_csv(path, stages, times, bounds_a, bounds_b, unit):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "boundary",
                "stage_before",
                "stage_after",
                "nominal_s",
                "tech_a_s",
                "tech_b_s",
                "shift_a_s",
                "shift_b_s",
            ]
        )
        for j, (ba, bb) in enumerate(zip(bounds_a, bounds_b), start=1):
            writer.writerow(
                [
                    j,
                    stages[j - 1],
                    stages[j],
                    times[j] * unit,
                    ba * unit,
                    bb * unit,
                    (ba - times[j]) * unit,
                    (bb - times[j]) * unit,
                ]
            )


def print_metrics(title, metrics, unit):
    agreement = metrics["agreement"]
    scored_both = metrics["scored_both"]
    print(f"  {title}")
    print(f"    一致时长(双方均已评分且一致): {agreement * unit:>10.0f} s")
    print(f"    分母(双方均已评分):           {scored_both * unit:>10.0f} s")
    print(f"    一致率:                       {agreement / scored_both:>10.4%}")
    print(
        f"    其中 明确清醒W一致: {metrics['wake_agree'] * unit:>8.0f} s   "
        f"睡眠期一致: {metrics['sleep_agree'] * unit:>8.0f} s   "
        f"不一致: {metrics['disagree'] * unit:>8.0f} s"
    )
    sleep_den = metrics["sleep_agree"] + metrics["disagree"]
    print(
        f"    排除双方W后睡眠分期一致率:   "
        f"{metrics['sleep_agree'] / sleep_den:>10.4%}  (分母 {sleep_den * unit:.0f} s)"
    )
    print(
        f"    不计入分母: 双方未评分 {metrics['both_unscored'] * unit:.0f} s, "
        f"单方未评分 {metrics['one_unscored'] * unit:.0f} s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypnogram",
        default="tests/TEST_DATA/SC4001EC-Hypnogram.edf",
        help="EDF+ file with 'Sleep stage X' annotations",
    )
    parser.add_argument("--window-a", type=float, default=15.0,
                        help="allowed boundary shift for technician A (s)")
    parser.add_argument("--window-b", type=float, default=15.0,
                        help="allowed boundary shift for technician B (s)")
    parser.add_argument("--min-duration", type=float, default=30.0,
                        help="minimum duration of every segment (s)")
    parser.add_argument("--step", type=float, default=1.0,
                        help="time grid resolution (s)")
    parser.add_argument("--csv-prefix", default="agreement",
                        help="output prefix for the two achieving schedules")
    args = parser.parse_args()

    segments = load_segments(args.hypnogram)
    stages = [s[0] for s in segments]
    t_start, t_end = segments[0][1], segments[-1][2]

    unit = args.step
    times_f = [s[1] for s in segments] + [t_end]
    for x in times_f:
        assert abs(x / unit - round(x / unit)) < 1e-6, "times not on grid"
    times = [int(round(x / unit)) for x in times_f]
    for name, v in [("window-a", args.window_a), ("window-b", args.window_b),
                    ("min-duration", args.min_duration)]:
        assert abs(v / unit - round(v / unit)) < 1e-6, f"{name} not on grid"
    window_a = int(round(args.window_a / unit))
    window_b = int(round(args.window_b / unit))
    min_dur = int(round(args.min_duration / unit))

    if min_dur < window_a + window_b:
        raise SystemExit(
            "require min-duration >= window-a + window-b so that boundary "
            "regions of neighbours cannot overlap"
        )

    n_bnd = len(stages) - 1
    print("== 数据 ==")
    print(f"  文件: {args.hypnogram}")
    print(f"  段数: {len(stages)}  可调整边界数: {n_bnd}  总时长: {(t_end - t_start):.0f} s")
    nominal = evaluate_schedule(stages, times[0], times[-1], times[1:-1], times[1:-1])
    print(
        f"  名义分期: 双方已评分 {nominal['scored_both'] * unit:.0f} s "
        f"(其中双方W {nominal['wake_agree'] * unit:.0f} s), "
        f"未评分? {nominal['both_unscored'] * unit:.0f} s"
    )
    print("== 参数 ==")
    print(
        f"  边界调整窗: 技师A ±{args.window_a:.0f} s, 技师B ±{args.window_b:.0f} s; "
        f"每段最短时长 {args.min_duration:.0f} s; 网格 {args.step:.0f} s"
    )

    results = {}
    for mode, title in [("max", "上界(最大一致)"), ("min", "下界(最小一致)")]:
        value, bounds_a, bounds_b = optimize_agreement(
            stages, times, window_a, window_b, min_dur, mode
        )
        check_constraints(times, bounds_a, window_a, min_dur, f"{mode}/A")
        check_constraints(times, bounds_b, window_b, min_dur, f"{mode}/B")
        metrics = evaluate_schedule(stages, times[0], times[-1], bounds_a, bounds_b)
        assert abs(metrics["agreement"] - value) < 1e-6, (metrics["agreement"], value)
        results[mode] = (bounds_a, bounds_b, metrics)
        print(f"== 一致时长可达范围: {title} ==")
        print_metrics(title, metrics, unit)
        csv_path = f"{args.csv_prefix}_{mode}_schedule.csv"
        write_schedule_csv(csv_path, stages, times, bounds_a, bounds_b, unit)
        deviating = [
            (j, times[j], ba, bb)
            for j, (ba, bb) in enumerate(zip(bounds_a, bounds_b), start=1)
            if ba != times[j] or bb != times[j]
        ]
        print(
            f"    达到该界的边界安排已写入 {csv_path} "
            f"(全部 {n_bnd} 个边界; 其中 {len(deviating)} 个偏离名义位置)"
        )
        for j, tj, ba, bb in deviating[:10]:
            print(
                f"      边界{j:>3} {stages[j-1]}→{stages[j]} 名义{tj * unit:>7.0f} "
                f"A={ba * unit:>7.0f} B={bb * unit:>7.0f}"
            )
        if len(deviating) > 10:
            print(f"      ... 其余 {len(deviating) - 10} 个见 CSV")

    max_agreement = results["max"][2]["agreement"]
    min_agreement = results["min"][2]["agreement"]
    relax = n_bnd * (window_a + window_b)
    print("== 对照(逐边界独立松弛) ==")
    print(
        f"  每个边界独立取极端的松弛下界: "
        f"{(max_agreement - relax) * unit:.0f} s "
        f"(= 上界 {max_agreement * unit:.0f} s - {n_bnd} 边界 x "
        f"{(window_a + window_b) * unit:.0f} s)"
    )
    if abs(min_agreement - (max_agreement - relax)) < 1e-9:
        print(
            "  本例中松弛下界与整晚DP下界相同: 两位技师各自整体均匀平移"
            "即可同时达到所有边界极端, 该分期真实存在(已由约束校验确认)。"
        )
    else:
        print(
            f"  整晚约束使可达下界为 {min_agreement * unit:.0f} s, "
            f"高于松弛下界 {(max_agreement - relax) * unit:.0f} s; "
            f"松弛对应的分期不存在, 不能作为答案。"
        )
    print("== 约束校验 == 通过 (窗口/顺序/每段最短时长, 且DP目标值与逐秒扫描一致)")


if __name__ == "__main__":
    main()
