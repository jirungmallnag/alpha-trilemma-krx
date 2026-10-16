#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit.py — 데이터 감사 (사전등록 2절 + 3장 사전 점검)

분석 결과가 아니라 '데이터가 믿을 만한가'만 봅니다. 그래서 사전등록 전후 언제 돌려도 됩니다.
Quantit 발표: "결과를 믿기 전에 데이터부터 의심한다."

    py audit.py              # 유가증권 + ETF 감사
    py audit.py --only kospi
    py audit.py --only etf

결과:
    화면 요약
    output/audit/audit_report.json        숫자 요약 (GitHub에 올려도 되는 수준)
    output/audit/*.csv                    문제 행 목록 (원데이터 일부라 GitHub 금지 — .gitignore가 막음)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

DATA = "./data"
OUT = "./output/audit"
LIMIT_CHANGE = pd.Timestamp("2015-06-15")     # 가격제한폭 ±15% → ±30%
TOL = 0.005                                     # 호가 단위 반올림 여유 0.5%p


def read(name: str, cols: list[str]) -> pd.DataFrame:
    folder = os.path.join(DATA, name)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(os.path.join(folder, f))
        frames.append(df[[c for c in cols if c in df.columns]])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["BAS_DD"], format="%Y%m%d")
    for c in ("ISU_CD", "ISU_NM"):
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def line(title):
    print("\n" + "═" * 78 + f"\n  {title}\n" + "═" * 78)


# ══════════════════════════════════════════════════════════════════
# 유가증권
# ══════════════════════════════════════════════════════════════════

def audit_kospi(rep: dict):
    line("유가증권 일별 — 사전등록 2절 감사")
    df = read("kospi", ["BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC",
                        "CMPPREVDD_PRC", "FLUC_RT", "LIST_SHRS"])
    df = df.sort_values(["ISU_CD", "date"]).reset_index(drop=True)
    r = {}

    # ── 0. 규모
    per_day = df.groupby("date").size()
    r["rows"] = int(len(df))
    r["days"] = int(per_day.size)
    r["tickers"] = int(df["ISU_CD"].nunique())
    r["rows_per_day"] = {"min": int(per_day.min()), "median": int(per_day.median()),
                         "max": int(per_day.max())}
    print(f"  {len(df):,}행 · 거래일 {per_day.size:,}일 · 종목 {r['tickers']:,}개 · "
          f"하루 종목 수 {per_day.min()}~{per_day.max()} (중앙값 {int(per_day.median())})")
    thin = per_day[per_day < per_day.median() * 0.8]
    r["thin_days"] = [d.strftime("%Y-%m-%d") for d in thin.index]
    if len(thin):
        print(f"  △ 종목 수가 평소의 80% 미만인 날 {len(thin)}일: "
              f"{', '.join(r['thin_days'][:5])}{' ...' if len(thin) > 5 else ''}")

    # ── 1. 일별 수익률 = 전일대비 ÷ (종가 − 전일대비)
    base = df["TDD_CLSPRC"] - df["CMPPREVDD_PRC"]
    ret = df["CMPPREVDD_PRC"] / base.where(base > 0)
    df["ret"] = ret
    traded = df["TDD_CLSPRC"] > 0
    r["ret_missing"] = int((ret.isna() & traded).sum())
    print(f"\n  ① 일별 수익률 계산 불가(기준가 0 이하·빈값): {r['ret_missing']:,}행")

    # 거래소 등락률과 대조 — 같은 기준가를 쓰는지
    if "FLUC_RT" in df.columns:
        diff = (ret - df["FLUC_RT"] / 100).abs()
        bad = (diff > 0.0001) & ret.notna() & df["FLUC_RT"].notna()
        r["ret_vs_flucrt_mismatch"] = int(bad.sum())
        print(f"     거래소 등락률과 0.01%p 넘게 다른 행: {bad.sum():,}행 "
              f"({bad.mean():.4%})  ← 0에 가까우면 계산식이 거래소와 같다는 뜻")

    # ── 2. 가격제한폭 밖
    limit = np.where(df["date"] < LIMIT_CHANGE, 0.15, 0.30) + TOL
    first_day = df.groupby("ISU_CD", observed=True).cumcount() == 0
    out = (ret.abs() > limit) & ret.notna()
    r["over_limit"] = int((out & ~first_day).sum())
    r["over_limit_first_day"] = int((out & first_day).sum())
    print(f"\n  ② 가격제한폭(2015-06-15 전 ±15%, 후 ±30%) 밖")
    print(f"     상장 첫 거래일 제외: {r['over_limit']:,}행   (첫 거래일: {r['over_limit_first_day']:,}행)")
    os.makedirs(OUT, exist_ok=True)
    viol = df[out & ~first_day].assign(abs_ret=lambda x: x["ret"].abs()) \
        .sort_values("abs_ret", ascending=False)
    if len(viol):
        viol[["date", "ISU_CD", "ISU_NM", "TDD_CLSPRC", "CMPPREVDD_PRC", "ret"]] \
            .to_csv(os.path.join(OUT, "kospi_over_limit.csv"), index=False, encoding="utf-8-sig")
        for _, x in viol.head(5).iterrows():
            print(f"       {x['date']:%Y-%m-%d} {x['ISU_CD']} {str(x['ISU_NM'])[:12]:<12} {x['ret']:+.1%}")

    # ── 3. 분할·병합 후보일에 기준가 가정이 맞는가
    g = df.groupby("ISU_CD", observed=True)
    share_ratio = df["LIST_SHRS"] / g["LIST_SHRS"].shift(1)
    raw_ratio = df["TDD_CLSPRC"] / g["TDD_CLSPRC"].shift(1)
    cand = (share_ratio >= 1.5) | (share_ratio <= 1 / 1.5)
    cand &= share_ratio.notna() & raw_ratio.notna()
    c = df[cand].assign(share_ratio=share_ratio[cand], raw_ratio=raw_ratio[cand])
    c["raw_ret"] = c["raw_ratio"] - 1
    c["ret_ok"] = c["ret"].abs() <= np.where(c["date"] < LIMIT_CHANGE, 0.15, 0.30) + TOL
    c["raw_broken"] = (c["raw_ret"].abs() > 0.5)
    r["split_candidates"] = int(len(c))
    r["split_raw_broken"] = int(c["raw_broken"].sum())
    r["split_raw_broken_ret_ok"] = int((c["raw_broken"] & c["ret_ok"]).sum())
    print(f"\n  ③ 상장주식수가 1.5배 이상 변한 날: {len(c):,}건")
    print(f"     그중 종가가 50% 넘게 끊긴 날(분할·병합): {r['split_raw_broken']:,}건")
    print(f"     그중 '전일대비' 수익률은 정상 범위인 날: {r['split_raw_broken_ret_ok']:,}건")
    if r["split_raw_broken"]:
        share = r["split_raw_broken_ret_ok"] / r["split_raw_broken"]
        r["split_assumption_holds_share"] = share
        verdict = "가정이 맞습니다 — 전일대비 수익률을 그대로 써도 됩니다" if share > 0.95 \
            else "가정이 일부 틀립니다 — 상장주식수 보정이 필요합니다 (사전등록 변경 기록에 남길 것)"
        print(f"     → {share:.0%}  {verdict}")
    c[["date", "ISU_CD", "ISU_NM", "share_ratio", "raw_ret", "ret", "ret_ok"]] \
        .to_csv(os.path.join(OUT, "kospi_split_days.csv"), index=False, encoding="utf-8-sig")

    # ── 4. 연말 배당락 갭 (보정하지 않으므로 크기만 기록)
    print(f"\n  ④ 연말 배당락 갭 — 12월 마지막 5거래일 중 종목 수익률 중앙값이 가장 낮은 날")
    daily_med = df.groupby("date")["ret"].median()
    normal = daily_med.median()
    rows = []
    for y in sorted(daily_med.index.year.unique()):
        dec = daily_med[(daily_med.index.year == y) & (daily_med.index.month == 12)].tail(5)
        if len(dec) == 0:
            continue
        d = dec.idxmin()
        rows.append({"year": int(y), "date": d.strftime("%Y-%m-%d"), "median_ret": float(dec.min())})
        print(f"     {y}  {d:%m-%d}  {dec.min():+.2%}")
    r["year_end_gap"] = rows
    r["median_daily_ret_all_days"] = float(normal)
    print(f"     (참고: 전체 거래일의 종목 수익률 중앙값 {normal:+.3%})")
    print("     배당은 보정하지 않습니다. 12월 음의 갭이 매년 반복되면 '가격수익률'이라는 한계로 보고합니다.")

    rep["kospi"] = r


# ══════════════════════════════════════════════════════════════════
# ETF
# ══════════════════════════════════════════════════════════════════

def audit_etf(rep: dict):
    line("ETF 일별 — 3장 사전 점검 (괴리율의 원재료)")
    df = read("etf", ["BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC", "NAV",
                      "LIST_SHRS", "IDX_IND_NM", "OBJ_STKPRC_IDX", "ACC_TRDVAL"])
    r = {"rows": int(len(df)), "etfs": int(df["ISU_CD"].nunique())}
    print(f"  {len(df):,}행 · ETF {r['etfs']:,}개")

    # ── 1. NAV 채움
    yr = df["date"].dt.year
    has_nav = df["NAV"] > 0
    cov = has_nav.groupby(yr).mean()
    r["nav_coverage_by_year"] = {int(k): float(v) for k, v in cov.items()}
    print("\n  ① NAV가 0보다 큰 행의 비율")
    print("     " + "  ".join(f"{y}:{v:.0%}" for y, v in cov.items()))

    # ── 2. 괴리율 = (종가 − NAV) ÷ NAV
    ok = has_nav & (df["TDD_CLSPRC"] > 0)
    prem = (df.loc[ok, "TDD_CLSPRC"] - df.loc[ok, "NAV"]) / df.loc[ok, "NAV"]
    df.loc[ok, "premium"] = prem
    a = prem.abs()
    by_year = a.groupby(df.loc[ok, "date"].dt.year)
    r["abs_premium_by_year"] = {
        int(y): {"median": float(s.median()), "p99": float(s.quantile(0.99))} for y, s in by_year}
    print("\n  ② |괴리율| 연도별 중앙값 · 99백분위")
    for y, s in by_year:
        print(f"     {y}  {s.median():.3%}  ·  {s.quantile(0.99):.2%}")

    # 단위 오류 의심: 종가와 NAV가 10배 이상 차이
    unit = ok & ((df["TDD_CLSPRC"] / df["NAV"] > 10) | (df["TDD_CLSPRC"] / df["NAV"] < 0.1))
    r["suspected_unit_errors"] = int(unit.sum())
    big = ok & (df["premium"].abs() > 0.05) & ~unit
    r["abs_premium_over_5pct"] = int(big.sum())
    print(f"\n  ③ 종가와 NAV가 10배 넘게 다른 행(단위 오류 의심): {r['suspected_unit_errors']:,}")
    print(f"     |괴리율| 5% 초과(단위 오류 제외): {r['abs_premium_over_5pct']:,}행")
    os.makedirs(OUT, exist_ok=True)
    ext = df[big | unit].assign(abs_p=lambda x: x["premium"].abs()) \
        .sort_values("abs_p", ascending=False)
    ext[["date", "ISU_CD", "ISU_NM", "TDD_CLSPRC", "NAV", "premium", "IDX_IND_NM"]] \
        .to_csv(os.path.join(OUT, "etf_extreme_premium.csv"), index=False, encoding="utf-8-sig")
    for _, x in ext.head(5).iterrows():
        print(f"       {x['date']:%Y-%m-%d} {str(x['ISU_NM'])[:18]:<18} 종가 {x['TDD_CLSPRC']:,.0f} "
              f"NAV {x['NAV']:,.2f}  {x['premium']:+.1%}")

    # ── 3. 상장좌수 급변 (자금흐름 계산 전에 분할·병합을 걸러야 함)
    df = df.sort_values(["ISU_CD", "date"])
    sr = df["LIST_SHRS"] / df.groupby("ISU_CD", observed=True)["LIST_SHRS"].shift(1)
    jump = (sr >= 3) | (sr <= 1 / 3)
    r["list_shares_jumps_3x"] = int(jump.sum())
    print(f"\n  ④ 하루에 상장좌수가 3배 이상 변한 날: {jump.sum():,}건 "
          f"← 분할·병합인지 실제 대량 설정인지 3장 자금흐름 계산 전에 확인")

    # ── 4. 기초지수
    has_idx = df["OBJ_STKPRC_IDX"] > 0
    r["index_coverage"] = float(has_idx.mean())
    print(f"\n  ⑤ 기초지수 종가가 있는 행: {has_idx.mean():.0%} (해외·합성 지수는 비어 있을 수 있음)")

    rep["etf"] = r


def main():
    ap = argparse.ArgumentParser(description="데이터 감사")
    ap.add_argument("--only", choices=["kospi", "etf"])
    args = ap.parse_args()

    rep = {"run_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    if args.only in (None, "kospi"):
        audit_kospi(rep)
    if args.only in (None, "etf"):
        audit_etf(rep)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("\n" + "═" * 78)
    print(f"  저장: {OUT}/audit_report.json  (+ 문제 행 목록 csv)")
    print("  화면에 나온 내용을 그대로 보내 주세요.")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    main()
