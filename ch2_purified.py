#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch2_purified.py — 2장 Purified Alpha 판정

prereg/ch2_purified_alpha.md 에 적은 정의를 그대로 구현합니다.
여기 있는 숫자를 바꾸면 그 결과는 '탐색'이지 판정이 아닙니다.

    py ch2_purified.py

필요: data/kospi, data/index (krx_daily.py 로 받은 것). API 호출 없음.
시간: 몇 분 (일별 362만 행).

결과:
    화면 요약
    output/ch2/ch2_results.json     판정표 숫자 (GitHub에 올려도 되는 요약)
    output/ch2/ch2_monthly.csv      전략별 월간 수익률 (.gitignore 대상)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# 사전등록 파라미터 — 커밋된 값. 바꾸지 않습니다.
# ══════════════════════════════════════════════════════════════════
TOP_N = 350
MIN_VALUE_TRADED = 5e8
N_Q = 5
LOOKBACK = 12
DELIST_RET = -0.30
EXCLUDE_KW = ("스팩", "리츠")
COST_RT = 0.00015 + 0.00165 + 2 * 0.001        # 회전율 1단위당 (매수·매도·슬리피지)

TARGET_VOL = 0.12      # H2a  Barroso & Santa-Clara
VOL_DAYS = 126         # H2a·H2b
BEAR_MONTHS = 24       # H2b  Daniel & Moskowitz
DM_MIN_OBS = 36        # H2b  확장창 최소 관측
BHM_WINDOW = 36        # H2c  Blitz, Huij & Martens
NW_LAG = 6
T_PASS = 2.33
EVAL_END = "2025-12"

LIMIT_CHANGE = pd.Timestamp("2015-06-15")
TOL = 0.005

DATA = "./data"
OUT = "./output/ch2"


# ══════════════════════════════════════════════════════════════════
# 1. 데이터
# ══════════════════════════════════════════════════════════════════

def read_folder(name, cols):
    folder = os.path.join(DATA, name)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(os.path.join(folder, f), columns=cols) for f in files],
                   ignore_index=True)
    df["date"] = pd.to_datetime(df["BAS_DD"], format="%Y%m%d")
    return df


def load():
    print("  데이터 읽는 중 ...")
    k = read_folder("kospi", ["BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC",
                              "CMPPREVDD_PRC", "MKTCAP", "ACC_TRDVAL"])
    k = k.drop_duplicates(["date", "ISU_CD"]).sort_values(["ISU_CD", "date"])

    base = k["TDD_CLSPRC"] - k["CMPPREVDD_PRC"]
    ret = k["CMPPREVDD_PRC"] / base.where(base > 0)
    first = k.groupby("ISU_CD").cumcount() == 0
    limit = np.where(k["date"] < LIMIT_CHANGE, 0.15, 0.30) + TOL
    over = (ret.abs() > limit) & ~first & ret.notna()
    k["ret_raw"] = ret
    k["ret_clean"] = ret.mask(over)
    print(f"    가격제한폭 밖 → 결측 처리: {int(over.sum())}행")

    R_clean = k.pivot(index="date", columns="ISU_CD", values="ret_clean")
    R_raw = k.pivot(index="date", columns="ISU_CD", values="ret_raw")
    dates = R_clean.index
    month = dates.to_period("M")
    month_end = pd.Series(dates, index=dates).groupby(month).max()     # 월 → 마지막 거래일
    month_start = pd.Series(dates, index=dates).groupby(month).min()

    ke = k[k["date"].isin(month_end.values)]
    MC = ke.pivot(index="date", columns="ISU_CD", values="MKTCAP")
    VT = ke.pivot(index="date", columns="ISU_CD", values="ACC_TRDVAL")
    MC.index = MC.index.to_period("M")
    VT.index = VT.index.to_period("M")
    names = k.groupby("ISU_CD")["ISU_NM"].last()
    last_date = k.groupby("ISU_CD")["date"].max()

    ix = read_folder("index", ["BAS_DD", "IDX_NM", "CLSPRC_IDX"])
    ix = ix[ix["IDX_NM"].str.strip() == "코스피"].drop_duplicates("date").set_index("date")["CLSPRC_IDX"]
    ix = ix.reindex(dates).ffill()
    if ix.isna().all():
        sys.exit("[오류] index 데이터에서 '코스피' 지수를 찾지 못했습니다.")

    print(f"    {len(dates):,}거래일 · {R_clean.shape[1]:,}종목 · "
          f"{month_end.index.min()} ~ {month_end.index.max()}")
    return dict(R_clean=R_clean, R_raw=R_raw, month_end=month_end, month_start=month_start,
                MC=MC, VT=VT, names=names, last_date=last_date, kospi=ix)


def monthly_returns(R, month_end):
    m = R.index.to_period("M")
    return (1 + R).groupby(m).prod(min_count=1) - 1


# ══════════════════════════════════════════════════════════════════
# 2. 유니버스·신호·분위 (기존 엔진과 같은 정의)
# ══════════════════════════════════════════════════════════════════

def universe(D):
    MC, VT, names = D["MC"], D["VT"], D["names"]
    cols = MC.columns
    common = pd.Series([str(c).endswith("0") for c in cols], index=cols)
    nm = names.reindex(cols).fillna("")
    bad = nm.map(lambda s: any(kw in s for kw in EXCLUDE_KW))
    mask = pd.DataFrame(True, index=MC.index, columns=cols)
    mask &= common & ~bad
    mask &= MC.notna()
    mask &= VT.reindex_like(MC).fillna(0) >= MIN_VALUE_TRADED
    rank = MC.where(mask).rank(axis=1, ascending=False, method="first")
    return (mask & (rank <= TOP_N)).fillna(False)


def momentum_signal(MR):
    lr = np.log1p(MR)
    return np.expm1(lr.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).sum())


def quintiles(sig_row, uni_row):
    s = sig_row.where(uni_row).dropna()
    if len(s) < N_Q * 5:
        return None
    r = s.rank(ascending=False, method="first")
    return np.ceil(r / (len(s) / N_Q)).clip(1, N_Q)


# ══════════════════════════════════════════════════════════════════
# 3. 보유 — 월초 동일가중, 월중 매수 후 보유, 일별 수익률
# ══════════════════════════════════════════════════════════════════

def hold_leg(R, tickers, days, last_day, last_date):
    sub = R.loc[days, tickers].fillna(0.0)
    delisted = [c for c in tickers if last_date[c] < last_day]
    for c in delisted:                                   # 그 달 안에 사라지면 월간 −30%가 되도록
        total = float((1 + sub[c]).prod())
        sub.loc[last_day, c] = (1 + DELIST_RET) / total - 1 if total > 0 else DELIST_RET
    wealth = (1 + sub).cumprod()
    V = wealth.mean(axis=1)
    daily = V / V.shift(1).fillna(1.0) - 1
    end_w = wealth.iloc[-1] / wealth.iloc[-1].sum()
    end_w = end_w.drop(delisted)                         # 사라진 종목은 다음 달에 들고 갈 수 없음
    return daily, float(V.iloc[-1] - 1), end_w


def turnover(prev_w, tickers):
    new = pd.Series(1.0 / len(tickers), index=tickers)
    if prev_w is None:
        return 1.0
    allc = new.index.union(prev_w.index)
    return float((new.reindex(allc, fill_value=0) - prev_w.reindex(allc, fill_value=0)).abs().sum() / 2)


def build_wml(D, R, quint_by_month, label):
    me, ms, ld = D["month_end"], D["month_start"], D["last_date"]
    months = list(me.index)
    daily, rows = [], []
    prev = {1: None, 5: None}
    for i, t in enumerate(months[:-1]):
        q = quint_by_month.get(t)
        if q is None:
            prev = {1: None, 5: None}
            continue
        M = months[i + 1]
        days = R.index[(R.index >= ms[M]) & (R.index <= me[M])]
        res = {}
        for leg in (1, 5):
            tk = list(q.index[q == leg])
            d_ret, m_ret, end_w = hold_leg(R, tk, days, me[M], ld)
            to = turnover(prev[leg], tk)
            res[leg] = (d_ret, m_ret, to)
            prev[leg] = end_w
        daily.append(res[1][0] - res[5][0])
        c1, c5 = res[1][2] * COST_RT, res[5][2] * COST_RT
        rows.append({"month": M, "q1": res[1][1], "q5": res[5][1],
                     "wml": res[1][1] - res[5][1],
                     "wml_net": (res[1][1] - c1) - (res[5][1] + c5),
                     "q1_net": res[1][1] - c1,
                     "to1": res[1][2], "to5": res[5][2]})
    mdf = pd.DataFrame(rows).set_index("month")
    ddf = pd.concat(daily).sort_index()
    print(f"    {label}: {mdf.index.min()} ~ {mdf.index.max()} ({len(mdf)}개월)")
    return mdf, ddf


# ══════════════════════════════════════════════════════════════════
# 4. 세 관리 전략
# ══════════════════════════════════════════════════════════════════

def h2a_vol_managed(wml_m, wml_d, me, ms):
    w = {}
    for M in wml_m.index:
        hist = wml_d[wml_d.index < ms[M]].tail(VOL_DAYS)
        if len(hist) < VOL_DAYS:
            continue
        sig = hist.std(ddof=1) * np.sqrt(252)
        if sig > 0:
            w[M] = TARGET_VOL / sig
    return pd.Series(w)


def h2b_dynamic(wml_m, wml_d, kospi, me, ms):
    km = kospi.groupby(kospi.index.to_period("M")).last()
    kd = kospi.pct_change()
    feat = {}
    for M in wml_m.index:
        prevM = M - 1
        back = prevM - BEAR_MONTHS
        if prevM not in km.index or back not in km.index:
            continue
        ib = float(km[prevM] / km[back] - 1 < 0)
        mh = kd[kd.index < ms[M]].dropna().tail(VOL_DAYS)
        wh = wml_d[wml_d.index < ms[M]].tail(VOL_DAYS)
        if len(mh) < VOL_DAYS or len(wh) < VOL_DAYS:
            continue
        feat[M] = (ib, float(mh.var(ddof=1)), float(wh.var(ddof=1)))
    F = pd.DataFrame(feat, index=["ib", "mvar", "wvar"]).T
    F["y"] = wml_m["wml"].reindex(F.index)

    raw = {}
    for j, M in enumerate(F.index):
        past = F.iloc[:j].dropna()
        if len(past) < DM_MIN_OBS:
            continue
        X = np.column_stack([np.ones(len(past)), past["ib"], past["ib"] * past["mvar"]])
        b, *_ = np.linalg.lstsq(X, past["y"].values, rcond=None)
        x = np.array([1.0, F.at[M, "ib"], F.at[M, "ib"] * F.at[M, "mvar"]])
        mu = float(x @ b)
        raw[M] = mu / F.at[M, "wvar"]
    return pd.Series(raw), F


def bhm_signals(D, MR, uni):
    """잔차 모멘텀 신호 — 시장·규모 2요인, 36개월 회귀, 12개월(최근 1개월 제외) 잔차 합 ÷ 표준편차."""
    me = D["month_end"]
    km = D["kospi"].groupby(D["kospi"].index.to_period("M")).last()
    mkt = km.pct_change()
    # SMB: 월말 KOSPI 전 종목 시가총액 중앙값으로 나눔 → 다음 달 동일가중 수익률 차
    smb = {}
    months = list(me.index)
    for i, t in enumerate(months[:-1]):
        mc = D["MC"].loc[t].dropna()
        if len(mc) < 50:
            continue
        med = mc.median()
        nxt = MR.loc[months[i + 1]]
        smb[months[i + 1]] = nxt.reindex(mc.index[mc <= med]).mean() - nxt.reindex(mc.index[mc > med]).mean()
    smb = pd.Series(smb)

    sig = {}
    for i, t in enumerate(months):
        if i < BHM_WINDOW - 1:
            continue
        win = months[i - BHM_WINDOW + 1: i + 1]                 # t−35 … t
        if not all(m in smb.index and m in mkt.index and pd.notna(mkt[m]) for m in win):
            continue
        tk = uni.columns[uni.loc[t].values] if t in uni.index else []
        if len(tk) == 0:
            continue
        Y = MR.loc[win, tk]
        Y = Y.loc[:, Y.notna().all()]
        if Y.shape[1] == 0:
            continue
        X = np.column_stack([np.ones(len(win)), mkt[win].values, smb[win].values])
        B, *_ = np.linalg.lstsq(X, Y.values, rcond=None)
        E = Y.values - X @ B
        e = E[-1 - LOOKBACK:-1]                                   # t−12 … t−1
        sd = e.std(axis=0, ddof=1)
        s = np.where(sd > 0, e.sum(axis=0) / sd, np.nan)
        sig[t] = pd.Series(s, index=Y.columns)
    return sig


# ══════════════════════════════════════════════════════════════════
# 5. 통계
# ══════════════════════════════════════════════════════════════════

def nw_alpha(y, x, lag=NW_LAG):
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return float(b[0]), float(b[0] / se[0]), float(b[1])


def perf(r):
    r = r.dropna()
    if len(r) < 2:
        return {}
    w = (1 + r).cumprod()
    return {"ann_mean": float(r.mean() * 12), "ann_vol": float(r.std(ddof=1) * np.sqrt(12)),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12)) if r.std() > 0 else np.nan,
            "mdd": float((w / w.cummax() - 1).min()), "months": int(len(r))}


# ══════════════════════════════════════════════════════════════════
# 6. 한 번 돌리기 (수익률 행렬 하나에 대해)
# ══════════════════════════════════════════════════════════════════

def run(D, R, label):
    print(f"\n  ── {label}")
    me, ms = D["month_end"], D["month_start"]
    MR = monthly_returns(R, me)
    MR.index = MR.index.astype("period[M]")
    uni = universe(D).reindex(index=MR.index, columns=MR.columns, fill_value=False)
    sig = momentum_signal(MR)

    qm = {}
    for t in MR.index:
        q = quintiles(sig.loc[t], uni.loc[t])
        if q is not None:
            qm[t] = q
    wml_m, wml_d = build_wml(D, R, qm, "모멘텀 WML")

    # 전달계수(TC): 동일가중 롱숏 비중 vs 신호 z점수
    tcs = {}
    for t, q in qm.items():
        s = sig.loc[t].reindex(q.index)
        z = (s - s.mean()) / s.std(ddof=1)
        a = pd.Series(0.0, index=q.index)
        a[q == 1] = 1 / (q == 1).sum()
        a[q == N_Q] = -1 / (q == N_Q).sum()
        tcs[t + 1] = float(np.corrcoef(a, z)[0, 1])
    tc = pd.Series(tcs)

    wa = h2a_vol_managed(wml_m, wml_d, me, ms)
    wb_raw, F = h2b_dynamic(wml_m, wml_d, D["kospi"], me, ms)

    bsig = bhm_signals(D, MR, uni)
    qr = {}
    for t, s in bsig.items():
        q = quintiles(s, pd.Series(True, index=s.index))
        if q is not None:
            qr[t] = q
    res_m, _ = build_wml(D, R, qr, "잔차 모멘텀 WML")

    # 공통 판정 표본: H2b 첫 예측월 ~ EVAL_END (사전등록 v3)
    start = wb_raw.index.min()
    end = pd.Period(EVAL_END, "M")
    idx = wml_m.index[(wml_m.index >= start) & (wml_m.index <= end)]
    idx = idx.intersection(wa.index).intersection(wb_raw.index).intersection(res_m.index)
    print(f"    판정 표본: {idx.min()} ~ {idx.max()} ({len(idx)}개월)")

    base, base_net = wml_m.loc[idx, "wml"], wml_m.loc[idx, "wml_net"]
    c = base.std(ddof=1) / (wb_raw[idx] * base).std(ddof=1)        # H2b 배율: WML과 같은 변동성
    strat = {
        "H2a": (wa[idx] * base, wa[idx] * base_net),
        "H2b": (c * wb_raw[idx] * base, c * wb_raw[idx] * base_net),
        "H2c": (res_m.loc[idx, "wml"], res_m.loc[idx, "wml_net"]),
    }

    out = {"label": label, "eval_start": str(idx.min()), "eval_end": str(idx.max()),
           "n_months": int(len(idx)), "tc_mean": float(tc.reindex(idx).mean()),
           "WML": {"gross": perf(base), "net": perf(base_net)}}
    for h, (g, n) in strat.items():
        a, t_, b = nw_alpha(g.values, base.values)
        out[h] = {"alpha_ann": a * 12, "t": t_, "beta": b, "pass": bool(t_ > T_PASS),
                  "gross": perf(g), "net": perf(n)}

    series = pd.DataFrame({"WML": wml_m["wml"], "WML_net": wml_m["wml_net"],
                           "Q1": wml_m["q1"], "Q1_net": wml_m["q1_net"],
                           "H2a_w": wa, "H2b_w_raw": wb_raw, "H2c": res_m["wml"]})
    series["H2a"] = series["H2a_w"] * series["WML"]
    series["H2b"] = c * series["H2b_w_raw"] * series["WML"]
    return out, series, tc


# ══════════════════════════════════════════════════════════════════
# 7. 출력
# ══════════════════════════════════════════════════════════════════

def show(out):
    print("\n" + "═" * 84)
    print(f"  판정 — {out['label']}   ({out['eval_start']} ~ {out['eval_end']}, {out['n_months']}개월, 비용 전)")
    print("═" * 84)
    print(f"  {'':<6}{'알파(연)':>10}{'NW t':>8}{'β':>7}{'연평균':>9}{'변동성':>9}{'샤프':>7}{'MDD':>9}   판정")
    g = out["WML"]["gross"]
    print(f"  {'WML':<6}{'':>10}{'':>8}{'':>7}{g['ann_mean']:>9.2%}{g['ann_vol']:>9.2%}"
          f"{g['sharpe']:>7.2f}{g['mdd']:>9.1%}   (기준)")
    for h in ("H2a", "H2b", "H2c"):
        r = out[h]; g = r["gross"]
        verdict = "통과" if r["pass"] else "통과 못 함"
        print(f"  {h:<6}{r['alpha_ann']:>10.2%}{r['t']:>8.2f}{r['beta']:>7.2f}{g['ann_mean']:>9.2%}"
              f"{g['ann_vol']:>9.2%}{g['sharpe']:>7.2f}{g['mdd']:>9.1%}   {verdict}")
    print(f"\n  기준: 알파의 Newey-West t > {T_PASS} (시차 {NW_LAG}, 본페로니 보정 단측 1%)")
    print("  비용 차감 후")
    for h in ("WML", "H2a", "H2b", "H2c"):
        n = out[h]["net"]
        print(f"    {h:<5} 연평균 {n['ann_mean']:+.2%}  샤프 {n['sharpe']:.2f}  MDD {n['mdd']:.1%}")
    print(f"  전달계수(TC) 평균: {out['tc_mean']:.2f}")


def main():
    for f in ("kospi", "index"):
        if not os.path.isdir(os.path.join(DATA, f)):
            sys.exit(f"[오류] {DATA}/{f} 가 없습니다. krx_daily.py 로 먼저 받으세요.")
    print("\n" + "═" * 84 + "\n  2장 Purified Alpha — 사전등록 v3 그대로 실행\n" + "═" * 84)
    D = load()

    main_out, series, tc = run(D, D["R_clean"], "판정용 (가격제한폭 밖 결측 처리)")
    chk_out, _, _ = run(D, D["R_raw"], "확인용 (결측 처리 없음)")

    show(main_out)
    show(chk_out)

    same = all(main_out[h]["pass"] == chk_out[h]["pass"] for h in ("H2a", "H2b", "H2c"))
    print("\n  두 처리 방식의 판정이 " + ("같습니다." if same else "다릅니다 — 보고서에 반드시 적어야 합니다."))

    print("\n" + "═" * 84 + "\n  2020년 월별 (서술용, 판정 아님 — 이미 본 구간)\n" + "═" * 84)
    y20 = series.loc[(series.index >= pd.Period("2020-01", "M")) & (series.index <= pd.Period("2020-12", "M")),
                     ["WML", "H2a", "H2b", "H2c"]]
    print(y20.map(lambda v: f"{v:+.1%}" if pd.notna(v) else "  —").to_string())

    pre_end = pd.Period(main_out["eval_start"], "M") - 1
    pre = series.loc[series.index <= pre_end, ["WML", "H2a", "H2b", "H2c"]]
    print("\n" + "═" * 84 + f"\n  판정 표본 이전 구간 (~{pre_end}, 처음 보는 구간 · 서술용)\n" + "═" * 84)
    for col in pre.columns:
        p = perf(pre[col])
        if p:
            print(f"    {col:<5} {p['months']:>3}개월  연평균 {p['ann_mean']:+.2%}  샤프 {p['sharpe']:.2f}")
        else:
            print(f"    {col:<5} 자료 없음")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ch2_results.json"), "w", encoding="utf-8") as f:
        json.dump({"main": main_out, "check": chk_out}, f, ensure_ascii=False, indent=1, default=str)
    s = series.copy()
    s.index = s.index.astype(str)
    s.to_csv(os.path.join(OUT, "ch2_monthly.csv"), encoding="utf-8-sig")
    print(f"\n  저장: {OUT}/ch2_results.json, ch2_monthly.csv")
    print("  화면에 나온 내용을 그대로 보내 주세요.\n")


if __name__ == "__main__":
    main()
