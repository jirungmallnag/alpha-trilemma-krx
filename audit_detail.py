#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_detail.py — audit.py 가 찾은 문제 행을 '분석에 영향을 주는가' 기준으로 쪼갭니다.

    py audit_detail.py

audit.py 를 먼저 돌려 output/audit/*.csv 가 있어야 합니다. API 호출 없음.

확인하는 것
  A. 가격제한폭 밖 469행 중 2장 유니버스(보통주·시가총액 상위 350)에 걸리는 것은 몇 개인가
  B. 분할·병합일 중 '전일대비'가 맞지 않은 1건은 무엇인가
  C. ETF 종가-NAV 10배 차이 행은 종목별로 어떻게 몰려 있는가 (실제 사건 vs 단위 오류)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA = "./data"
AUD = "./output/audit"


def read_kospi(cols):
    folder = os.path.join(DATA, "kospi")
    frames = [pd.read_parquet(os.path.join(folder, f), columns=cols)
              for f in sorted(os.listdir(folder)) if f.endswith(".parquet")]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["BAS_DD"], format="%Y%m%d")
    return df


def kind(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("9"):
        return "외국기업(KDR 등)"
    if not code.endswith("0"):
        return "우선주 등"
    return "보통주"


def part_a():
    print("\n" + "═" * 78 + "\n  A. 가격제한폭 밖 행 — 2장 유니버스에 걸리는가\n" + "═" * 78)
    v = pd.read_csv(os.path.join(AUD, "kospi_over_limit.csv"), dtype={"ISU_CD": str})
    v["ISU_CD"] = v["ISU_CD"].str.zfill(6)
    v["date"] = pd.to_datetime(v["date"])
    v["종류"] = v["ISU_CD"].map(kind)

    k = read_kospi(["BAS_DD", "ISU_CD", "MKTCAP"])
    k = k.sort_values(["ISU_CD", "date"])
    # 직전 거래 기록과의 간격 — 길면 거래정지 후 재개일
    k["gap_days"] = k.groupby("ISU_CD")["date"].diff().dt.days
    # 그날 보통주 중 시가총액 순위 (유니버스 근사)
    common = k[k["ISU_CD"].map(kind) == "보통주"].copy()
    common["rank"] = common.groupby("date")["MKTCAP"].rank(ascending=False, method="first")
    v = v.merge(k[["ISU_CD", "date", "gap_days"]], on=["ISU_CD", "date"], how="left") \
         .merge(common[["ISU_CD", "date", "rank"]], on=["ISU_CD", "date"], how="left")
    v["거래재개일(7일+공백)"] = v["gap_days"] > 7
    v["상위350"] = v["rank"] <= 350

    print(f"\n  전체 {len(v)}행")
    print(v.groupby("종류").size().rename("행").to_string())
    cm = v[v["종류"] == "보통주"]
    print(f"\n  보통주 {len(cm)}행 중")
    print(f"    거래정지 뒤 재개일(직전 기록과 7일 넘게 떨어짐): {cm['거래재개일(7일+공백)'].sum()}행")
    print(f"    그날 시가총액 상위 350 안: {cm['상위350'].sum()}행")
    hit = cm[cm["상위350"]].copy()
    if len(hit):
        print(f"\n  ▶ 유니버스에 걸리는 행 ({len(hit)}개) — 이것만 2장 결과에 영향을 줄 수 있습니다")
        hit["abs"] = hit["ret"].abs()
        for _, x in hit.sort_values("abs", ascending=False).head(15).iterrows():
            gap = f"{int(x['gap_days'])}일 공백" if pd.notna(x["gap_days"]) else "첫 기록"
            print(f"    {x['date']:%Y-%m-%d} {x['ISU_CD']} {str(x['ISU_NM'])[:14]:<14} "
                  f"{x['ret']:+9.1%}  순위 {int(x['rank']):>3}  {gap}")
    else:
        print("\n  ▶ 유니버스에 걸리는 행이 없습니다 — 2장 결과에는 영향이 없습니다")
    return len(hit)


def part_b():
    print("\n" + "═" * 78 + "\n  B. 분할·병합일 중 '전일대비'가 맞지 않은 행\n" + "═" * 78)
    s = pd.read_csv(os.path.join(AUD, "kospi_split_days.csv"), dtype={"ISU_CD": str})
    bad = s[(s["raw_ret"].abs() > 0.5) & (~s["ret_ok"].astype(bool))]
    if bad.empty:
        print("  없음")
    for _, x in bad.iterrows():
        print(f"  {x['date']}  {str(x['ISU_CD']).zfill(6)} {x['ISU_NM']}  "
              f"주식수 {x['share_ratio']:.2f}배  종가 변화 {x['raw_ret']:+.1%}  전일대비 수익률 {x['ret']:+.1%}  "
              f"[{kind(x['ISU_CD'])}]")


def part_c():
    print("\n" + "═" * 78 + "\n  C. ETF 종가와 NAV가 10배 넘게 다른 행 — 종목별\n" + "═" * 78)
    e = pd.read_csv(os.path.join(AUD, "etf_extreme_premium.csv"), dtype={"ISU_CD": str})
    e["ratio"] = e["TDD_CLSPRC"] / e["NAV"]
    unit = e[(e["ratio"] > 10) | (e["ratio"] < 0.1)]
    g = unit.groupby(["ISU_CD", "ISU_NM"]).agg(
        행=("date", "size"), 시작=("date", "min"), 끝=("date", "max"),
        배율_중앙값=("ratio", "median")).sort_values("행", ascending=False)
    print(f"\n  {len(unit)}행 · ETF {len(g)}개\n")
    with pd.option_context("display.width", 140, "display.max_colwidth", 24):
        print(g.head(20).to_string())
    print("\n  읽는 법")
    print("    · 한 종목에 수십~수백 행이 몰리고 배율이 수백 배 → 기초자산 가치 붕괴 같은 실제 사건 (예: 러시아)")
    print("    · 여러 종목에 1~2행씩 흩어지고 배율이 10·100·1000 같은 자릿수 → 단위·입력 오류")


def main():
    if not os.path.exists(os.path.join(AUD, "kospi_over_limit.csv")):
        print("output/audit 에 csv가 없습니다. 먼저 py audit.py 를 실행하세요.")
        return
    part_a()
    part_b()
    part_c()
    print("\n" + "═" * 78 + "\n  화면에 나온 내용을 그대로 보내 주세요.\n" + "═" * 78 + "\n")


if __name__ == "__main__":
    main()
