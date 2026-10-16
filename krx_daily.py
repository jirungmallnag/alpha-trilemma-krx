#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
krx_daily.py — 일별 데이터 수집기 (Alpha Trilemma 프로젝트 0단계)

날짜 하나 = 시장 전체 한 번 호출. 월 단위 Parquet 파일로 저장하고,
하루 호출 한도에 닿으면 스스로 멈췄다가 다음 날 이어받습니다.

    py krx_daily.py --plan        # 남은 호출 수만 계산 (호출 0회)
    py krx_daily.py               # 오늘 한도까지 수집 (기본 9,000회에서 멈춤)
    py krx_daily.py --only etf    # 특정 데이터만
    py krx_daily.py --status      # 무엇이 얼마나 받아졌는지

저장 위치 (C:\\quant\\data\\):
    data/kospi/2010-01.parquet ...   유가증권 일별매매정보
    data/kosdaq/...                  코스닥 일별매매정보
    data/index/...                   KOSPI 시리즈 일별시세
    data/etf/...                     ETF 일별매매정보
    data/etn/...                     ETN 일별매매정보
    data/_budget.json                날짜별 호출 수 (하루 10,000회 약관 한도 관리)

★ 필요: py -m pip install pyarrow
★ krx_api.py 와 같은 폴더. 인증키는 KRX_AUTH_KEY 환경변수.
★ 받은 원데이터는 약관상 제3자 제공 금지 — data 폴더는 GitHub에 올리지 마세요.
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from krx_api import BASE, SLEEP, Unauthorized, auth_key, call, num

DATA = "./data"

# (이름, 경로, 시작 연-월) — 순서가 곧 수집 순서. kospi 가 맨 앞이어야 합니다:
# 그 달의 '거래일 목록'을 kospi 파일에서 읽어 나머지 데이터의 호출 날짜로 씁니다.
DATASETS = [
    ("kospi",  "sto/stk_bydd_trd", (2010, 1)),
    ("kosdaq", "sto/ksq_bydd_trd", (2010, 1)),
    ("etf",    "etp/etf_bydd_trd", (2010, 1)),
    ("index",  "idx/kospi_dd_trd", (2010, 1)),
    ("etn",    "etp/etn_bydd_trd", (2014, 11)),   # ETN 시장 개설 2014년 11월
]
END = (2025, 12)

TEXT_COLS_SUFFIX = ("_CD", "_NM")
TEXT_COLS = {"BAS_DD", "IDX_CLSS"}


# ══════════════════════════════════════════════════════════════════
# 호출 한도 관리 — 날짜별로 센다
# ══════════════════════════════════════════════════════════════════

class Budget:
    def __init__(self, limit: int):
        self.path = os.path.join(DATA, "_budget.json")
        self.limit = limit
        self.today = datetime.date.today().isoformat()
        try:
            with open(self.path, encoding="utf-8") as f:
                self.log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.log = {}

    @property
    def used(self) -> int:
        return int(self.log.get(self.today, 0))

    @property
    def left(self) -> int:
        return max(0, self.limit - self.used)

    def spend(self, n: int = 1):
        self.log[self.today] = self.used + n
        os.makedirs(DATA, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=1)


class OutOfBudget(Exception):
    pass


# ══════════════════════════════════════════════════════════════════
# 월 단위 파일
# ══════════════════════════════════════════════════════════════════

def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def month_path(name, y, m, empty=False):
    return os.path.join(DATA, name, f"{y}-{m:02d}" + (".empty" if empty else ".parquet"))


def month_done(name, y, m) -> bool:
    return os.path.exists(month_path(name, y, m)) or os.path.exists(month_path(name, y, m, True))


def weekdays(y, m):
    n = calendar.monthrange(y, m)[1]
    for d in range(1, n + 1):
        day = datetime.date(y, m, d)
        if day.weekday() < 5:
            yield day.strftime("%Y%m%d")


def trading_days(y, m):
    """kospi 파일에 실제로 찍힌 날짜 = 그 달의 거래일."""
    p = month_path("kospi", y, m)
    if os.path.exists(p):
        return sorted(pd.read_parquet(p, columns=["BAS_DD"])["BAS_DD"].astype(str).unique())
    if os.path.exists(month_path("kospi", y, m, True)):
        return []
    return None   # 아직 모름


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """문자열 응답 → 코드·이름은 문자열, 나머지는 숫자."""
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c in TEXT_COLS or c.endswith(TEXT_COLS_SUFFIX):
            df[c] = df[c].astype(str).str.strip()
        else:
            df[c] = df[c].map(num).astype("float64")
    return df


def save_month(name, y, m, frames):
    os.makedirs(os.path.join(DATA, name), exist_ok=True)
    if not frames:
        open(month_path(name, y, m, True), "w").close()
        return 0
    df = pd.concat(frames, ignore_index=True)
    tmp = month_path(name, y, m) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, month_path(name, y, m))   # 다 쓴 뒤에만 이름을 붙여 반쪽 파일 방지
    return len(df)


# ══════════════════════════════════════════════════════════════════
# 수집
# ══════════════════════════════════════════════════════════════════

def collect(name, path, start, key, budget: Budget):
    url = f"{BASE}/{path}"
    todo = [(y, m) for y, m in months(start, END) if not month_done(name, y, m)]
    if not todo:
        print(f"  {name:<7} 이미 완료")
        return
    print(f"\n  {name:<7} 남은 {len(todo)}개월 수집 시작")

    for i, (y, m) in enumerate(todo, 1):
        if name == "kospi":
            days = list(weekdays(y, m))          # 거래일을 여기서 찾아냅니다
        else:
            days = trading_days(y, m)
            if days is None:
                print(f"  {name:<7} {y}-{m:02d}: kospi 가 먼저 받아져야 합니다 → 중단")
                return
        if budget.left < len(days):
            raise OutOfBudget()

        frames = []
        for d in days:
            budget.spend()                       # 실패한 호출도 한도에서 셉니다
            rows = call(url, d, key)
            time.sleep(SLEEP)
            if rows:
                f = to_frame(rows)
                if "BAS_DD" not in f.columns:
                    f["BAS_DD"] = d
                frames.append(f)
        n = save_month(name, y, m, frames)

        if i % 6 == 0 or i == len(todo):
            print(f"    {y}-{m:02d}  {n:>7,}행   (오늘 {budget.used:,}회 사용, 남은 한도 {budget.left:,})")


def plan():
    print("\n  남은 호출 수 (거래일을 아직 모르는 달은 평일 수로 추정)")
    total = 0
    for name, _, start in DATASETS:
        calls = 0
        for y, m in months(start, END):
            if month_done(name, y, m):
                continue
            td = trading_days(y, m)
            calls += len(td) if td is not None else len(list(weekdays(y, m)))
        total += calls
        print(f"    {name:<7} {calls:>6,}회")
    print(f"    {'합계':<6} {total:>6,}회   → 하루 9,000회 기준 약 {max(1, -(-total // 9000))}일")


def status():
    print("\n  받은 데이터")
    for name, _, start in DATASETS:
        ms = list(months(start, END))
        done = [(y, m) for y, m in ms if month_done(name, y, m)]
        full = [(y, m) for y, m in done if os.path.exists(month_path(name, y, m))]
        if full:
            first, last = full[0], full[-1]
            rows = sum(pd.read_parquet(month_path(name, y, m), columns=["BAS_DD"]).shape[0]
                       for y, m in full)
            span = f"{first[0]}-{first[1]:02d} ~ {last[0]}-{last[1]:02d}"
        else:
            rows, span = 0, "—"
        print(f"    {name:<7} {len(done):>3}/{len(ms)}개월   데이터 있는 구간 {span:<19} {rows:>11,}행")


def load(name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """분석 코드에서 쓰는 읽기 함수.  load('etf', '2020-01', '2020-12')"""
    folder = os.path.join(DATA, name)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    if start:
        files = [f for f in files if f[:7] >= start]
    if end:
        files = [f for f in files if f[:7] <= end]
    df = pd.concat([pd.read_parquet(os.path.join(folder, f)) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["BAS_DD"], format="%Y%m%d")
    return df


def main():
    ap = argparse.ArgumentParser(description="KRX 일별 데이터 수집기")
    ap.add_argument("--plan", action="store_true", help="남은 호출 수만 계산")
    ap.add_argument("--status", action="store_true", help="받은 데이터 현황")
    ap.add_argument("--only", help="kospi / kosdaq / etf / index / etn 중 하나")
    ap.add_argument("--budget", type=int, default=9000,
                    help="오늘 쓸 최대 호출 수 (약관 한도 10,000 — 기본 9,000으로 여유)")
    args = ap.parse_args()

    if args.plan:
        plan(); return
    if args.status:
        status(); return

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow 가 필요합니다:  py -m pip install pyarrow")
        sys.exit(1)

    key = auth_key()
    budget = Budget(args.budget)
    print(f"\n  오늘 사용 {budget.used:,}회 · 남은 한도 {budget.left:,}회")
    targets = [d for d in DATASETS if not args.only or d[0] == args.only]
    if not targets:
        print(f"  알 수 없는 이름: {args.only}"); sys.exit(1)

    try:
        for name, path, start in targets:
            try:
                collect(name, path, start, key, budget)
            except Unauthorized as e:
                print(f"  {name:<7} ✗ 접근 불가 ({e}) — 승인 상태를 확인하세요. 다음으로 넘어갑니다.")
    except OutOfBudget:
        print("\n  오늘 한도에 닿아 멈췄습니다. 내일 같은 명령으로 이어서 받으면 됩니다.")
    except KeyboardInterrupt:
        print("\n  중단했습니다. 끝난 달은 저장돼 있고, 진행 중이던 달만 다시 받습니다.")

    status()
    print(f"\n  오늘 총 {budget.used:,}회 사용")


if __name__ == "__main__":
    main()
