#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
krx_api.py — KRX Open API 수집기 (pykrx 대체)

pykrx 는 종목마다 한 번씩 긁어야 해서 350종목이면 350번 호출이었습니다.
Open API 의 일별매매정보는 '날짜 하나 = 전종목 한 방'이라 월말 132개면 132번입니다.
정식 경로이고 호출 수도 1/300 수준입니다.

    py krx_api.py --probe          # 응답 형태 점검 (호출 3회, 파일 안 만듦)
    py krx_api.py --fetch          # 전체 수집 → cache/ 에 저장
    py krx_api.py --fetch --from 2015 --to 2025

만드는 파일 (kospi_momentum.py 가 그대로 읽는 형식):
    cache/cross_sections.csv   월말 횡단면 (종가·시총·거래대금·상장주식수)
    cache/adj_prices.csv       수정주가   ← 모멘텀 계산에 쓰는 것
    cache/raw_prices.csv       수정 전 가격 ← 액면분할 영향을 비교해 보려고 같이 저장
    cache/benchmark.csv        코스피 지수
    cache/days/*.csv           날짜별 원본 (중단돼도 이어받기용)

★ 인증키는 환경변수 KRX_AUTH_KEY 에서 읽습니다. 코드에 적지 마세요.
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://data-dbg.krx.co.kr/svc/apis"

ENDPOINTS = {
    "stock": f"{BASE}/sto/stk_bydd_trd",        # 유가증권 일별매매정보
    "base":  f"{BASE}/sto/stk_isu_base_info",   # 유가증권 종목기본정보
}

# 지수는 '시리즈'별로 주소가 다릅니다. 어느 걸 승인받았는지 모르니 순서대로 찔러봅니다.
INDEX_URLS = [
    f"{BASE}/idx/krx_dd_trd",      # KRX 시리즈
    f"{BASE}/idx/kospi_dd_trd",    # 코스피 시리즈
    f"{BASE}/idx/kosdaq_dd_trd",   # 코스닥 시리즈
]

SLEEP = 0.3          # 호출 간격 (1일 10,000회 제한 — 이 수집은 300회 안팎)
TIMEOUT = 30


# ══════════════════════════════════════════════════════════════════
# 1. 호출
# ══════════════════════════════════════════════════════════════════

def auth_key() -> str:
    k = os.environ.get("KRX_AUTH_KEY", "").strip()
    if not k:
        print("[오류] 환경변수 KRX_AUTH_KEY 가 비어 있습니다.")
        print("       PowerShell:")
        print('         [Environment]::SetEnvironmentVariable("KRX_AUTH_KEY","받은키","User")')
        print("       넣은 뒤 터미널을 완전히 닫았다 다시 여세요.")
        sys.exit(1)
    return k


class Unauthorized(Exception):
    """해당 API 활용신청이 승인 안 됐거나 주소가 틀렸을 때."""


def call(url: str, bas_dd: str, key: str, retries: int = 3) -> list[dict]:
    """엔드포인트 1회 호출 → 행 리스트. 휴장일이면 빈 리스트."""
    qs = urllib.parse.urlencode({"basDd": bas_dd, "AUTH_KEY": key})
    full = f"{url}?{qs}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers={
                "AUTH_KEY": key,                 # 공식 문서는 헤더 방식
                "Accept": "application/json",
                "User-Agent": "python-urllib",
            })
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
                payload = json.loads(r.read().decode("utf-8"))
            return extract_rows(payload)

        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise Unauthorized(f"{e.code} {url}")
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"    [429] 호출 제한 — {wait}초 대기")
                time.sleep(wait)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(2)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    return []


def extract_rows(payload) -> list[dict]:
    """응답 껍데기 이름(OutBlock_1 등)이 뭐든 안의 리스트를 찾아냅니다."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
    return []


def num(v) -> float:
    """'1,234' · '' · '-' → float. 빈 값은 NaN."""
    if v is None:
        return np.nan
    s = str(v).replace(",", "").strip()
    if s in ("", "-", "null", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ══════════════════════════════════════════════════════════════════
# 2. 필드명 찾기 — 명세가 바뀌어도 버티도록
# ══════════════════════════════════════════════════════════════════

CANDIDATES = {
    "ticker":  ["ISU_CD", "ISU_SRT_CD", "SHORT_CODE", "ISU_SHRT_CD"],
    "name":    ["ISU_NM", "ISU_ABBRV", "ISU_KOR_NM"],
    "close":   ["TDD_CLSPRC", "CLSPRC", "TDD_CLS_PRC"],
    "mktcap":  ["MKTCAP", "MKT_CAP", "MKTTOTAMT"],
    "shares":  ["LIST_SHRS", "LISTSHRS", "LIST_SHRS_CNT"],
    "value":   ["ACC_TRDVAL", "ACCTRDVAL", "TRDVAL"],
    "idxname": ["IDX_NM", "IDXNM", "IDX_KOR_NM"],
    "idxclose": ["CLSPRC_IDX", "CLSPRC", "IDX_CLSPRC"],
}


def resolve(rows: list[dict], what: str) -> str | None:
    """후보 중 실제 응답에 있는 필드명을 고릅니다."""
    if not rows:
        return None
    keys = rows[0].keys()
    for c in CANDIDATES[what]:
        if c in keys:
            # 종목코드는 6자리인 쪽을 고릅니다 (ISU_CD 가 12자리 표준코드인 경우 대비)
            if what == "ticker":
                sample = str(rows[0].get(c, ""))
                if len(sample) != 6:
                    continue
            return c
    if what == "ticker":  # 6자리 조건으로 다 걸렀으면 그냥 첫 후보
        for c in CANDIDATES[what]:
            if c in keys:
                return c
    return None


# ══════════════════════════════════════════════════════════════════
# 3. 월말 영업일 찾기 (휴장일이면 하루씩 뒤로)
# ══════════════════════════════════════════════════════════════════

def month_ends(y0: int, m0: int, y1: int, m1: int) -> list[str]:
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y}{m:02d}{calendar.monthrange(y, m)[1]:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def resolve_trading_day(cal_day: str, key: str, cache: dict) -> tuple[str, list[dict]]:
    """달력상 말일부터 거꾸로 올라가며 거래가 있던 날을 찾습니다."""
    if cal_day in cache:
        d = cache[cal_day]
        return d, call(ENDPOINTS["stock"], d, key)

    ts = pd.Timestamp(cal_day)
    for back in range(12):
        d = (ts - pd.Timedelta(days=back)).strftime("%Y%m%d")
        rows = call(ENDPOINTS["stock"], d, key)
        time.sleep(SLEEP)
        if rows:
            cache[cal_day] = d
            return d, rows
    return "", []


# ══════════════════════════════════════════════════════════════════
# 4. 점검 모드
# ══════════════════════════════════════════════════════════════════

def probe(bas_dd: str):
    key = auth_key()
    print("\n" + "═" * 78)
    print(f"  응답 점검 — 기준일 {bas_dd}")
    print("═" * 78)

    targets = [("stock", ENDPOINTS["stock"]), ("base", ENDPOINTS["base"])]
    targets += [(f"index:{u.rsplit('/', 1)[-1]}", u) for u in INDEX_URLS]

    for tag, url in targets:
        print(f"\n[{tag}]  {url}")
        try:
            rows = call(url, bas_dd, key)
        except Unauthorized as e:
            print(f"  ✗ {e} — 이 API는 미승인이거나 주소가 다릅니다. (넘어감)")
            time.sleep(SLEEP)
            continue
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
            time.sleep(SLEEP)
            continue

        if not rows:
            print("  행 0개 — 휴장일일 수 있습니다.")
            time.sleep(SLEEP)
            continue

        print(f"  ✓ 행 {len(rows):,}개")
        print(f"  필드: {', '.join(rows[0].keys())}")

        if tag == "stock":
            print("  첫 행:")
            for k, v in rows[0].items():
                print(f"    {k:<16} {v}")
            print("\n  → 필드 매칭")
            for what in ("ticker", "name", "close", "mktcap", "shares", "value"):
                f = resolve(rows, what)
                print(f"      {what:<8} {f if f else '✗ 못 찾음'}")

        elif tag.startswith("index"):
            nm, cl = resolve(rows, "idxname"), resolve(rows, "idxclose")
            print(f"  지수명 필드={nm}  종가 필드={cl}")
            if nm:
                names = sorted({str(r.get(nm, "")).strip() for r in rows})
                print(f"  이 시리즈에 담긴 지수 ({len(names)}개):")
                for n in names[:25]:
                    mark = "  ★ 이게 벤치마크" if n == "코스피" else ""
                    print(f"      · {n}{mark}")
                if len(names) > 25:
                    print(f"      ... 외 {len(names) - 25}개")

        time.sleep(SLEEP)

    print("\n" + "═" * 78)
    print("  stock 이 ✓ 면 수집은 됩니다. 지수는 ★ 표시가 뜬 시리즈를 씁니다.")
    print("  ★ 가 어디에도 없으면 이 출력을 보내주세요.")
    print("═" * 78 + "\n")


# ══════════════════════════════════════════════════════════════════
# 5. 수집
# ══════════════════════════════════════════════════════════════════

def pick_index_source(bas_dd: str, key: str, want_name: str | None = None):
    """승인된 지수 시리즈 중 '코스피'가 들어 있는 걸 고릅니다.

    없으면 '코스피 200'으로 대체하고, 그것도 없으면 벤치마크를 포기합니다.
    포기해도 전략 자체는 돌아갑니다 — 시장 대비 비교만 못 하게 됩니다.
    want_name 을 주면 그 이름의 지수를 찾습니다 (예: "KRX 300").
    """
    targets = [want_name] if want_name else ["코스피", "코스피 200"]
    fallback = None
    for url in INDEX_URLS:
        try:
            rows = call(url, bas_dd, key)
        except Exception:
            time.sleep(SLEEP)
            continue
        time.sleep(SLEEP)
        if not rows:
            continue
        nm, cl = resolve(rows, "idxname"), resolve(rows, "idxclose")
        if not (nm and cl):
            continue
        names = {str(r.get(nm, "")).strip() for r in rows}
        if targets[0] in names:
            print(f"  벤치마크 : {targets[0]}  ({url.rsplit('/', 1)[-1]})")
            return url, nm, cl, targets[0]
        for alt in targets[1:]:
            if fallback is None and alt in names:
                fallback = (url, nm, cl, alt)
    if fallback:
        print(f"  벤치마크 : {fallback[3]} ({targets[0]} 미승인 → 대체)")
        return fallback
    print("  벤치마크 : ✗ 사용 가능한 지수 없음 — 전략만 돌립니다")
    print("            지수 → '코스피 시리즈 일별시세' 활용신청 후")
    print("            py krx_api.py --bench  로 나중에 따로 받으면 됩니다.")
    return None


def fetch_bench_only(y0, m0, y1, m1, cache_dir="./cache", want_name=None):
    """벤치마크만 따로 수집 — 주가 수집을 이미 끝낸 뒤에 씁니다.

    cache/month_end_dates.json 에 적힌 '이미 확인된 영업일'을 재사용하므로
    휴장일 되짚기가 없고, 딱 월 수만큼만 호출합니다.
    """
    key = auth_key()
    dmap_path = os.path.join(cache_dir, "month_end_dates.json")
    if not os.path.exists(dmap_path):
        print(f"[오류] {dmap_path} 가 없습니다. 먼저 --fetch 를 돌리세요.")
        sys.exit(1)
    dmap = json.load(open(dmap_path, encoding="utf-8"))

    days = [dmap[c] for c in month_ends(y0, m0, y1, m1) if c in dmap]
    print(f"\n  벤치마크 수집 — {len(days)}개월")

    src = pick_index_source(days[-1], key, want_name)
    if not src:
        sys.exit(1)
    url, f_nm, f_cl, want = src

    bench = {}
    for i, d in enumerate(days, 1):
        try:
            for r in call(url, d, key):
                if str(r.get(f_nm, "")).strip() == want:
                    bench[pd.to_datetime(d)] = num(r.get(f_cl))
                    break
        except Exception as e:
            print(f"    [경고] {d}: {e}")
        if i % 24 == 0:
            print(f"    {i}/{len(days)} ...")
        time.sleep(SLEEP)

    if not bench:
        print("[오류] 한 건도 못 받았습니다.")
        sys.exit(1)

    s = pd.Series(bench).sort_index()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    s.name = want
    s.to_frame().to_csv(os.path.join(cache_dir, "benchmark.csv"))
    print(f"\n  저장: {cache_dir}/benchmark.csv  "
          f"({len(s)}개월, {s.index.min():%Y-%m} ~ {s.index.max():%Y-%m})")
    print(f"  {want} 누적: {s.iloc[-1] / s.iloc[0] - 1:+.1%}\n")


def fetch(y0, m0, y1, m1, cache_dir="./cache"):
    key = auth_key()
    days_dir = os.path.join(cache_dir, "days")
    os.makedirs(days_dir, exist_ok=True)

    cal_days = month_ends(y0, m0, y1, m1)
    print(f"\n  요청 기간 : {y0}-{m0:02d} ~ {y1}-{m1:02d}  ({len(cal_days)}개월)")

    idx_src = "unset"   # 첫 영업일을 찾은 뒤에 한 번만 결정합니다

    dmap_path = os.path.join(cache_dir, "month_end_dates.json")
    dmap = json.load(open(dmap_path, encoding="utf-8")) if os.path.exists(dmap_path) else {}

    frames, bench = [], {}
    for i, cal in enumerate(cal_days, 1):
        day_csv = os.path.join(days_dir, f"{cal}.csv")

        # ── 이미 받은 달은 건너뜁니다 (중단돼도 이어받기)
        if os.path.exists(day_csv):
            frames.append(pd.read_csv(day_csv, dtype={"ticker": str}))
            continue

        d, rows = resolve_trading_day(cal, key, dmap)
        if not rows:
            print(f"    [빠짐] {cal} — 12일 거슬러도 데이터 없음")
            continue

        f_tk = resolve(rows, "ticker")
        f_nm = resolve(rows, "name")
        f_cl = resolve(rows, "close")
        f_mc = resolve(rows, "mktcap")
        f_sh = resolve(rows, "shares")
        f_va = resolve(rows, "value")
        missing = [n for n, f in
                   [("ticker", f_tk), ("close", f_cl), ("mktcap", f_mc)] if not f]
        if missing:
            print(f"\n[오류] 필수 필드를 못 찾았습니다: {missing}")
            print("       py krx_api.py --probe 결과를 보내주세요.")
            sys.exit(1)

        sub = pd.DataFrame({
            "ticker": [str(r.get(f_tk, "")).zfill(6) for r in rows],
            "name": [r.get(f_nm, "") for r in rows] if f_nm else "",
            "close_raw": [num(r.get(f_cl)) for r in rows],
            "mktcap": [num(r.get(f_mc)) for r in rows],
            "value_traded": [num(r.get(f_va)) for r in rows] if f_va else np.nan,
            "shares": [num(r.get(f_sh)) for r in rows] if f_sh else np.nan,
        })
        sub["date"] = pd.to_datetime(d)
        sub = sub[sub["close_raw"] > 0]
        sub.to_csv(day_csv, index=False)
        frames.append(sub)

        # ── 같은 날 지수 (첫 영업일에 어느 시리즈를 쓸지 한 번만 정합니다)
        if idx_src == "unset":
            idx_src = pick_index_source(d, key)
        if idx_src:
            iurl, f_in, f_ic, want = idx_src
            try:
                irows = call(iurl, d, key)
                time.sleep(SLEEP)
                for r in irows:
                    if str(r.get(f_in, "")).strip() == want:
                        bench[pd.to_datetime(d)] = num(r.get(f_ic))
                        break
            except Exception as e:
                print(f"    [경고] {d} 지수 실패: {e}")

        if i % 12 == 0:
            print(f"    {i}/{len(cal_days)} ...  ({d}, {len(sub)}종목)")
        time.sleep(SLEEP)

    json.dump(dmap, open(dmap_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if not frames:
        print("[오류] 수집된 데이터가 없습니다.")
        sys.exit(1)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # ── 요청한 개월 수와 실제로 받은 개월 수가 다르면 크게 경고
    got = panel["date"].dt.to_period("M").nunique()
    if got < len(cal_days):
        print("\n" + "!" * 78)
        print(f"  경고 — 요청 {len(cal_days)}개월 중 {got}개월만 받았습니다.")
        print("  이 상태로 돌리면 결과표는 멀쩡해 보이는데 기간이 다릅니다.")
        print("  아래 '실제 기간'을 반드시 확인하고 해석하세요.")
        print("!" * 78)

    names = panel.sort_values("date").groupby("ticker")["name"].last().to_dict() \
        if "name" in panel.columns else {}

    # ── 가격 행렬
    panel["month"] = panel["date"].dt.to_period("M").dt.to_timestamp("M")
    raw = panel.pivot_table(index="month", columns="ticker", values="close_raw")
    shares = panel.pivot_table(index="month", columns="ticker", values="shares")
    adj, n_split, split_log = adjust_prices(raw, shares)

    out = panel.drop(columns=["month", "name"], errors="ignore")
    os.makedirs(cache_dir, exist_ok=True)
    out.to_csv(os.path.join(cache_dir, "cross_sections.csv"), index=False)
    adj.to_csv(os.path.join(cache_dir, "adj_prices.csv"))
    raw.to_csv(os.path.join(cache_dir, "raw_prices.csv"))
    if names:
        json.dump(names, open(os.path.join(cache_dir, "names.json"), "w",
                              encoding="utf-8"), ensure_ascii=False)
    if bench:
        s = pd.Series(bench).sort_index()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        s.name = "KOSPI"
        s.to_frame().to_csv(os.path.join(cache_dir, "benchmark.csv"))

    print("\n" + "═" * 78)
    print("  수집 완료")
    print(f"    실제 기간   : {raw.index.min():%Y-%m} ~ {raw.index.max():%Y-%m}  ({len(raw)}개월)")
    print(f"    종목 수     : {raw.shape[1]:,}개 (연인원 기준, 상장폐지 종목 포함)")
    print(f"    벤치마크    : {'수집됨 ' + str(len(bench)) + '개월' if bench else '✗ 실패'}")
    print(f"    액면분할·병합 보정 : {n_split}건")
    for tk, mth, r in split_log[:10]:
        print(f"        {tk}  {mth:%Y-%m}  주식수 {r:.2f}배 → 이전 가격 보정")
    if n_split > 10:
        print(f"        ... 외 {n_split - 10}건")
    print("═" * 78)
    print("\n  이제:  py kospi_momentum.py\n")


# ══════════════════════════════════════════════════════════════════
# 6. 수정주가 — Open API 는 '그날 실제 거래가'만 줍니다
# ══════════════════════════════════════════════════════════════════

def adjust_prices(raw: pd.DataFrame, shares: pd.DataFrame,
                  min_ratio: float = 1.5, tol: float = 0.30):
    """상장주식수 급변 + 가격의 역방향 급락을 함께 보고 분할·병합을 잡아냅니다.

    왜 둘 다 봐야 하는가:
      · 주식수만 보면 유상증자도 분할로 잘못 잡습니다.
      · 가격만 보면 그냥 폭락한 날도 분할로 잘못 잡습니다.
      분할이면 주식수 r배 ↔ 가격 1/r배 라서 둘을 곱하면 1에 가깝습니다.
      유상증자는 주식수만 늘고 가격은 그대로라 곱이 1에서 멉니다.

    보정 방향: 분할 '이전' 가격을 r 로 나눠 오늘 기준으로 끌어내립니다.
    """
    shares = shares.reindex_like(raw)
    sr = shares / shares.shift(1)          # 주식수 비율
    pr = raw / raw.shift(1)                # 가격 비율

    big = (sr >= min_ratio) | (sr <= 1.0 / min_ratio)
    prod = sr * pr
    inverse = (prod >= 1 - tol) & (prod <= 1 + tol)
    flag = big & inverse & sr.notna() & pr.notna()

    log = [(c, i, float(sr.loc[i, c]))
           for c in flag.columns for i in flag.index[flag[c]]]
    log.sort(key=lambda x: x[1])

    f = sr.where(flag, 1.0).fillna(1.0)
    # adj[t] = t 이후에 일어난 모든 분할 배수의 곱
    adj_factor = f[::-1].shift(1).fillna(1.0).cumprod()[::-1]
    return raw / adj_factor, int(flag.sum().sum()), log


# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="KRX Open API 수집기")
    ap.add_argument("--probe", nargs="?", const="", metavar="YYYYMMDD",
                    help="응답 형태 점검 (기본: 최근 평일)")
    ap.add_argument("--fetch", action="store_true", help="전체 수집")
    ap.add_argument("--bench", action="store_true",
                    help="벤치마크만 따로 수집 (--fetch 를 이미 돌린 뒤)")
    ap.add_argument("--bench-name", default=None,
                    help='지수 이름 직접 지정 (예: "KRX 300")')
    ap.add_argument("--from", dest="y0", type=int, default=2015)
    ap.add_argument("--to", dest="y1", type=int, default=2025)
    ap.add_argument("--cache", default="./cache")
    args = ap.parse_args()

    if args.probe is not None:
        d = args.probe
        if not d:
            t = datetime.date.today() - datetime.timedelta(days=1)
            while t.weekday() >= 5:
                t -= datetime.timedelta(days=1)
            d = t.strftime("%Y%m%d")
        probe(d)
        return

    if args.bench:
        fetch_bench_only(args.y0, 1, args.y1, 12, args.cache, args.bench_name)
        return

    if args.fetch:
        fetch(args.y0, 1, args.y1, 12, args.cache)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
