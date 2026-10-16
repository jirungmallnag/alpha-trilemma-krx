#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
krx_probe_all.py — 0단계 점검: 승인받은 API가 실제로 무엇을, 언제부터 주는가

프로젝트 설계는 '공개 명세'를 믿고 짰습니다. 이 스크립트는 그 믿음을 실제 응답으로
확인합니다. (Quantit: "존재 여부는 문서가 아니라 쿼리로 확인해야 한다")

    py krx_probe_all.py            # 필드 점검 + 제공 시작 연도 점검 (호출 약 150회)
    py krx_probe_all.py --fields   # 필드 점검만 (호출 약 20회)

결과는 화면에 찍히고 probe_report.json 으로도 저장됩니다.
★ krx_api.py 와 같은 폴더에 있어야 합니다. 인증키는 KRX_AUTH_KEY 환경변수.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time

from krx_api import BASE, SLEEP, Unauthorized, auth_key, call, num

# (이름, 경로, 명세 확인 여부) — '추정'은 공개 명세에 없어 이름으로 짐작한 경로
ENDPOINTS = [
    ("유가증권 일별매매",   "sto/stk_bydd_trd",      "명세"),
    ("유가증권 종목기본",   "sto/stk_isu_base_info", "명세"),
    ("KOSPI 시리즈",        "idx/kospi_dd_trd",      "명세"),
    ("KRX 시리즈",          "idx/krx_dd_trd",        "명세"),
    ("ETF 일별",            "etp/etf_bydd_trd",      "명세"),
    ("ETN 일별",            "etp/etn_bydd_trd",      "명세"),
    ("ELW 일별",            "etp/elw_bydd_trd",      "추정"),
    ("선물(주식선물外)",     "drv/fut_bydd_trd",      "명세"),
    ("주식선물(유가)",       "drv/eqsfu_stk_bydd_trd", "명세"),
    ("주식선물(코스닥)",     "drv/eqkfu_ksq_bydd_trd", "명세"),
    ("옵션(주식옵션外)",     "drv/opt_bydd_trd",      "명세"),
    ("주식옵션(유가)",       "drv/eqsop_bydd_trd",    "명세"),
    ("주식옵션(코스닥)",     "drv/eqkop_bydd_trd",    "명세"),
    ("국채전문유통",         "bon/kts_bydd_trd",      "추정"),
    ("일반채권",             "bon/bnd_bydd_trd",      "추정"),
    ("채권지수",             "idx/bon_dd_trd",        "추정"),
    ("석유시장",             "gen/oil_bydd_trd",      "추정"),
]

# 프로젝트가 '반드시' 기대는 필드 — 없으면 해당 장이 성립하지 않습니다
CRITICAL = {
    "etp/etf_bydd_trd": ["NAV", "OBJ_STKPRC_IDX", "IDX_IND_NM", "LIST_SHRS"],   # 3장 괴리율, 1장 자금흐름
    "etp/etn_bydd_trd": ["PER1SECU_INDIC_VAL"],                                # 3장 ETN 괴리
    "drv/opt_bydd_trd": ["IMP_VOLT", "PROD_NM", "RGHT_TP_NM"],                 # 2장 내재변동성
    "drv/fut_bydd_trd": ["SPOT_PRC", "SETL_PRC"],                              # 3장 베이시스
}

# 제공 시작 연도를 확인할 핵심 API
HISTORY = ["sto/stk_bydd_trd", "idx/kospi_dd_trd", "etp/etf_bydd_trd",
           "etp/etn_bydd_trd", "drv/opt_bydd_trd"]
YEARS = [1995, 2000, 2005, 2010, 2015]


def recent_weekday() -> str:
    t = datetime.date.today() - datetime.timedelta(days=1)
    while t.weekday() >= 5:
        t -= datetime.timedelta(days=1)
    return t.strftime("%Y%m%d")


def try_day(path: str, start: str, key: str, back: int = 6):
    """start 부터 하루씩 거슬러 올라가며 행이 있는 날을 찾습니다."""
    d = datetime.datetime.strptime(start, "%Y%m%d").date()
    for _ in range(back):
        ds = d.strftime("%Y%m%d")
        rows = call(f"{BASE}/{path}", ds, key)
        time.sleep(SLEEP)
        if rows:
            return ds, rows
        d -= datetime.timedelta(days=1)
    return None, []


def filled_ratio(rows, field):
    """그 필드가 실제 숫자로 채워진 행의 비율 (빈칸·0 제외)."""
    if not rows:
        return 0.0
    ok = sum(1 for r in rows if (v := num(r.get(field))) == v and v != 0)
    return ok / len(rows)


def check_fields(key: str, day: str) -> dict:
    print("\n" + "═" * 80)
    print(f"  ① 필드 점검 — 기준일 {day} 부근")
    print("═" * 80)
    out = {}
    for name, path, src in ENDPOINTS:
        tag = f"{name} [{src}]"
        try:
            d, rows = try_day(path, day, key)
        except Unauthorized as e:
            print(f"\n  ✗ {tag:<26} 401/404 — 미승인이거나 경로가 다름 ({path})")
            out[path] = {"status": "unauthorized_or_wrong_path"}
            continue
        except Exception as e:
            print(f"\n  ✗ {tag:<26} {type(e).__name__}: {e}")
            out[path] = {"status": "error", "detail": str(e)}
            continue

        if not rows:
            print(f"\n  ? {tag:<26} 6일 동안 행 0개")
            out[path] = {"status": "empty"}
            continue

        fields = list(rows[0].keys())
        print(f"\n  ✓ {tag:<26} {d} · {len(rows):,}행")
        print(f"      필드: {', '.join(fields)}")
        rec = {"status": "ok", "date": d, "rows": len(rows), "fields": fields}

        need = CRITICAL.get(path, [])
        if need:
            rec["critical"] = {}
            for f in need:
                if f not in fields:
                    print(f"      ✗ {f:<20} 필드 없음  ← 이 필드에 기대는 장이 성립하지 않습니다")
                    rec["critical"][f] = None
                    continue
                if f in ("IDX_IND_NM", "PROD_NM", "RGHT_TP_NM"):
                    nonblank = sum(1 for r in rows if str(r.get(f, "")).strip()) / len(rows)
                    print(f"      ✓ {f:<20} 채워진 행 {nonblank:.0%}")
                    rec["critical"][f] = nonblank
                else:
                    fr = filled_ratio(rows, f)
                    mark = "✓" if fr > 0.5 else "△"
                    print(f"      {mark} {f:<20} 숫자로 채워진 행 {fr:.0%}")
                    rec["critical"][f] = fr

            # 옵션: KOSPI200 옵션이 들어 있는지
            if path == "drv/opt_bydd_trd" and "PROD_NM" in fields:
                prods = sorted({str(r.get("PROD_NM", "")).strip() for r in rows})
                print(f"      상품 종류: {prods[:12]}{' ...' if len(prods) > 12 else ''}")
                rec["products"] = prods

            # ETF: 대표 샘플 한 줄
            if path == "etp/etf_bydd_trd":
                r0 = rows[0]
                print(f"      예시: {r0.get('ISU_NM')} 종가={r0.get('TDD_CLSPRC')} "
                      f"NAV={r0.get('NAV')} 기초지수={r0.get('IDX_IND_NM')}")
        out[path] = rec
    return out


def check_history(key: str) -> dict:
    print("\n" + "═" * 80)
    print("  ② 제공 시작 연도 점검 — 각 연도 3월 중순 부근에 데이터가 있는가")
    print("═" * 80)
    out = {}
    for path in HISTORY:
        name = next(n for n, p, _ in ENDPOINTS if p == path)
        marks = []
        out[path] = {}
        for y in YEARS:
            try:
                d, rows = try_day(path, f"{y}0315", key, back=5)
            except Unauthorized:
                marks.append(f"{y}:✗")
                out[path][y] = None
                continue
            ok = bool(rows)
            marks.append(f"{y}:{'✓' if ok else '·'}{'(' + str(len(rows)) + ')' if ok else ''}")
            out[path][y] = len(rows) if ok else 0
        print(f"  {name:<18} " + "  ".join(marks))
    print("\n  ✓ = 데이터 있음(행 수)   · = 없음   ✗ = 접근 불가")
    return out


def main():
    ap = argparse.ArgumentParser(description="KRX Open API 0단계 점검")
    ap.add_argument("--fields", action="store_true", help="필드 점검만")
    args = ap.parse_args()

    key = auth_key()
    report = {"run_at": datetime.datetime.now().isoformat(timespec="seconds")}
    report["fields"] = check_fields(key, recent_weekday())
    if not args.fields:
        report["history"] = check_history(key)

    with open("probe_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n" + "═" * 80)
    print("  저장: probe_report.json")
    print("  ✗ 나 △ 가 있으면 그 줄을 그대로 보내주세요. 설계를 그에 맞게 고칩니다.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()
