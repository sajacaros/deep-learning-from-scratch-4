#!/usr/bin/env python3
"""스터디 데일리 응원 문구를 Discord 웹훅으로 전송한다.

- 표준 라이브러리만 사용 (pip install 불필요)
- 모든 날짜 판정은 KST(UTC+9) 기준
- 사용법:
    DISCORD_WEBHOOK_URL=https://... python3 scripts/daily_cheer.py
    python3 scripts/daily_cheer.py --date 2026-08-16 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

KST = dt.timezone(dt.timedelta(hours=9), "KST")
REPO_URL = "https://github.com/sajacaros/deep-learning-from-scratch-4"
MESSAGES_PATH = Path(__file__).with_name("messages.json")

# (발표일, 챕터 번호, 챕터 제목, 저장소 내 코드 경로)
SCHEDULE = [
    ("2026-08-16", 1, "밴디트 문제", "ch01"),
    ("2026-08-23", 2, "마르코프 결정 과정", None),
    ("2026-08-30", 3, "벨만 방정식", None),
    ("2026-09-06", 4, "동적 프로그래밍", "ch04"),
    ("2026-09-13", 5, "몬테카를로법", "ch05"),
    ("2026-09-20", 6, "TD법", "ch06"),
    ("2026-10-04", 7, "신경망과 Q러닝", "ch07"),
    ("2026-10-11", 8, "DQN", "ch08"),
    ("2026-10-18", 9, "정책 경사법", "ch09"),
    ("2026-10-25", 10, "한 걸음 더", None),
]

WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

# type 별 표시 설정: (임베드 색상, 헤더 접두 이모지)
TYPE_STYLE = {
    "study": (0x5865F2, "🎯"),   # 발표일
    "holiday": (0xF4B942, "🎏"),  # 공휴일
    "normal": (0x57F287, "📗"),   # 평일/주말
}


def load_messages() -> dict:
    with MESSAGES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def next_session(today: dt.date):
    """오늘 이후(오늘 포함) 가장 가까운 발표일 정보를 돌려준다. 없으면 None."""
    for iso, num, title, path in SCHEDULE:
        d = dt.date.fromisoformat(iso)
        if d >= today:
            return d, num, title, path
    return None


def build_embed(today: dt.date, entry: dict) -> dict:
    kind = entry.get("type", "normal")
    color, emoji = TYPE_STYLE.get(kind, TYPE_STYLE["normal"])

    weekday = WEEKDAY_KR[today.weekday()]
    date_label = f"{today.month}/{today.day}({weekday})"

    nxt = next_session(today)
    if nxt is None:
        title = f"{emoji} {date_label} · 스터디 완주 🎉"
        footer = "10주간 정말 고생 많으셨습니다!"
    else:
        d, num, chapter_title, path = nxt
        dday = (d - today).days
        dday_label = "D-DAY" if dday == 0 else f"D-{dday}"
        title = f"{emoji} {date_label} · Chapter {num}. {chapter_title} {dday_label}"
        footer = f"{path}/" if path else ""

    embed = {
        "title": title,
        "description": entry["msg"],
        "color": color,
        "url": REPO_URL,
    }
    if footer:
        embed["footer"] = {"text": footer}
    return embed


def post(webhook: str, embed: dict) -> None:
    payload = json.dumps(
        {"username": "밑시딥4 스터디봇", "embeds": [embed]},
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        webhook,
        data=payload,
        headers={
            "Content-Type": "application/json",
            # 기본 UA(Python-urllib/x.y)는 Discord 앞단 Cloudflare가 403으로 막는다.
            "User-Agent": "daily-cheer-bot/1.0 (+https://github.com/sajacaros/deep-learning-from-scratch-4)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        if resp.status not in (200, 204):
            raise RuntimeError(f"Discord 응답 코드 {resp.status}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (기본값: 오늘 KST)")
    ap.add_argument("--dry-run", action="store_true", help="전송하지 않고 출력만")
    args = ap.parse_args()

    today = (
        dt.date.fromisoformat(args.date)
        if args.date
        else dt.datetime.now(KST).date()
    )

    messages = load_messages()
    entry = messages.get(today.isoformat())
    if entry is None:
        print(f"[skip] {today} 에 해당하는 문구가 없습니다. 전송하지 않습니다.")
        return 0

    embed = build_embed(today, entry)

    if args.dry_run:
        print(json.dumps(embed, ensure_ascii=False, indent=2))
        return 0

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("[error] DISCORD_WEBHOOK_URL 환경변수가 비어 있습니다.", file=sys.stderr)
        return 1

    try:
        post(webhook, embed)
    except urllib.error.HTTPError as e:
        print(f"[error] HTTP {e.code}: {e.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print(f"[ok] {today} 전송 완료: {embed['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
