from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class CalendarWindow:
    start_time: str
    end_time: str
    timezone: str
    anchor_time: str


@dataclass(frozen=True)
class RelativePeriodContext:
    as_of_date: str
    timezone_name: str
    current_month: str
    previous_month: str
    same_month_last_year: str
    current_quarter_start: str
    current_quarter_number: int
    current_year: int
    previous_quarter_start: str
    previous_quarter_end: str
    previous_quarter_number: int
    previous_quarter_year: int
    same_quarter_last_year_start: str
    same_quarter_last_year_end: str
    current_year_start: str
    previous_year_start: str
    previous_year_end: str
    ltm_start: str


_RELATIVE_DATE_OFFSETS = {
    "today": 0,
    "tonight": 0,
    "今天": 0,
    "今日": 0,
    "tomorrow": 1,
    "tmr": 1,
    "明天": 1,
    "明日": 1,
    "yesterday": -1,
    "昨天": -1,
    "昨日": -1,
}


def parse_aware_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def current_turn_anchor_utc(*, env_getter=None) -> datetime | None:
    if env_getter is None:
        try:
            from gateway.session_context import get_session_env
        except Exception:
            get_session_env = None
        env_getter = get_session_env

    if env_getter is not None:
        for name in (
            "HERMES_SESSION_MESSAGE_CREATED_AT",
            "HERMES_SESSION_TURN_TIMESTAMP",
        ):
            parsed = parse_aware_utc(env_getter(name, ""))
            if parsed is not None:
                return parsed
    return None


def _anchor_for_resolution(anchor_utc: datetime | None) -> datetime | None:
    return anchor_utc.astimezone(timezone.utc) if anchor_utc is not None else current_turn_anchor_utc()


def _parse_clock(text: str) -> time | None:
    normalized = text.strip().lower().replace("上午", "am").replace("下午", "pm")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", normalized)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    suffix = match.group(3)
    if minute > 59:
        return None
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return time(hour=hour, minute=minute)


def _split_relative_datetime(value: str) -> tuple[int | None, str]:
    text = value.strip()
    lowered = text.lower()
    for token, offset in _RELATIVE_DATE_OFFSETS.items():
        if lowered.startswith(token):
            return offset, text[len(token):].strip()
    return None, text


def parse_local_datetime(value: str, timezone_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("datetime value is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M")
        except ValueError as exc:
            raise ValueError(f"Unsupported datetime format: {value}") from exc

    local_tz = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(local_tz)


def normalize_calendar_instant(
    *,
    value: str,
    timezone_name: str,
    anchor_utc: datetime | None = None,
    allow_past: bool = False,
) -> str:
    local_tz = ZoneInfo(timezone_name)
    anchor = _anchor_for_resolution(anchor_utc)
    anchor_local = anchor.astimezone(local_tz) if anchor is not None else None
    text = str(value or "").strip()
    if not text:
        raise ValueError("datetime value is required")

    relative_offset, remainder = _split_relative_datetime(text)
    if relative_offset is not None:
        if anchor_local is None:
            raise ValueError("relative date requires platform turn timestamp")
        clock = _parse_clock(remainder)
        if clock is None:
            raise ValueError(f"Unsupported relative datetime format: {value}")
        target_date = anchor_local.date() + timedelta(days=relative_offset)
        parsed = datetime.combine(target_date, clock, tzinfo=local_tz)
    else:
        clock = _parse_clock(text)
        if clock is not None:
            if anchor_local is None:
                raise ValueError("clock-only datetime requires platform turn timestamp")
            parsed = datetime.combine(anchor_local.date(), clock, tzinfo=local_tz)
        else:
            parsed = parse_local_datetime(text, timezone_name)

    if anchor_local is not None and not allow_past and parsed < anchor_local:
        raise ValueError(
            "datetime value is before the current turn time "
            f"({anchor_local.isoformat()}); resolve relative dates from the "
            "inbound message timestamp or pass allow_past=true for explicit past events"
        )
    return parsed.astimezone(local_tz).isoformat()


def normalize_calendar_slots(
    values: list[str],
    *,
    timezone_name: str,
    anchor_utc: datetime | None = None,
    allow_past: bool = False,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_calendar_instant(
            value=value,
            timezone_name=timezone_name,
            anchor_utc=anchor_utc,
            allow_past=allow_past,
        )
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def normalize_once_run_at(
    value: str,
    *,
    timezone_name: str,
    anchor_utc: datetime | None = None,
    allow_past: bool = False,
) -> str:
    return normalize_calendar_instant(
        value=value,
        timezone_name=timezone_name,
        anchor_utc=anchor_utc,
        allow_past=allow_past,
    )


def normalize_calendar_window(
    *,
    start_time: str,
    end_time: str,
    timezone_name: str,
    anchor_utc: datetime | None = None,
    allow_past: bool = False,
) -> CalendarWindow:
    anchor = _anchor_for_resolution(anchor_utc)
    start_dt = parse_local_datetime(
        normalize_calendar_instant(
            value=start_time,
            timezone_name=timezone_name,
            anchor_utc=anchor,
            allow_past=allow_past,
        ),
        timezone_name,
    )
    end_dt = parse_local_datetime(
        normalize_calendar_instant(
            value=end_time,
            timezone_name=timezone_name,
            anchor_utc=anchor,
            allow_past=allow_past,
        ),
        timezone_name,
    )
    if end_dt <= start_dt:
        raise ValueError("end_time must be later than start_time")
    return CalendarWindow(
        start_time=start_dt.isoformat(),
        end_time=end_dt.isoformat(),
        timezone=timezone_name,
        anchor_time=anchor.isoformat().replace("+00:00", "Z") if anchor is not None else "",
    )


def _quarter(month: int) -> int:
    return ((month - 1) // 3) + 1


def _month_shift(value: date, offset: int) -> date:
    month_index = (value.year * 12 + (value.month - 1)) + offset
    year = month_index // 12
    month = (month_index % 12) + 1
    return date(year, month, 1)


def _quarter_start(value: date) -> date:
    return date(value.year, ((_quarter(value.month) - 1) * 3) + 1, 1)


def resolve_relative_period_context(
    *,
    anchor_utc: datetime | None = None,
    timezone_name: str = "UTC",
) -> RelativePeriodContext:
    local_tz = ZoneInfo(timezone_name)
    anchor = _anchor_for_resolution(anchor_utc) or datetime.now(timezone.utc)
    current = anchor.astimezone(local_tz).date()
    current_month = date(current.year, current.month, 1)
    previous_month = _month_shift(current_month, -1)
    same_month_last_year = _month_shift(current_month, -12)
    current_quarter_start = _quarter_start(current)
    previous_quarter_start = _month_shift(current_quarter_start, -3)
    same_quarter_last_year = _month_shift(current_quarter_start, -12)
    current_year_start = date(current.year, 1, 1)
    previous_year_start = date(current.year - 1, 1, 1)
    return RelativePeriodContext(
        as_of_date=current.isoformat(),
        timezone_name=timezone_name,
        current_month=current_month.strftime("%Y-%m"),
        previous_month=previous_month.strftime("%Y-%m"),
        same_month_last_year=same_month_last_year.strftime("%Y-%m"),
        current_quarter_start=current_quarter_start.strftime("%Y-%m"),
        current_quarter_number=_quarter(current.month),
        current_year=current.year,
        previous_quarter_start=previous_quarter_start.strftime("%Y-%m"),
        previous_quarter_end=_month_shift(previous_quarter_start, 2).strftime("%Y-%m"),
        previous_quarter_number=_quarter(previous_quarter_start.month),
        previous_quarter_year=previous_quarter_start.year,
        same_quarter_last_year_start=same_quarter_last_year.strftime("%Y-%m"),
        same_quarter_last_year_end=_month_shift(same_quarter_last_year, 2).strftime("%Y-%m"),
        current_year_start=current_year_start.strftime("%Y-%m"),
        previous_year_start=previous_year_start.strftime("%Y-%m"),
        previous_year_end=date(current.year - 1, 12, 1).strftime("%Y-%m"),
        ltm_start=same_month_last_year.strftime("%Y-%m"),
    )
