"""Normalization logic for temporal expressions to ISO-like values."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from .schemas import DocumentContext, NormalizedTemporal, TemporalExpression, TemporalExpressionType

MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# Precompiled month name regex for range detection; kept at module scope for staticmethod visibility
MONTH_LOOKUP_REGEX = "(" + "|".join(MONTH_LOOKUP.keys()) + ")"


_DURATION_TOKEN_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>years?|yrs?|y|months?|mo|weeks?|w|days?|d|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)",
    re.IGNORECASE,
)


def _format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


class TemporalNormalizer:
    """Convert tagged expressions into normalized temporal forms."""

    def __init__(self, default_reference: Optional[datetime] = None):
        self.default_reference = default_reference or datetime.now(timezone.utc)

    def normalize(self, expression: TemporalExpression, context: Optional[DocumentContext] = None) -> NormalizedTemporal:
        reference = (context.reference_time if context and context.reference_time else self.default_reference).replace(hour=0, minute=0, second=0, microsecond=0)
        text = expression.text.strip()
        lowered = text.lower()
        errors = []

        normalized = None
        granularity = expression.granularity
        alternatives = []

        try:
            # Prefer routing by explicit relative cues for DATE expressions
            if expression.expr_type == TemporalExpressionType.DATE:
                if self._is_relative_with_time(lowered):
                    normalized, granularity = self._resolve_relative_with_time(lowered, reference), "minute"
                elif lowered.startswith("last ") or lowered.startswith("next "):
                    if any(day in lowered for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
                        normalized, granularity = self._resolve_relative_weekday(lowered, reference), "day"
                elif lowered in {"today", "yesterday", "tomorrow", "last week", "next week", "last month", "next month", "last year", "next year"}:
                    normalized, granularity = self._resolve_simple_relative(lowered, reference), "day"

            if self._looks_like_iso_date(lowered):
                normalized, granularity = lowered, "day"
            elif self._looks_like_month_name(lowered):
                normalized, granularity = self._parse_month_name(text), "day"
            elif self._looks_like_month_range(lowered):
                normalized, granularity = self._parse_month_range(text), "day"
            elif self._looks_like_iso_range(lowered):
                normalized, granularity = self._parse_iso_range(lowered), "day"
            elif self._looks_like_clock_time(lowered):
                normalized, granularity = self._normalize_time(lowered), "minute"
            elif self._is_fuzzy_time(lowered):
                normalized, granularity = self._normalize_fuzzy_time(lowered), "minute"
            elif self._is_relative_with_time(lowered):
                normalized, granularity = self._resolve_relative_with_time(lowered, reference), "minute"
            elif self._is_relative_simple(lowered):
                normalized, granularity = self._resolve_simple_relative(lowered, reference), "day"
            elif self._is_relative_weekday(lowered):
                normalized, granularity = self._resolve_relative_weekday(lowered, reference), "day"
            elif self._is_recurring(lowered):
                normalized, granularity = self._normalize_recurring(lowered), "set"
            elif self._looks_like_duration(lowered):
                normalized, granularity = self._normalize_duration(lowered), "duration"
            else:
                errors.append("unhandled_expression")
        except Exception as exc:  # pragma: no cover - defensive guard
            errors.append(str(exc))

        # Best-effort fallback based on expression type if primary routing failed
        if normalized is None:
            try:
                if expression.expr_type == TemporalExpressionType.TIME:
                    normalized, granularity = self._normalize_time(lowered), "minute"
                elif expression.expr_type == TemporalExpressionType.DURATION:
                    if self._looks_like_duration(lowered):
                        normalized, granularity = self._normalize_duration(lowered), "duration"
                elif expression.expr_type == TemporalExpressionType.DATE:
                    if " at " in lowered:
                        normalized, granularity = self._resolve_relative_with_time(lowered, reference), "minute"
                    elif self._looks_like_month_range(lowered):
                        normalized, granularity = self._parse_month_range(text), "day"
                    elif self._looks_like_iso_date(lowered):
                        normalized, granularity = lowered, "day"
                elif expression.expr_type == TemporalExpressionType.SET:
                    if self._is_recurring(lowered):
                        normalized, granularity = self._normalize_recurring(lowered), "set"
            except Exception as exc:  # pragma: no cover - defensive guard
                errors.append(str(exc))

        return NormalizedTemporal(
            expression=expression,
            normalized=normalized,
            expr_type=expression.expr_type,
            granularity=granularity,
            alternatives=alternatives,
            reference_time=reference,
            errors=errors,
        )

    @staticmethod
    def _looks_like_iso_date(text: str) -> bool:
        parts = text.split("-")
        return len(parts) == 3 and all(part.isdigit() for part in parts)

    @staticmethod
    def _looks_like_month_name(text: str) -> bool:
        return any(text.lower().startswith(name) for name in MONTH_LOOKUP.keys())

    @staticmethod
    def _looks_like_month_range(text: str) -> bool:
        return bool(re.match(rf"^{MONTH_LOOKUP_REGEX}\s+\d{{1,2}}-\d{{1,2}},\s*\d{{4}}$", text))

    @staticmethod
    def _parse_month_range(text: str) -> str:
        # "January 5-7, 2025" -> "2025-01-05/2025-01-07"
        cleaned = text.replace(",", "")
        tokens = cleaned.split()
        if len(tokens) < 3:
            raise ValueError("Unexpected month range format")
        month = MONTH_LOOKUP[tokens[0].lower()]
        day_part = tokens[1]
        year = int(tokens[2])
        if "-" not in day_part:
            raise ValueError("Expected day range with dash")
        start_day, end_day = [int(x) for x in day_part.split("-")]
        return f"{year:04d}-{month:02d}-{start_day:02d}/{year:04d}-{month:02d}-{end_day:02d}"

    @staticmethod
    def _looks_like_iso_range(text: str) -> bool:
        return " to " in text and all(part.strip() for part in text.split(" to "))

    @staticmethod
    def _parse_iso_range(text: str) -> str:
        parts = text.split(" to ")
        if len(parts) != 2:
            raise ValueError("Unexpected ISO range format")
        start, end = parts[0].strip(), parts[1].strip()
        return f"{start}/{end}"

    @staticmethod
    def _looks_like_clock_time(text: str) -> bool:
        return bool(re.match(r"^\d{1,2}:\d{2}$", text)) or bool(re.match(r"^\d{1,2}(:\d{2})?\s*(am|pm)$", text))

    @staticmethod
    def _normalize_time(text: str) -> str:
        text = text.strip().lower()
        if text.endswith("am") or text.endswith("pm"):
            meridiem = text[-2:]
            base = text[:-2].strip()
            if ":" in base:
                parts = base.split(":")
                hours = int(parts[0])
                minutes = int(parts[1])
            else:
                hours = int(base)
                minutes = 0
            if meridiem == "pm" and hours != 12:
                hours += 12
            if meridiem == "am" and hours == 12:
                hours = 0
        else:
            parts = text.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
        return f"{hours:02d}:{minutes:02d}"

    @staticmethod
    def _is_fuzzy_time(text: str) -> bool:
        return text in {"around noon", "around midnight"}

    @staticmethod
    def _normalize_fuzzy_time(text: str) -> str:
        if text == "around noon":
            return "12:00"
        if text == "around midnight":
            return "00:00"
        raise ValueError("Unhandled fuzzy time")

    @staticmethod
    def _parse_month_name(text: str) -> str:
        # Supports formats like "January 15, 2024"
        cleaned = text.replace(",", "")
        tokens = cleaned.split()
        if len(tokens) < 3:
            raise ValueError("Unexpected month-name expression format")
        month = MONTH_LOOKUP[tokens[0].lower()]
        day = int(tokens[1])
        year = int(tokens[2])
        return f"{year:04d}-{month:02d}-{day:02d}"

    @staticmethod
    def _is_relative_simple(text: str) -> bool:
        return text in {"today", "yesterday", "tomorrow", "last week", "next week", "last month", "next month", "last year", "next year"}

    @staticmethod
    def _is_relative_with_time(text: str) -> bool:
        return bool(re.match(r"^(today|tomorrow|yesterday)\s+at\s+\d{1,2}(:\d{2})?\s*(am|pm)?$", text))

    @staticmethod
    def _resolve_simple_relative(text: str, reference: datetime) -> str:
        if text == "today":
            return _format_date(reference)
        if text == "yesterday":
            return _format_date(reference - timedelta(days=1))
        if text == "tomorrow":
            return _format_date(reference + timedelta(days=1))
        if text == "last week":
            return _format_date(reference - timedelta(weeks=1))
        if text == "next week":
            return _format_date(reference + timedelta(weeks=1))
        if text == "last month":
            return _format_date(_shift_month(reference, -1))
        if text == "next month":
            return _format_date(_shift_month(reference, 1))
        if text == "last year":
            return _format_date(reference.replace(year=reference.year - 1))
        if text == "next year":
            return _format_date(reference.replace(year=reference.year + 1))
        raise ValueError("Unhandled relative expression")

    def _resolve_relative_with_time(self, text: str, reference: datetime) -> str:
        # text like "tomorrow at 3pm" or "today at 14:30"
        parts = text.split(" at ")
        if len(parts) != 2:
            raise ValueError("Unexpected relative-with-time format")
        day_part, time_part = parts[0], parts[1]
        date_str = self._resolve_simple_relative(day_part, reference)
        time_str = self._normalize_time(time_part)
        return f"{date_str}T{time_str}"

    @staticmethod
    def _is_relative_weekday(text: str) -> bool:
        is_relative = text.startswith("last ") or text.startswith("next ")
        return is_relative and any(day in text for day in [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ])

    @staticmethod
    def _resolve_relative_weekday(text: str, reference: datetime) -> str:
        direction, weekday = text.split()
        weekday_index = _weekday_index(weekday)
        current_index = reference.weekday()

        if direction == "last":
            delta = (current_index - weekday_index) or 7
            target = reference - timedelta(days=delta)
        else:
            delta = (weekday_index - current_index) or 7
            target = reference + timedelta(days=delta)
        return _format_date(target)

    @staticmethod
    def _is_recurring(text: str) -> bool:
        return text.startswith("every ") or text in {"daily", "every day", "weekly", "monthly", "yearly"} or "twice a" in text

    @staticmethod
    def _normalize_recurring(text: str) -> str:
        # Normalize to a simple canonical label; more advanced recurrence rules can be layered later.
        cleaned = text.replace("  ", " ").strip()
        return cleaned.lower().replace(" ", "-")

    @staticmethod
    def _looks_like_duration(text: str) -> bool:
        tokens = _extract_duration_tokens(text)
        return bool(tokens)

    @staticmethod
    def _normalize_duration(text: str) -> str:
        tokens = _extract_duration_tokens(text)
        if not tokens:
            raise ValueError("Unhandled duration expression")

        date_parts = {"Y": 0.0, "MO": 0.0, "W": 0.0, "D": 0.0}
        time_parts = {"H": 0.0, "MI": 0.0, "S": 0.0}

        for value, unit in tokens:
            val = float(value)
            category, normalized_unit = _normalize_duration_unit(unit)
            if category == "date":
                date_parts[normalized_unit] += val
            else:
                time_parts[normalized_unit] += val

        date_str = "".join(_format_part(date_parts[k], "M" if k == "MO" else k) for k in ["Y", "MO", "W", "D"] if date_parts[k])
        time_str = "".join(_format_part(time_parts[k], "M" if k == "MI" else k) for k in ["H", "MI", "S"] if time_parts[k])

        if not date_str and not time_str:
            raise ValueError("Duration tokens could not be normalized")

        iso = "P" + date_str
        if time_str:
            iso += "T" + time_str
        return iso


def _weekday_index(name: str) -> int:
    mapping = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    return mapping[name.lower()]


def _shift_month(dt: datetime, delta: int) -> datetime:
    month = dt.month - 1 + delta
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, _days_in_month(year, month))
    return dt.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    if month in {1, 3, 5, 7, 8, 10, 12}:
        return 31
    return 30


def _extract_duration_tokens(text: str):
    tokens = []
    for m in _DURATION_TOKEN_RE.finditer(text):
        tokens.append((m.group("num"), m.group("unit")))
    return tokens


def _normalize_duration_unit(unit: str) -> str:
    unit = unit.lower()
    if unit.startswith("y"):
        return "date", "Y"
    if unit.startswith("mo") or unit.startswith("mon"):
        return "date", "MO"
    if unit.startswith("w"):
        return "date", "W"
    if unit.startswith("d"):
        return "date", "D"
    if unit.startswith("h") or unit.startswith("hr"):
        return "time", "H"
    if unit.startswith("min") or unit == "m":
        return "time", "MI"
    if unit.startswith("s"):
        return "time", "S"
    raise ValueError("Unhandled duration unit")


def _format_part(value: float, suffix: str) -> str:
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value}{suffix}"
