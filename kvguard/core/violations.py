"""Outcomes and reporting.

Every event the core judges ends in exactly one outcome: a P1–P4 **violation**, an **integration
error** (a declared path did not supply the facts a check needs), **unsupported** (rejected at
start-up or dispatch), or it is fine. Reports name the first invalid transition or access.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


class Policy(enum.Enum):
    P1 = "P1 lifecycle"
    P2 = "P2 access"
    P3 = "P3 mapping"
    P4 = "P4 initialization"


class OutcomeKind(enum.Enum):
    VIOLATION = "violation"
    INTEGRATION_ERROR = "integration_error"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Report:
    """One outcome. ``kind`` names the check that fired (stable identifier, snake_case)."""

    outcome: OutcomeKind
    kind: str
    detail: str
    policy: Policy | None = None
    pool: str | None = None
    block: int | None = None
    slot: int | None = None
    generation: int | None = None
    request: str | None = None
    site: str | None = None
    """Where it was observed: a snapshot seq, a dispatch id, or an allocator event."""

    def key(self) -> tuple:
        """Deduplication key for drain mode: (kind, pool, block, slot)."""
        return (self.kind, self.pool, self.block, self.slot)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        where = " ".join(
            f"{k}={v}"
            for k, v in (
                ("pool", self.pool),
                ("block", self.block),
                ("slot", self.slot),
                ("gen", self.generation),
                ("request", self.request),
                ("site", self.site),
            )
            if v is not None
        )
        pol = f" [{self.policy.value}]" if self.policy else ""
        return f"{self.outcome.value}{pol} {self.kind}: {self.detail} ({where})"


class KVGuardError(Exception):
    """Raised in fail-fast mode. Carries the report."""

    def __init__(self, report: Report):
        super().__init__(str(report))
        self.report = report


class KVGuardViolation(KVGuardError):
    pass


class IntegrationError(KVGuardError):
    pass


class Unsupported(KVGuardError):
    pass


_EXC = {
    OutcomeKind.VIOLATION: KVGuardViolation,
    OutcomeKind.INTEGRATION_ERROR: IntegrationError,
    OutcomeKind.UNSUPPORTED: Unsupported,
}


@dataclass
class Reporter:
    """Fail-fast raises on the first report; drain collects with deduplication."""

    fail_fast: bool = True
    reports: list[Report] = field(default_factory=list)
    seen: set[tuple] = field(default_factory=set)
    duplicates: int = 0
    sink: Callable[[Report], None] | None = None

    def emit(self, report: Report) -> None:
        if self.fail_fast:
            raise _EXC[report.outcome](report)
        key = report.key()
        if key in self.seen:
            self.duplicates += 1
            return
        self.seen.add(key)
        self.reports.append(report)
        if self.sink is not None:
            self.sink(report)

    def extend(self, reports: Iterable[Report]) -> None:
        for r in reports:
            self.emit(r)

    def violations(self) -> list[Report]:
        return [r for r in self.reports if r.outcome is OutcomeKind.VIOLATION]

    def integration_errors(self) -> list[Report]:
        return [r for r in self.reports if r.outcome is OutcomeKind.INTEGRATION_ERROR]

    def clear(self) -> None:
        self.reports.clear()
        self.seen.clear()
        self.duplicates = 0


def violation(policy: Policy, kind: str, detail: str, **where) -> Report:
    return Report(OutcomeKind.VIOLATION, kind, detail, policy=policy, **where)


def integration_error(kind: str, detail: str, **where) -> Report:
    return Report(OutcomeKind.INTEGRATION_ERROR, kind, detail, **where)


def unsupported(kind: str, detail: str, **where) -> Report:
    return Report(OutcomeKind.UNSUPPORTED, kind, detail, **where)


__all__ = [
    "IntegrationError",
    "KVGuardError",
    "KVGuardViolation",
    "OutcomeKind",
    "Policy",
    "Report",
    "Reporter",
    "Unsupported",
    "integration_error",
    "unsupported",
    "violation",
]
