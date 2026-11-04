"""Mutable descriptor buffers remain exclusive through asynchronous completion."""

from kvguard.core import Ledger, Policy, Reporter


def test_reusing_an_inflight_descriptor_resource_is_p3():
    reporter = Reporter(fail_fast=False)
    ledger = Ledger(reporter)
    ledger.begin_resource_use("zero-1", ("pinned:10", "device:20"), site="zero@1")
    ledger.begin_resource_use("zero-2", ("pinned:10", "device:20"), site="zero@2")

    assert [report.kind for report in reporter.reports] == ["resource_reused_in_flight"]
    assert reporter.reports[0].policy is Policy.P3
    assert ledger.resource_users == {
        "pinned:10": {"zero-1", "zero-2"},
        "device:20": {"zero-1", "zero-2"},
    }

    ledger.end_resource_use("zero-1", site="done@1")
    ledger.end_resource_use("zero-2", site="done@2")
    assert not ledger.resource_uses and not ledger.resource_users


def test_disjoint_resources_may_overlap_and_completed_storage_may_be_reused():
    reporter = Reporter(fail_fast=False)
    ledger = Ledger(reporter)
    ledger.begin_resource_use("zero-1", ("pinned:10", "device:20"))
    ledger.begin_resource_use("zero-2", ("pinned:11", "device:21"))
    ledger.end_resource_use("zero-1")
    ledger.begin_resource_use("zero-3", ("pinned:10", "device:20"))

    assert not reporter.reports


def test_resource_operation_lifecycle_errors_are_integration_errors():
    reporter = Reporter(fail_fast=False)
    ledger = Ledger(reporter)
    ledger.begin_resource_use("zero-1", ("pinned:10",))
    ledger.begin_resource_use("zero-1", ("pinned:11",))
    ledger.end_resource_use("absent")

    assert [report.kind for report in reporter.reports] == [
        "duplicate_resource_operation",
        "unknown_resource_operation",
    ]
    assert len(reporter.integration_errors()) == 2
