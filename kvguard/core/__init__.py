"""Shared safety core: object model, shadow ledger, lifecycle automaton, snapshot differencing,
dispatch epochs and violation reporting.

Adapters build :class:`Snapshot` and :class:`DispatchDescriptor` objects from framework state and
feed them, together with mediated allocator events, to a :class:`Ledger`. The ledger makes every
P1–P4 decision and reports through a :class:`Reporter`.
"""

from .ledger import BlockShadow, Epoch, Ledger, RequestShadow
from .model import (
    BlockId,
    BlockSnapshot,
    BlockState,
    CSRPageList,
    DenseBlockTable,
    DispatchDescriptor,
    DispatchRow,
    ExternalCopy,
    ExternalFill,
    ExternalRead,
    ExternalWrite,
    MappingRepresentation,
    PoolGeometry,
    PoolId,
    PositionedTokenTable,
    RequestId,
    RequestSnapshot,
    SlotId,
    SlotMap,
    Snapshot,
    TokenTable,
)
from .violations import (
    IntegrationError,
    KVGuardError,
    KVGuardViolation,
    OutcomeKind,
    Policy,
    Report,
    Reporter,
    Unsupported,
)

__all__ = [
    "BlockId",
    "BlockShadow",
    "BlockSnapshot",
    "BlockState",
    "CSRPageList",
    "DenseBlockTable",
    "DispatchDescriptor",
    "DispatchRow",
    "Epoch",
    "ExternalCopy",
    "ExternalFill",
    "ExternalRead",
    "ExternalWrite",
    "IntegrationError",
    "KVGuardError",
    "KVGuardViolation",
    "Ledger",
    "MappingRepresentation",
    "OutcomeKind",
    "Policy",
    "PoolGeometry",
    "PoolId",
    "PositionedTokenTable",
    "Report",
    "Reporter",
    "RequestId",
    "RequestShadow",
    "RequestSnapshot",
    "SlotId",
    "SlotMap",
    "Snapshot",
    "TokenTable",
    "Unsupported",
]
