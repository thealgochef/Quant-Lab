"""Enumerations shared across all agents and modules."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class AgentID(StrEnum):
    ORCHESTRATOR = "ORCH-001"
    DATA_INFRA = "DATA-001"
    SIGNAL_ENG = "SIG-001"
    VALIDATION = "VAL-001"
    EXECUTION = "EXEC-001"
    MONITORING = "MON-001"


class AgentState(StrEnum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"
    SHUTDOWN = "shutdown"


class PipelineState(StrEnum):
    INIT = "INIT"
    PHASE_1_2 = "PHASE_1_2"
    PHASE_3_4 = "PHASE_3_4"
    PHASE_5_6 = "PHASE_5_6"
    PHASE_7 = "PHASE_7"
    PHASE_8_9 = "PHASE_8_9"
    DEPLOYED = "DEPLOYED"
    HALT = "HALT"


class Priority(IntEnum):
    NORMAL = 1
    HIGH = 5
    CRITICAL = 10


class MessageType(StrEnum):
    DATA_REQUEST = "DATA_REQUEST"
    DATA_BUNDLE = "DATA_BUNDLE"
    SIGNAL_BUNDLE = "SIGNAL_BUNDLE"
    VALIDATION_REPORT = "VALIDATION_REPORT"
    REFINE_REQUEST = "REFINE_REQUEST"
    EXECUTION_REQUEST = "EXECUTION_REQUEST"
    EXECUTION_REPORT = "EXECUTION_REPORT"
    DEPLOY_COMMAND = "DEPLOY_COMMAND"
    ALERT = "ALERT"
    REGIME_SHIFT = "REGIME_SHIFT"
    RISK_VETO = "RISK_VETO"
    DAILY_REPORT = "DAILY_REPORT"
    HALT_COMMAND = "HALT_COMMAND"
    RESUME_COMMAND = "RESUME_COMMAND"
    ACK = "ACK"
    NACK = "NACK"


class Timeframe(StrEnum):
    TICK_987 = "987t"
    TICK_2000 = "2000t"
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M10 = "10m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"


class SignalTier(IntEnum):
    CORE = 1
    ICT_STRUCTURAL = 2
    COMPOSITE = 3


class SessionType(StrEnum):
    RTH = "RTH"
    GLOBEX = "GLOBEX"
    PRE_MARKET = "PRE_MARKET"
    POST_MARKET = "POST_MARKET"


class Killzone(StrEnum):
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    ASIA = "ASIA"
    OVERLAP = "OVERLAP"
    NONE = "NONE"


class Verdict(StrEnum):
    DEPLOY = "DEPLOY"
    REFINE = "REFINE"
    REJECT = "REJECT"


class ExecutionVerdict(StrEnum):
    APPROVED = "APPROVED"
    VETOED = "VETOED"


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    HALT = "HALT"


class Regime(StrEnum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    TRANSITIONAL = "TRANSITIONAL"


class SignalHealth(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADING = "DEGRADING"
    FAILING = "FAILING"


class DecayClass(StrEnum):
    ULTRA_FAST = "ultra-fast"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    PERSISTENT = "persistent"
