"""Tests for interface contracts — all Pydantic models serialize and deserialize."""

from datetime import UTC, datetime

from alpha_lab.core.contracts import (
    Alert,
    CostAnalysis,
    PreviousDayLevels,
    PropFirmFeasibility,
    QualityReport,
    SessionMetadata,
    SignalHealthReport,
    SignalVector,
    SignalVerdict,
)


class TestQualityReport:
    def test_create_passing(self, sample_quality_report):
        assert sample_quality_report.passed is True
        assert sample_quality_report.total_bars == 5000

    def test_roundtrip_json(self, sample_quality_report):
        data = sample_quality_report.model_dump()
        restored = QualityReport.model_validate(data)
        assert restored.passed == sample_quality_report.passed
        assert restored.total_bars == sample_quality_report.total_bars


class TestSessionMetadata:
    def test_create(self, sample_session_metadata):
        assert sample_session_metadata.session_type == "RTH"
        assert sample_session_metadata.killzone == "NEW_YORK"

    def test_roundtrip_json(self, sample_session_metadata):
        data = sample_session_metadata.model_dump()
        restored = SessionMetadata.model_validate(data)
        assert restored.session_id == sample_session_metadata.session_id


class TestPreviousDayLevels:
    def test_create(self, sample_pd_levels):
        assert sample_pd_levels.pd_high > sample_pd_levels.pd_low
        assert sample_pd_levels.pd_mid == 22065.0

    def test_roundtrip_json(self, sample_pd_levels):
        data = sample_pd_levels.model_dump()
        restored = PreviousDayLevels.model_validate(data)
        assert restored.pd_high == sample_pd_levels.pd_high


class TestDataBundle:
    def test_create(self, sample_data_bundle):
        assert sample_data_bundle.instrument == "NQ"
        assert "5m" in sample_data_bundle.bars

    def test_has_request_id(self, sample_data_bundle):
        assert sample_data_bundle.request_id is not None
        assert len(sample_data_bundle.request_id) > 0


class TestSignalVector:
    def test_create(self, sample_signal_vector):
        assert sample_signal_vector.signal_id == "SIG_EMA_CONFLUENCE_5m_v1"
        assert sample_signal_vector.version == 1

    def test_roundtrip_json(self, sample_signal_vector):
        data = sample_signal_vector.model_dump()
        restored = SignalVector.model_validate(data)
        assert restored.signal_id == sample_signal_vector.signal_id


class TestSignalBundle:
    def test_create(self, sample_signal_bundle):
        assert sample_signal_bundle.instrument == "NQ"
        assert sample_signal_bundle.total_signals == 1
        assert len(sample_signal_bundle.signals) == 1


class TestSignalVerdict:
    def test_deploy_verdict(self):
        verdict = SignalVerdict(
            signal_id="SIG_TEST_v1",
            verdict="DEPLOY",
            ic=0.05,
            ic_tstat=2.5,
            hit_rate=0.55,
            hit_rate_long=0.57,
            hit_rate_short=0.53,
            sharpe=1.5,
            sortino=2.0,
            max_drawdown=0.08,
            profit_factor=1.5,
            decay_half_life=30.0,
            decay_class="medium",
            max_factor_corr=0.15,
            incremental_r2=0.01,
            is_orthogonal=True,
            subsample_stable=True,
        )
        assert verdict.verdict == "DEPLOY"
        assert len(verdict.failed_metrics) == 0

    def test_refine_verdict_with_failed_metrics(self):
        verdict = SignalVerdict(
            signal_id="SIG_TEST_v1",
            verdict="REFINE",
            ic=0.03,
            ic_tstat=1.7,
            hit_rate=0.52,
            hit_rate_long=0.53,
            hit_rate_short=0.51,
            sharpe=0.8,
            sortino=1.0,
            max_drawdown=0.12,
            profit_factor=1.1,
            decay_half_life=20.0,
            decay_class="fast",
            max_factor_corr=0.25,
            incremental_r2=0.008,
            is_orthogonal=True,
            subsample_stable=False,
            failed_metrics=[
                {
                    "metric": "ic_tstat", "value": 1.7,
                    "threshold": 2.0, "suggestion": "add confluence",
                },
            ],
        )
        assert verdict.verdict == "REFINE"
        assert len(verdict.failed_metrics) == 1


class TestCostAnalysis:
    def test_create(self):
        ca = CostAnalysis(
            gross_pnl=10000.0,
            total_costs=2000.0,
            net_pnl=8000.0,
            cost_drag_pct=0.20,
            gross_sharpe=1.5,
            net_sharpe=1.2,
            breakeven_hit_rate=0.48,
        )
        assert ca.net_pnl == 8000.0
        assert ca.cost_drag_pct == 0.20


class TestPropFirmFeasibility:
    def test_passing_check(self):
        pf = PropFirmFeasibility(
            worst_day_pnl=-800.0,
            max_trailing_dd=-1500.0,
            passes_daily_limit=True,
            passes_trailing_dd=True,
            consistency_score=0.85,
            mc_ruin_probability=0.02,
            passes_mc_check=True,
            recommended_contracts=2,
            kelly_fraction=0.15,
            half_kelly_contracts=1,
        )
        assert pf.passes_mc_check is True


class TestAlert:
    def test_create(self):
        alert = Alert(
            level="WARNING",
            metric="rolling_ic",
            current_value=0.02,
            threshold=0.025,
            backtest_value=0.05,
            message="IC degraded below 50% of backtest",
            recommended_action="Reduce position to 50%",
            timestamp=datetime.now(UTC).isoformat(),
        )
        assert alert.level == "WARNING"


class TestSignalHealthReport:
    def test_create(self):
        shr = SignalHealthReport(
            signal_id="SIG_TEST_v1",
            live_ic=0.04,
            backtest_ic=0.05,
            ic_ratio=0.80,
            live_hit_rate=0.53,
            live_sharpe=1.2,
            gross_pnl_today=500.0,
            net_pnl_today=450.0,
            trades_today=5,
            realized_slippage=2.5,
            status="HEALTHY",
        )
        assert shr.status == "HEALTHY"
