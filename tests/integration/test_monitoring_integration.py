"""Integration tests for MON-001 deployment, alerts, and regime detection."""

from __future__ import annotations

from alpha_lab.core.enums import AgentID, MessageType, PipelineState, Regime
from alpha_lab.core.message import MessageEnvelope


class TestMonitoringDeployment:
    def test_deploy_command_via_bus(self, full_pipeline_bus):
        """DEPLOY_COMMAND through bus initializes MON-001 tracking."""
        bus, agents = full_pipeline_bus
        mon = agents["mon"]

        env = MessageEnvelope(
            request_id="deploy-int-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.MONITORING,
            message_type=MessageType.DEPLOY_COMMAND,
            payload={
                "approved_signals": [
                    {
                        "signal_id": "SIG_EMA_v1",
                        "ic": 0.06,
                        "hit_rate": 0.57,
                        "sharpe": 1.8,
                        "risk_parameters": {"max_contracts": 4},
                    },
                    {
                        "signal_id": "SIG_KAMA_v1",
                        "ic": 0.04,
                        "hit_rate": 0.54,
                        "sharpe": 1.3,
                        "risk_parameters": {"max_contracts": 2},
                    },
                ]
            },
        )
        bus.send(env)

        assert len(mon.active_signals) == 2
        assert "SIG_EMA_v1" in mon.active_signals
        assert "SIG_KAMA_v1" in mon.active_signals
        assert mon.active_signals["SIG_EMA_v1"].backtest_ic == 0.06

    def test_alert_escalation_triggers_halt(self, full_pipeline_bus):
        """MON-001 HALT alert escalation causes ORCH to halt pipeline."""
        bus, agents = full_pipeline_bus
        orch = agents["orch"]

        # Send alert directly as MON-001 would
        env = MessageEnvelope(
            request_id="alert-halt-001",
            sender=AgentID.MONITORING,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.ALERT,
            payload={
                "level": "HALT",
                "source": "MON-001",
                "issue": "Daily loss buffer at 15%",
            },
        )
        bus.send(env)

        assert orch.pipeline.current_state == PipelineState.HALT

    def test_regime_shift_message_sent(self, full_pipeline_bus):
        """MON-001 sends REGIME_SHIFT to ORCH on regime transition."""
        bus, agents = full_pipeline_bus
        mon = agents["mon"]

        # Start in RANGING (default), transition to TRENDING
        trending_data = {
            "ema_values": [22200.0, 22100.0, 22000.0],
            "kama_slope": 0.5,
            "atr_current": 50.0,
            "atr_avg": 50.0,
            "adx": 30.0,
        }
        transition = mon.update_regime(trending_data)
        assert transition is not None
        assert mon.current_regime == Regime.TRENDING

        # Verify REGIME_SHIFT message was sent
        audit = bus.get_audit_log()
        regime_msgs = [
            e for e in audit
            if e.message_type == MessageType.REGIME_SHIFT
            and e.sender == AgentID.MONITORING
        ]
        assert len(regime_msgs) >= 1

    def test_daily_report_after_deploy(self, full_pipeline_bus):
        """Deploy -> update metrics -> daily report has correct structure."""
        bus, agents = full_pipeline_bus
        mon = agents["mon"]

        # Deploy a signal
        env = MessageEnvelope(
            request_id="deploy-report-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.MONITORING,
            message_type=MessageType.DEPLOY_COMMAND,
            payload={
                "approved_signals": [
                    {"signal_id": "SIG_TEST", "ic": 0.05, "hit_rate": 0.55, "sharpe": 1.5}
                ]
            },
        )
        bus.send(env)

        # Update metrics
        mon.update_metrics(
            "SIG_TEST",
            live_ic=0.04,
            live_hit_rate=0.53,
            live_sharpe=1.1,
            trades_today=10,
            gross_pnl_today=200.0,
            net_pnl_today=150.0,
        )

        # Generate daily report
        report = mon.generate_daily_report()
        assert report.report_type == "DAILY_SUMMARY"
        assert len(report.signals) == 1
        assert report.signals[0].signal_id == "SIG_TEST"
        assert report.signals[0].live_ic == 0.04
        assert report.regime["current"] is not None

    def test_monitoring_healthy_signals_no_alerts(self, full_pipeline_bus):
        """Healthy signal metrics produce zero alerts."""
        bus, agents = full_pipeline_bus
        mon = agents["mon"]

        # Deploy
        env = MessageEnvelope(
            request_id="deploy-healthy-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.MONITORING,
            message_type=MessageType.DEPLOY_COMMAND,
            payload={
                "approved_signals": [
                    {"signal_id": "SIG_HEALTHY", "ic": 0.05, "hit_rate": 0.55, "sharpe": 1.5}
                ]
            },
        )
        bus.send(env)

        # Update with healthy metrics
        mon.update_metrics(
            "SIG_HEALTHY",
            live_ic=0.04,  # 80% of backtest (above 50% threshold)
            live_hit_rate=0.56,
            live_sharpe=1.2,
        )

        alerts = mon.evaluate_alerts()
        assert len(alerts) == 0
