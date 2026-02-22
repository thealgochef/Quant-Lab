"""Tests for configuration loading."""


from alpha_lab.core.config import Settings, load_settings


class TestConfigLoader:
    def test_load_settings(self):
        settings = load_settings()
        assert isinstance(settings, Settings)

    def test_instruments_loaded(self):
        settings = load_settings()
        assert "NQ" in settings.instruments
        assert "ES" in settings.instruments
        assert settings.instruments["NQ"].tick_size == 0.25
        assert settings.instruments["NQ"].tick_value == 5.00
        assert settings.instruments["ES"].tick_value == 12.50

    def test_prop_firms_loaded(self):
        settings = load_settings()
        assert "apex_50k" in settings.prop_firms
        assert "topstep_50k" in settings.prop_firms
        assert settings.prop_firms["apex_50k"].trailing_max_drawdown == 2500
        assert settings.prop_firms["topstep_50k"].daily_loss_limit == 1000

    def test_killzones_loaded(self):
        settings = load_settings()
        assert "london" in settings.killzones
        assert "new_york" in settings.killzones
        assert settings.killzones["london"].start == "02:00"

    def test_validation_thresholds(self):
        settings = load_settings()
        assert settings.signal_thresholds.ic_tstat_min == 2.0
        assert settings.signal_thresholds.hit_rate_min == 0.51
        assert settings.signal_thresholds.sharpe_min == 1.0

    def test_execution_thresholds(self):
        settings = load_settings()
        assert settings.execution_thresholds.mc_ruin_probability_max == 0.05

    def test_portfolio_thresholds(self):
        settings = load_settings()
        assert settings.portfolio_thresholds.min_deploy_signals == 8

    def test_default_symbols(self):
        settings = load_settings()
        assert "NQ" in settings.symbols
        assert "ES" in settings.symbols
