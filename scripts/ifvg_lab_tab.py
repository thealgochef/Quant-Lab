"""IFVG Lab dashboard tab (plan Part B): B1 Experiments + B2 Replay/verifier.

Thin streamlit layer — all data/chart-payload logic lives in the PURE module
``ifvg_lab_charts.py`` and the engine ``alpha_lab...ifvg.experiment``. Loaders
here only add ``st.cache_data`` keyed by (day, tags).

SEALED discipline: the replay day list is filtered IN the loader
(``ifvg_lab_charts.list_replay_days``); sealed-day replay appears only inside
the explicit "Sealed replay" expander of a selected saved run that has
ledgered sealed validations, scoped to that run. No sealed statistic renders
anywhere in this tab.

Read-only over ``data/databento`` — this module never writes there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ifvg_lab_charts as charts  # noqa: E402
import ifvg_recapture_job as recapture  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: E402
    SEALED_HOLDOUT_START,
    IfvgCaptureConfig,
    custom_session_capture_config,
)
from alpha_lab.agents.data_infra.ifvg.experiment import (  # noqa: E402
    IfvgDocDefaultsConfig,
    IfvgExperimentConfig,
    IfvgFilterConfig,
    IfvgModelConfig,
    IfvgScoringConfig,
    IfvgSlTpConfig,
    SessionWindow,
    delete_experiment,
    list_experiments,
    load_experiment,
    run_ifvg_experiment,
    run_sealed_validation,
    save_experiment,
    sealed_ledger_count,
)

_FAMILIES = ("fresh_fvg_continuation", "ifvg_retest")
_SESSION_NAMES = ("asia", "london", "ny", "none")
_TP_LOGGED = "(logged family default)"


# ── cached loaders (keyed by day + tags; pure functions underneath) ───────────


@st.cache_data(show_spinner=False)
def _cached_replay_days(symbol_dir: str, atag: str, ctag: str, sealed: bool) -> list[str]:
    return charts.list_replay_days(symbol_dir, atag, ctag, sealed=sealed)


@st.cache_data(show_spinner=False)
def _cached_day_payload(
    symbol_dir: str, atag: str, ctag: str, day: str, allow_sealed: bool
) -> dict:
    return charts.load_day_payload(symbol_dir, atag, ctag, day, allow_sealed=allow_sealed)


@st.cache_data(show_spinner=False)
def _cached_recomputed_gaps(
    symbol_dir: str, atag: str, ctag: str, day: str, timeframe_seconds: int, allow_sealed: bool
) -> pd.DataFrame:
    payload = _cached_day_payload(symbol_dir, atag, ctag, day, allow_sealed)
    return charts.recompute_day_gaps(payload["bars"], timeframe_seconds)


@st.cache_data(show_spinner=False)
def _cached_entry_dataset(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _cfg(profile: dict | None = None) -> tuple[IfvgCaptureConfig, str, str, str]:
    """Capture config for the selected profile (None/default = canonical)."""
    if profile is None or profile.get("is_default"):
        cfg = IfvgCaptureConfig()
    else:
        cfg = custom_session_capture_config(
            recapture.windows_to_times(
                {n: (w[0], w[1]) for n, w in profile["windows"].items()}
            )
        )
    symbol_dir = str(Path(cfg.data_dir) / cfg.symbol)
    return cfg, symbol_dir, cfg.artifacts_tag(), cfg.capture_tag()


def _profile_selector(st, key: str) -> dict | None:
    """Capture-profile dropdown (default first + ready custom re-captures).

    Sealed guards are IDENTICAL on every profile: the engine clamp and the
    replay-day loader filter key off SEALED_HOLDOUT_START, not off tags.
    """
    profiles = [p for p in recapture.list_profiles() if p.get("status") == "ready"]
    by_label = {f"{p['name']} · {p['ctag'][:8]}": p for p in profiles}
    options = list(by_label)
    _sanitize_select(st, key, options)
    choice = st.selectbox(
        "Capture profile", options, key=key,
        help="Selects which capture (session-window scheme) drives datasets, "
        "experiments and replay. Custom profiles come from the Profile "
        "re-capture job.",
    )
    return by_label.get(choice)


def _dataset_path(cfg: IfvgCaptureConfig, ctag: str) -> str:
    return str(Path(cfg.data_dir) / cfg.symbol / f"ifvg_entry_dataset_{ctag}.parquet")


def _capture_cfg_for_tag(ctag: str) -> IfvgCaptureConfig | None:
    """Resolve the capture config a run was SAVED against (default or a custom
    profile from the registry); None when the profile is unknown."""
    default = IfvgCaptureConfig()
    if ctag == default.capture_tag():
        return default
    for p in recapture.list_profiles():
        if p.get("ctag") == ctag and not p.get("is_default"):
            cfg = custom_session_capture_config(
                recapture.windows_to_times({n: (w[0], w[1]) for n, w in p["windows"].items()})
            )
            if cfg.capture_tag() == ctag:
                return cfg
    return None


# ── small render helpers ──────────────────────────────────────────────────────


def _fmt(value, pattern: str = "{:.2f}") -> str:
    if value is None:
        return "—"
    try:
        return pattern.format(value)
    except (TypeError, ValueError):
        return str(value)


def _ci(pair) -> str:
    if not pair:
        return "—"
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"

def _sanitize_select(st, key: str, options: list) -> None:
    """Drop a stale session value that is no longer among the options."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def _run_label(run: dict) -> str:
    stats = run.get("stats") or {}
    net = stats.get("net_r")
    n = stats.get("n_trades")
    pf = stats.get("profit_factor")
    created = (run.get("created_utc") or "")[:10]
    return (
        f"{run.get('name') or run['experiment_hash']} - {created or '?'} - "
        f"net R {_fmt(net, '{:.1f}')} / n {_fmt(n, '{}')} / PF {_fmt(pf, '{:.2f}')}"
        f" · {run['experiment_hash']}"
    )


def _run_picker(st, runs: list[dict], key: str, none_label: str) -> dict | None:
    by_label = {_run_label(r): r for r in runs}
    options = [none_label, *by_label]
    _sanitize_select(st, key, options)
    choice = st.selectbox(
        "Saved run", options, key=key, help="name - date - net R / n / PF · hash"
    )
    return by_label.get(choice)


def _render_config_panel(st, config_dict: dict, header: dict) -> None:
    """Every knob with its effective value; non-default values highlighted."""
    cols = st.columns(3)
    cols[0].caption(f"hash `{header.get('experiment_hash', '?')}`")
    cols[1].caption(f"capture `{header.get('capture_tag', '?')}`")
    cols[2].caption(f"created {header.get('created_utc', '?')}")
    defaults = charts.flatten_config(IfvgExperimentConfig().model_dump(mode="json"))
    flat = charts.flatten_config(config_dict)
    fields = sorted(set(defaults) | set(flat))
    frame = pd.DataFrame(
        {
            "field": fields,
            "value": [flat.get(f, "(absent)") for f in fields],
            "default": [defaults.get(f, "(absent)") for f in fields],
        }
    )
    frame["non_default"] = frame["value"] != frame["default"]
    frame = frame.sort_values(["non_default", "field"], ascending=[False, True])
    st.dataframe(
        frame.style.apply(
            lambda row: (
                ["background-color: rgba(230,159,0,0.25)"] * len(row)
                if row["non_default"]
                else [""] * len(row)
            ),
            axis=1,
        ),
        use_container_width=True,
        height=280,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  B1 — Experiments
# ══════════════════════════════════════════════════════════════════════════════


def render_ifvg_experiments_tab(st) -> None:
    profile = _profile_selector(st, key="ifl_exp_profile")
    cfg, symbol_dir, atag, ctag = _cfg(profile)
    ds = _cached_entry_dataset(_dataset_path(cfg, ctag))
    _profile_header(st, cfg, ctag, ds)
    if ds is None:
        st.warning("Entry dataset parquet not found — run the capture pipeline first.")
        _recapture_section(st)
        return

    runs = list_experiments()
    selected = _run_picker(st, runs, key="ifl_exp_run", none_label="(new experiment)")
    if selected is not None:
        _render_saved_run(st, selected)
        st.divider()

    st.subheader("Configure & run")
    config = _config_form(st, ds)
    if config is not None:
        _run_controls(st, cfg, ctag, ds, config, runs)

    last = st.session_state.get("ifl_last_run")
    if last is not None:
        st.divider()
        st.subheader("Last run result")
        _save_controls(st, last)
        _render_result(st, last["result"], key_prefix="last")

    st.divider()
    _history_and_compare(st, runs)

    st.divider()
    _recapture_section(st)


def _profile_header(st, cfg: IfvgCaptureConfig, ctag: str, ds: pd.DataFrame | None) -> None:
    st.markdown("### IFVG Lab — Experiments")
    cols = st.columns(4)
    cols[0].caption(f"profile `{cfg.profile_hash[:16]}`")
    cols[1].caption(f"capture tag `{ctag}`")
    if ds is not None:
        pre = ds[ds["trading_day"].astype(str) < SEALED_HOLDOUT_START]
        cols[2].caption(
            f"pre-seal: {len(pre)} rows / {pre['trading_day'].nunique()} days "
            f"(dataset total {len(ds)})"
        )
    cols[3].caption(
        f"SEALED: days >= {SEALED_HOLDOUT_START} excluded from every experiment "
        "(engine-enforced)"
    )
    # The GLOBAL trust meter: per-look, across ALL configs (later configs are
    # chosen with knowledge of earlier sealed results).
    looks = sealed_ledger_count()
    if looks == 0:
        st.success("Sealed range evaluated: 0 times — the holdout is intact.")
    else:
        st.warning(
            f"Sealed range evaluated: {looks} time(s) GLOBALLY (all configs). "
            "Every look after the first degrades the holdout — discount later "
            "sealed results accordingly."
        )


def _render_saved_run(st, run: dict) -> None:
    loaded = load_experiment(run["experiment_hash"])
    with st.expander("Run configuration", expanded=True):
        _render_config_panel(
            st,
            loaded["config"].model_dump(mode="json"),
            {
                "experiment_hash": loaded["experiment_hash"],
                "capture_tag": loaded["capture_tag"],
                "created_utc": (loaded.get("meta") or {}).get("created_utc"),
            },
        )
        note = (loaded.get("meta") or {}).get("note")
        if note:
            st.caption(f"note: {note}")
        seqs = run.get("sealed_validations") or []
        if seqs:
            st.caption(
                "sealed history: " + ", ".join(f"sealed validation #{s}" for s in seqs)
            )
    cols = st.columns([1, 1, 2])
    if cols[0].button("Load config into form", key="ifl_load_into_form"):
        _seed_form_state(st, loaded["config"].model_dump(mode="json"))
        st.rerun()
    confirm = cols[2].checkbox("confirm delete", key="ifl_del_confirm")
    if cols[1].button("Delete run", key="ifl_del_btn", disabled=not confirm):
        try:
            delete_experiment(run["experiment_hash"])
            st.success("Run deleted.")
            st.rerun()
        except ValueError as exc:  # sealed-refusal from the engine
            st.error(f"Refused: {exc}")
    _sealed_validation_controls(st, run, loaded)
    if loaded.get("result") is not None:
        _render_result(st, loaded["result"], key_prefix="saved")


def _sealed_validation_controls(st, run: dict, loaded: dict) -> None:
    """The one-shot holdout evaluation: type-to-confirm friction, global look
    count, permanent ledger — and the run's sealed results rendered after."""
    with st.expander("Sealed validation — one-shot holdout evaluation"):
        looks = sealed_ledger_count()
        st.warning(
            f"The sealed range (days >= {SEALED_HOLDOUT_START}) can only give ONE "
            f"unbiased answer. It has been evaluated {looks} time(s) globally across "
            "all configs; every evaluation is permanently recorded on the append-only "
            "ledger and every look after the first degrades the holdout. There is no "
            "way to un-see the result."
        )
        typed = st.text_input(
            f"Type the run hash `{run['experiment_hash']}` to confirm",
            key="ifl_sealed_confirm_text",
        )
        run_cfg = _capture_cfg_for_tag(loaded["capture_tag"])
        if run_cfg is None:
            st.error(
                f"This run's capture profile (`{loaded['capture_tag']}`) is not in the "
                "profile registry — cannot sealed-validate it."
            )
        if st.button(
            "Run sealed validation (permanent)",
            key="ifl_sealed_run_btn",
            disabled=typed.strip() != run["experiment_hash"] or run_cfg is None,
        ):
            with st.spinner("Evaluating the sealed holdout..."):
                # capture_cfg from the RUN's stored tag (engine hard-guards the
                # match) — never from the dropdown's selected profile.
                path = run_sealed_validation(run["experiment_hash"], capture_cfg=run_cfg)
            st.success(f"Sealed validation recorded -> {path.name}")
            st.rerun()
        seqs = sorted(run.get("sealed_validations") or [])
        for seq in seqs:
            result_path = Path(loaded["run_dir"]) / f"sealed_result_{seq}.json"
            if not result_path.exists():
                continue
            st.markdown(f"**Sealed validation #{seq}** (global sequence)")
            st.json(json.loads(result_path.read_text(encoding="utf-8")), expanded=False)


# ── the form (widget -> IfvgExperimentConfig, EVERY field) ────────────────────


def _seed_form_state(st, config: dict) -> None:
    """Seed widget session-state from a saved config (load-into-form)."""
    ss = st.session_state
    f = config.get("filters") or {}
    ss["ifl_sessions_engine"] = list(f.get("sessions_engine") or [])
    ss["ifl_sessions_doc"] = list(f.get("sessions_doc") or [])
    ss["ifl_direction"] = f.get("direction", "both")
    ss["ifl_families"] = list(f.get("families") or _FAMILIES)
    ss["ifl_include_warmup"] = bool(f.get("include_warmup"))
    ss["ifl_day_start"] = f.get("day_start") or "(none)"
    ss["ifl_day_end"] = f.get("day_end") or "(none)"
    ss["ifl_max_tpd"] = int(f.get("max_trades_per_day") or 0)
    cs = config.get("custom_sessions")
    ss["ifl_custom_enable"] = cs is not None
    for name in ("asia", "london", "ny"):
        default = charts.ENGINE_SESSION_WINDOWS.get(name, ("00:00", "00:00"))
        window = (cs or {}).get(name)
        ss[f"ifl_cs_{name}_start"] = (window or {}).get("start", default[0])
        ss[f"ifl_cs_{name}_end"] = (window or {}).get("end", default[1])
    ss["ifl_sessions_custom"] = list(config.get("sessions_custom") or [])
    dd = config.get("doc_defaults") or {}
    dd_defaults = IfvgDocDefaultsConfig().model_dump()
    ss["ifl_dd_apply"] = bool(dd.get("apply"))
    for field in (
        "tap_fvg_size_min",
        "parent_fvg_size_min",
        "opp_fvg_size_min",
        "parent_distance_max",
        "opp_distance_max",
        "bars_since_inversion_max",
    ):
        ss[f"ifl_dd_{field}"] = float(dd.get(field, dd_defaults[field]))
    sc = config.get("scoring") or {}
    ss["ifl_r_family"] = sc.get("r_family", "r10")
    ss["ifl_cost"] = float(sc.get("cost_points", IfvgScoringConfig().cost_points))
    sl = config.get("sl_tp") or {}
    ss["ifl_tp_mode"] = sl.get("tp_mode") or _TP_LOGGED
    ss["ifl_tp_value"] = float(sl.get("tp_value") or 1.0)
    ss["ifl_sl_mode"] = sl.get("sl_mode") or "logged"
    ss["ifl_sl_value"] = float(sl.get("sl_value") or 5.0)
    m = config.get("model")
    ss["ifl_model_on"] = m is not None
    md = IfvgModelConfig().model_dump()
    m = m or md
    ss["ifl_m_iterations"] = int(m.get("iterations", md["iterations"]))
    ss["ifl_m_depth"] = int(m.get("depth", md["depth"]))
    ss["ifl_m_lr"] = float(m.get("learning_rate", md["learning_rate"]))
    ss["ifl_m_seed"] = int(m.get("seed", md["seed"]))
    ss["ifl_m_splits"] = ",".join(str(v) for v in m.get("split_fracs", md["split_fracs"]))
    ss["ifl_m_purge"] = int(m.get("purge_days", md["purge_days"]))
    ss["ifl_m_thresholds"] = ",".join(str(v) for v in m.get("thresholds", md["thresholds"]))


def _seed_default(st, key: str, value) -> None:
    """Widget defaults via session_state (avoids the default-vs-state warning
    and lets load-into-form seed every widget uniformly)."""
    if key not in st.session_state:
        st.session_state[key] = value


def _config_form(st, ds: pd.DataFrame) -> IfvgExperimentConfig | None:
    pre_days = sorted(
        ds.loc[ds["trading_day"].astype(str) < SEALED_HOLDOUT_START, "trading_day"]
        .astype(str)
        .unique()
    )
    day_options = ["(none)", *pre_days]

    dd_defaults = IfvgDocDefaultsConfig()
    md = IfvgModelConfig()
    for key, value in (
        ("ifl_families", list(_FAMILIES)),
        ("ifl_dd_tap_fvg_size_min", float(dd_defaults.tap_fvg_size_min)),
        ("ifl_dd_parent_fvg_size_min", float(dd_defaults.parent_fvg_size_min)),
        ("ifl_dd_opp_fvg_size_min", float(dd_defaults.opp_fvg_size_min)),
        ("ifl_dd_parent_distance_max", float(dd_defaults.parent_distance_max)),
        ("ifl_dd_opp_distance_max", float(dd_defaults.opp_distance_max)),
        ("ifl_dd_bars_since_inversion_max", float(dd_defaults.bars_since_inversion_max)),
        ("ifl_cost", float(IfvgScoringConfig().cost_points)),
        ("ifl_tp_value", 1.0),
        ("ifl_sl_value", 5.0),
        ("ifl_m_iterations", md.iterations),
        ("ifl_m_depth", md.depth),
        ("ifl_m_lr", md.learning_rate),
        ("ifl_m_seed", md.seed),
        ("ifl_m_splits", ",".join(str(v) for v in md.split_fracs)),
        ("ifl_m_purge", md.purge_days),
        ("ifl_m_thresholds", ",".join(str(v) for v in md.thresholds)),
    ):
        _seed_default(st, key, value)
    for name in ("asia", "london", "ny"):
        window = charts.ENGINE_SESSION_WINDOWS[name]
        _seed_default(st, f"ifl_cs_{name}_start", window[0])
        _seed_default(st, f"ifl_cs_{name}_end", window[1])

    with st.expander("Row filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        sessions_engine = c1.multiselect(
            "Engine sessions (empty = all)", _SESSION_NAMES, key="ifl_sessions_engine"
        )
        sessions_doc = c2.multiselect(
            "Doc sessions (empty = all)", _SESSION_NAMES, key="ifl_sessions_doc"
        )
        direction = c3.selectbox("Direction", ["both", "long", "short"], key="ifl_direction")
        c4, c5, c6 = st.columns(3)
        families = c4.multiselect(
            "Entry families", list(_FAMILIES), key="ifl_families"
        )
        include_warmup = c5.checkbox("Include warmup days", key="ifl_include_warmup")
        max_tpd = c6.number_input(
            "Max trades/day (0 = unlimited)", 0, 100, key="ifl_max_tpd",
            help="Deterministic first-N by entry time over the logged candidates.",
        )
        c7, c8 = st.columns(2)
        _sanitize_select(st, "ifl_day_start", day_options)
        _sanitize_select(st, "ifl_day_end", day_options)
        day_start = c7.selectbox("Day start", day_options, key="ifl_day_start")
        day_end = c8.selectbox("Day end", day_options, key="ifl_day_end")

    with st.expander("Custom session windows (filter semantics only)"):
        st.caption(
            "Trades are re-stamped offline from entry_ts_utc against these ET windows. "
            "This does NOT rebuild session H/L LEVELS — those are baked into the capture; "
            "to change the levels themselves use the 'Profile re-capture' section below."
        )
        custom_enable = st.checkbox("Enable custom session stamps", key="ifl_custom_enable")
        windows: dict[str, SessionWindow] = {}
        for name in ("asia", "london", "ny"):
            w1, w2 = st.columns(2)
            start = w1.text_input(
                f"{name} start (ET HH:MM)", key=f"ifl_cs_{name}_start"
            )
            end = w2.text_input(
                f"{name} end (ET HH:MM)", key=f"ifl_cs_{name}_end"
            )
            if custom_enable:
                windows[name] = SessionWindow(start=start, end=end)
        sessions_custom = st.multiselect(
            "Filter on custom stamps (empty = no filter)",
            _SESSION_NAMES,
            key="ifl_sessions_custom",
            disabled=not custom_enable,
        )

    with st.expander("Doc-default floors & caps"):
        dd_apply = st.checkbox("Apply doc-default pass", key="ifl_dd_apply")
        d1, d2, d3 = st.columns(3)
        tap_min = d1.number_input(
            "tap FVG size min (ticks)", 0.0, 1000.0, key="ifl_dd_tap_fvg_size_min",
        )
        parent_min = d2.number_input(
            "parent FVG size min", 0.0, 1000.0, key="ifl_dd_parent_fvg_size_min",
        )
        opp_min = d3.number_input(
            "opposing FVG size min", 0.0, 1000.0, key="ifl_dd_opp_fvg_size_min",
        )
        d4, d5, d6 = st.columns(3)
        parent_dist = d4.number_input(
            "parent->HTF distance max", 0.0, 10000.0, key="ifl_dd_parent_distance_max",
        )
        opp_dist = d5.number_input(
            "opposing->parent distance max", 0.0, 10000.0, key="ifl_dd_opp_distance_max",
        )
        bars_inv = d6.number_input(
            "bars-since-inversion max", 0.0, 10000.0, key="ifl_dd_bars_since_inversion_max",
        )

    with st.expander("Scoring & SL/TP overrides"):
        s1, s2 = st.columns(2)
        r_family = s1.selectbox("R family", ["r10", "r15", "r20"], key="ifl_r_family")
        cost = s2.number_input(
            "Cost (points, round turn)", 0.0, 10.0, step=0.05, key="ifl_cost",
        )
        t1, t2 = st.columns(2)
        tp_mode = t1.selectbox(
            "TP override", [_TP_LOGGED, "r_multiple", "fixed_points"], key="ifl_tp_mode"
        )
        tp_value = t2.number_input(
            "TP value (R or points)", 0.01, 1000.0, key="ifl_tp_value",
            disabled=tp_mode == _TP_LOGGED,
        )
        t3, t4 = st.columns(2)
        sl_mode = t3.selectbox(
            "SL override", ["logged", "fixed_points", "swing_buffer_ticks"], key="ifl_sl_mode"
        )
        sl_value = t4.number_input(
            "SL value (points or ticks)", 0.01, 1000.0, key="ifl_sl_value",
            disabled=sl_mode == "logged",
        )
        if tp_mode != _TP_LOGGED or sl_mode != "logged":
            st.caption(
                "Overrides recompute outcomes through the shared SC label kernel over the "
                "cached 1m bars — label-layer only, no path-dependence re-simulation."
            )

    with st.expander("Model block (CatBoost gate)"):
        model_on = st.checkbox(
            "Train the gate (off = expectancy-only, near-instant)", key="ifl_model_on"
        )
        m1, m2, m3, m4 = st.columns(4)
        iterations = m1.number_input(
            "iterations", 1, 5000, key="ifl_m_iterations"
        )
        depth = m2.number_input("depth", 1, 16, key="ifl_m_depth")
        lr = m3.number_input(
            "learning rate", 0.001, 1.0, step=0.01, format="%.3f", key="ifl_m_lr",
        )
        seed = m4.number_input("seed", 0, 10_000, key="ifl_m_seed")
        m5, m6, m7 = st.columns(3)
        splits_text = m5.text_input("split fractions", key="ifl_m_splits")
        purge = m6.number_input("purge days", 0, 30, key="ifl_m_purge")
        thr_text = m7.text_input("p(win) thresholds", key="ifl_m_thresholds")

    try:
        filters = IfvgFilterConfig(
            sessions_engine=tuple(sessions_engine) if sessions_engine else None,
            sessions_doc=tuple(sessions_doc) if sessions_doc else None,
            direction=direction,
            families=tuple(families),
            include_warmup=include_warmup,
            day_start=None if day_start == "(none)" else day_start,
            day_end=None if day_end == "(none)" else day_end,
            max_trades_per_day=int(max_tpd) or None,
        )
        sl_tp = None
        if tp_mode != _TP_LOGGED or sl_mode != "logged":
            sl_tp = IfvgSlTpConfig(
                tp_mode=None if tp_mode == _TP_LOGGED else tp_mode,
                tp_value=None if tp_mode == _TP_LOGGED else float(tp_value),
                sl_mode=sl_mode,
                sl_value=None if sl_mode == "logged" else float(sl_value),
            )
        model = None
        if model_on:
            model = IfvgModelConfig(
                iterations=int(iterations),
                depth=int(depth),
                learning_rate=float(lr),
                seed=int(seed),
                split_fracs=tuple(float(v) for v in splits_text.split(",") if v.strip()),
                purge_days=int(purge),
                thresholds=tuple(float(v) for v in thr_text.split(",") if v.strip()),
            )
        return IfvgExperimentConfig(
            filters=filters,
            custom_sessions=windows if custom_enable else None,
            sessions_custom=(
                tuple(sessions_custom) if custom_enable and sessions_custom else None
            ),
            doc_defaults=IfvgDocDefaultsConfig(
                apply=dd_apply,
                tap_fvg_size_min=float(tap_min),
                parent_fvg_size_min=float(parent_min),
                opp_fvg_size_min=float(opp_min),
                parent_distance_max=float(parent_dist),
                opp_distance_max=float(opp_dist),
                bars_since_inversion_max=float(bars_inv),
            ),
            scoring=IfvgScoringConfig(r_family=r_family, cost_points=float(cost)),
            sl_tp=sl_tp,
            model=model,
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Invalid configuration: {exc}")
        return None


def _run_controls(
    st,
    cfg: IfvgCaptureConfig,
    ctag: str,
    ds: pd.DataFrame,
    config: IfvgExperimentConfig,
    runs: list[dict],
) -> None:
    exp_hash = config.experiment_hash(ctag)
    dup = next((r for r in runs if r["experiment_hash"] == exp_hash), None)
    c1, c2 = st.columns([1, 3])
    c2.caption(f"experiment hash `{exp_hash}`")
    if dup is not None:
        st.warning(
            f"This exact configuration already ran on this capture "
            f"(saved run '{dup.get('name')}')."
        )
        if st.button("Load existing result instead", key="ifl_load_dup"):
            st.session_state["ifl_show_dup"] = exp_hash
    if st.session_state.get("ifl_show_dup") == exp_hash and dup is not None:
        loaded = load_experiment(exp_hash)
        if loaded.get("result") is not None:
            _render_result(st, loaded["result"], key_prefix="dup")
    if c1.button("Run experiment", type="primary", key="ifl_run_btn"):
        with st.spinner("Running IFVG experiment (pre-seal rows only)..."):
            try:
                result = run_ifvg_experiment(config, dataset=ds, capture_cfg=cfg)
            except Exception as exc:  # surfaced, not swallowed
                st.error(f"Experiment failed: {exc}")
                return
        st.session_state["ifl_last_run"] = {"config": config, "result": result}
        st.session_state.pop("ifl_show_dup", None)


def _save_controls(st, last: dict) -> None:
    c1, c2, c3 = st.columns([1, 2, 1])
    name = c1.text_input("Name (optional)", key="ifl_save_name")
    note = c2.text_input("Note (optional)", key="ifl_save_note")
    if c3.button("Save run", key="ifl_save_btn"):
        run_dir = save_experiment(
            last["config"], last["result"], name=name or None, note=note or None
        )
        st.success(f"Saved to {run_dir}")


# ── result rendering (fully in-dashboard) ─────────────────────────────────────


def _render_result(st, result: dict, key_prefix: str = "res") -> None:
    """``key_prefix`` keeps widget keys unique when several results render on
    one page (saved run + last run + duplicate-load)."""
    meta = result.get("meta") or {}
    counts = meta.get("counts") or {}
    ts = result.get("trade_stats") or {}
    st.caption(
        f"hash `{meta.get('experiment_hash', '?')}` · rows in {counts.get('rows_in', '?')} "
        f"-> after filters {counts.get('rows_after_filters', '?')} -> final "
        f"{counts.get('rows_final', '?')} · sealed rows excluded: "
        f"{(meta.get('sealed') or {}).get('sealed_rows_excluded', '?')}"
    )
    if not ts or ts.get("n", 0) == 0:
        st.info("No trades after filters.")
        return

    _headline_tiles(st, ts)
    _trade_stats_tables(st, ts)
    _equity_and_distributions(st, ts, key_prefix)
    model = result.get("model")
    if model:
        _model_section(st, model, key_prefix)
    _expectancy_tables(st, result.get("expectancy") or {})
    _feature_insight_panel(st, result.get("feature_insight") or {}, key_prefix)
    with st.expander("Assumptions & caveats"):
        for line in [*(meta.get("caveats") or []), *(ts.get("assumptions") or [])]:
            st.markdown(f"- {line}")
    _trade_list(st, ts, key_prefix)


def _headline_tiles(st, ts: dict) -> None:
    usd = (ts.get("equity") or {}).get("usd") or {}
    pnl = (ts.get("pnl_usd") or {}).get("all") or {}
    wr = (ts.get("win_rate") or {}).get("all") or {}
    risk = ts.get("risk_adjusted") or {}
    dd = (usd.get("drawdown") or {})
    row1 = st.columns(4)
    row1[0].metric("NET PnL ($)", _fmt(usd.get("net"), "{:,.0f}"))
    row1[1].metric(
        "Win rate", _fmt(wr.get("win_rate"), "{:.1%}"), help=f"95% CI {_ci(wr.get('ci95'))}"
    )
    row1[2].metric("Profit factor", _fmt(pnl.get("profit_factor")))
    row1[3].metric("Expectancy/trade ($)", _fmt(pnl.get("expectancy_per_trade"), "{:,.0f}"))
    row2 = st.columns(4)
    row2[0].metric("Sharpe (daily, ann.)", _fmt(risk.get("sharpe_daily_annualized")))
    row2[1].metric("Sortino (daily, ann.)", _fmt(risk.get("sortino_daily_annualized")))
    row2[2].metric("Max DD ($, close)", _fmt(dd.get("max_drawdown_close"), "{:,.0f}"))
    row2[3].metric("RoMaD", _fmt(usd.get("romad")))


def _sided_row(block: dict, field: str, pattern: str = "{:.2f}") -> dict:
    return {
        side: _fmt((block.get(side) or {}).get(field), pattern)
        for side in ("all", "long", "short")
    }


def _trade_stats_tables(st, ts: dict) -> None:
    st.markdown("**Trade statistics (ALL / LONG / SHORT)**")
    rows: list[dict] = []

    def add(metric: str, values: dict) -> None:
        rows.append({"metric": metric, **values})

    counts, wr = ts.get("counts") or {}, ts.get("win_rate") or {}
    for field in ("trades", "winners", "losers", "eod_timeouts"):
        add(field, _sided_row(counts, field, "{}"))
    add("win rate", _sided_row(wr, "win_rate", "{:.1%}"))
    add(
        "win rate 95% CI",
        {s: _ci((wr.get(s) or {}).get("ci95")) for s in ("all", "long", "short")},
    )
    for unit, key in (("$", "pnl_usd"), ("R", "pnl_r")):
        block = ts.get(key) or {}
        pattern = "{:,.0f}" if unit == "$" else "{:.2f}"
        for field in (
            "net", "gross_profit", "gross_loss", "avg_per_trade", "avg_win", "avg_loss",
            "largest_win", "largest_loss", "profit_factor", "expectancy_per_trade",
            "avg_win_avg_loss_ratio", "payoff_adjusted_expectancy",
        ):
            pat = "{:.2f}" if field in ("profit_factor", "avg_win_avg_loss_ratio") else pattern
            add(f"{field} ({unit})", _sided_row(block, field, pat))
    time_block = ts.get("time") or {}
    for field in ("avg_minutes_in_trade", "avg_minutes_winners", "avg_minutes_losers"):
        add(field, _sided_row(time_block, field, "{:.0f}"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420, hide_index=True)

    eod = ts.get("eod_timeout_breakout") or {}
    extra = {
        "trades/day": _fmt(time_block.get("trades_per_day")),
        "traded days": _fmt(time_block.get("n_traded_days"), "{}"),
        "exposure fraction": _fmt(time_block.get("exposure_fraction"), "{:.3f}"),
        "eod net $ (all)": _fmt((eod.get("all") or {}).get("net_pnl_usd"), "{:,.0f}"),
        "max consec wins": _fmt((ts.get("streaks") or {}).get("max_consecutive_wins"), "{}"),
        "max consec losses": _fmt(
            (ts.get("streaks") or {}).get("max_consecutive_losses"), "{}"
        ),
        "Sharpe/trade R": _fmt((ts.get("risk_adjusted") or {}).get("sharpe_per_trade_r")),
        "Sortino/trade R": _fmt((ts.get("risk_adjusted") or {}).get("sortino_per_trade_r")),
    }
    st.dataframe(
        pd.DataFrame([extra]), use_container_width=True, hide_index=True
    )


def _equity_and_distributions(st, ts: dict, key_prefix: str) -> None:
    unit_label = st.radio(
        "Equity units", ["$", "R"], horizontal=True, key=f"ifl_{key_prefix}_equity_unit"
    )
    unit = "usd" if unit_label == "$" else "r"
    st.plotly_chart(
        charts.build_equity_figure(ts, unit), use_container_width=True,
        key=f"ifl_{key_prefix}_equity_fig",
    )
    dd = ((ts.get("equity") or {}).get(unit) or {}).get("drawdown") or {}
    st.caption(
        f"max DD close {_fmt(dd.get('max_drawdown_close'))} · intrabar "
        f"{_fmt(dd.get('max_drawdown_intrabar'))} · avg depth {_fmt(dd.get('avg_drawdown_depth'))}"
        f" · avg duration {_fmt(dd.get('avg_drawdown_duration_minutes'), '{:.0f}')}m · max "
        f"duration {_fmt(dd.get('max_drawdown_duration_minutes'), '{:.0f}')}m"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Net-R histogram**")
        st.plotly_chart(
            charts.build_r_histogram_figure(ts.get("r_histogram") or {}),
            use_container_width=True, key=f"ifl_{key_prefix}_rhist_fig",
        )
    with c2:
        st.markdown("**MFE / MAE distributions (R)**")
        mm = ts.get("mfe_mae") or {}
        frame = pd.DataFrame(
            [
                {"series": "MFE (R)", **(mm.get("mfe_r") or {})},
                {"series": "MAE (R)", **(mm.get("mae_r") or {})},
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption(
            f"avg MFE of losers {_fmt(mm.get('avg_mfe_r_of_losers'))} R · "
            f"avg MAE of winners {_fmt(mm.get('avg_mae_r_of_winners'))} R"
        )


def _model_section(st, model: dict, key_prefix: str) -> None:
    st.markdown("**Model (CatBoost gate)**")
    if model.get("skipped_reason"):
        st.info(f"Model skipped: {model['skipped_reason']}")
        return
    st.dataframe(
        pd.DataFrame(model.get("splits") or []), use_container_width=True, hide_index=True
    )
    pooled = model.get("pooled") or {}
    dedup = model.get("dedup") or {}
    cols = st.columns(4)
    cols[0].metric("pooled OOS n", _fmt(pooled.get("n"), "{}"))
    cols[1].metric("dedup OOS net R", _fmt(dedup.get("mean_net_r"), "{:.3f}"))
    cols[2].metric("Brier (dedup)", _fmt(dedup.get("brier"), "{:.4f}"))
    cols[3].metric(
        "ref Brier (base rate)", _fmt(model.get("ref_brier"), "{:.4f}"),
        help=f"base rate {_fmt(model.get('base_rate'), '{:.3f}')}",
    )
    c1, c2 = st.columns(2)
    if model.get("calibration"):
        with c1:
            st.markdown("Calibration")
            st.plotly_chart(
                charts.build_calibration_figure(model), use_container_width=True,
                key=f"ifl_{key_prefix}_calib_fig",
            )
    if model.get("coverage"):
        with c2:
            st.markdown("Coverage sweep")
            st.plotly_chart(
                charts.build_coverage_figure(model), use_container_width=True,
                key=f"ifl_{key_prefix}_cov_fig",
            )


def _stat_table(entries: dict) -> pd.DataFrame:
    rows = []
    for key, stat in (entries or {}).items():
        rows.append(
            {
                "group": key,
                "n": stat.get("n"),
                "win rate": _fmt(stat.get("win_rate"), "{:.1%}"),
                "win rate CI95": _ci(stat.get("win_rate_ci95")),
                "mean net R": _fmt(stat.get("mean_net_r"), "{:.3f}"),
                "net R CI95": _ci(stat.get("net_r_ci95")),
            }
        )
    return pd.DataFrame(rows)


def _expectancy_tables(st, expectancy: dict) -> None:
    st.markdown("**Expectancy (per session x family)**")
    st.dataframe(
        _stat_table(expectancy.get("per_session_engine_family")),
        use_container_width=True, hide_index=True,
    )
    with st.expander("More expectancy breakdowns"):
        for key in (
            "overall_wrapper", "per_family", "per_session_engine", "per_session_doc",
            "per_session_custom",
        ):
            block = (
                {"overall": expectancy.get("overall")}
                if key == "overall_wrapper"
                else expectancy.get(key)
            )
            if block:
                st.caption(key.replace("overall_wrapper", "overall"))
                st.dataframe(_stat_table(block), use_container_width=True, hide_index=True)


def _feature_insight_panel(st, fi: dict, key_prefix: str) -> None:
    st.markdown("**Feature insight**")
    numeric = fi.get("numeric") or []
    if numeric:
        order = fi.get("rank_by_net_r_spread") or [e["feature"] for e in numeric]
        by_name = {e["feature"]: e for e in numeric}
        ranked = pd.DataFrame(
            [
                {
                    "feature": name,
                    "net R spread": _fmt(by_name[name].get("net_r_spread"), "{:.3f}"),
                    "spearman(bucket, win)": _fmt(
                        by_name[name].get("spearman_bucket_vs_win"), "{:.3f}"
                    ),
                    "n valid": by_name[name].get("n_valid"),
                }
                for name in order
                if name in by_name
            ]
        )
        st.dataframe(ranked, use_container_width=True, height=280, hide_index=True)
        pick = st.selectbox(
            "Bucket drill-down", [e["feature"] for e in numeric],
            key=f"ifl_{key_prefix}_fi_pick",
        )
        entry = by_name.get(pick)
        if entry:
            st.dataframe(
                pd.DataFrame(entry.get("buckets") or []),
                use_container_width=True, hide_index=True,
            )
    for cat in fi.get("categorical") or []:
        with st.expander(f"categorical: {cat.get('feature')}"):
            st.dataframe(
                pd.DataFrame(cat.get("values") or []),
                use_container_width=True, hide_index=True,
            )
    importance = fi.get("model_importance_per_split") or []
    if importance:
        agg: dict[str, list[float]] = {}
        for split in importance:
            for feat, value in (split.get("importances") or {}).items():
                agg.setdefault(feat, []).append(float(value))
        frame = (
            pd.DataFrame(
                [{"feature": k, "mean gain": sum(v) / len(v)} for k, v in agg.items()]
            )
            .sort_values("mean gain", ascending=False)
            .head(20)
        )
        st.markdown("CatBoost gain (mean across splits, top 20)")
        st.dataframe(frame, use_container_width=True, hide_index=True)
    permutation = fi.get("permutation_importance") or []
    if permutation:
        st.markdown("Permutation importance (OOS delta-Brier, top 20)")
        st.dataframe(
            pd.DataFrame(permutation).drop(columns=["per_split"], errors="ignore").head(20),
            use_container_width=True, hide_index=True,
        )
    if fi.get("caveat"):
        st.caption(fi["caveat"])


def _trade_list(st, ts: dict, key_prefix: str) -> None:
    trades = ts.get("trades") or []
    if not trades:
        return
    st.markdown(f"**Trade list ({len(trades)})**")
    frame = pd.DataFrame(trades)
    st.dataframe(frame, use_container_width=True, height=360)
    st.download_button(
        "Download trades CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="ifvg_trades.csv",
        mime="text/csv",
        key=f"ifl_{key_prefix}_trades_csv",
    )


def _headline_compare(result: dict) -> dict:
    ts = (result or {}).get("trade_stats") or {}
    usd = (ts.get("equity") or {}).get("usd") or {}
    r_curve = (ts.get("equity") or {}).get("r") or {}
    pnl = (ts.get("pnl_usd") or {}).get("all") or {}
    wr = (ts.get("win_rate") or {}).get("all") or {}
    return {
        "n trades": ts.get("n"),
        "net $": usd.get("net"),
        "net R": r_curve.get("net"),
        "win rate": wr.get("win_rate"),
        "profit factor": pnl.get("profit_factor"),
        "max DD $ (close)": (usd.get("drawdown") or {}).get("max_drawdown_close"),
        "RoMaD": usd.get("romad"),
        "sharpe daily ann.": (ts.get("risk_adjusted") or {}).get("sharpe_daily_annualized"),
    }


def _history_and_compare(st, runs: list[dict]) -> None:
    st.subheader("History & compare")
    if not runs:
        st.info("No saved runs yet.")
        return
    hist = pd.DataFrame(
        [
            {
                "hash": r["experiment_hash"],
                "name": r.get("name"),
                "created": r.get("created_utc"),
                "n": (r.get("stats") or {}).get("n_trades"),
                "net R": (r.get("stats") or {}).get("net_r"),
                "PF": (r.get("stats") or {}).get("profit_factor"),
                "sealed looks": len(r.get("sealed_validations") or []),
            }
            for r in runs
        ]
    )
    st.dataframe(hist, use_container_width=True, hide_index=True)
    labels = {_run_label(r): r for r in runs}
    picked = st.multiselect(
        "Compare two runs", list(labels), max_selections=2, key="ifl_cmp_pick"
    )
    if len(picked) != 2:
        return
    run_a, run_b = labels[picked[0]], labels[picked[1]]
    loaded_a = load_experiment(run_a["experiment_hash"])
    loaded_b = load_experiment(run_b["experiment_hash"])
    diff = charts.config_diff_frame(
        loaded_a["config"].model_dump(mode="json"),
        loaded_b["config"].model_dump(mode="json"),
        label_a=run_a["experiment_hash"],
        label_b=run_b["experiment_hash"],
    )
    st.markdown("**Config diff (differing fields highlighted)**")
    st.dataframe(
        diff.style.apply(
            lambda row: (
                ["background-color: rgba(230,159,0,0.25)"] * len(row)
                if row["differs"]
                else [""] * len(row)
            ),
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("**Stat comparison**")
    stats = pd.DataFrame(
        {
            "metric": list(_headline_compare(loaded_a.get("result")).keys()),
            run_a["experiment_hash"]: list(_headline_compare(loaded_a.get("result")).values()),
            run_b["experiment_hash"]: list(_headline_compare(loaded_b.get("result")).values()),
        }
    )
    st.dataframe(stats, use_container_width=True, hide_index=True)


# ── Part C — profile re-capture (managed background job) ──────────────────────


def _render_job_status(st, status: dict) -> None:
    state = status.get("state", "?")
    done = int(status.get("done_days") or 0)
    total = int(status.get("total_days") or 0)
    day = status.get("day")
    if total:
        st.progress(
            min(1.0, done / total),
            text=f"{state}: day {done}/{total}" + (f" ({day})" if day else ""),
        )
    else:
        st.caption(f"state: {state}")
    if state == "failed":
        st.error(f"Job failed: {status.get('error')}")
    elif state == "done":
        st.success("Re-capture complete — the profile is now in the dropdown.")
    st.caption(
        f"started {status.get('started_utc', '?')} · updated {status.get('updated_utc', '?')}"
    )


def _recapture_section(st) -> None:
    with st.expander("Profile re-capture — rebuild session H/L LEVELS under custom windows"):
        st.caption(
            "LEVEL-CONSTRUCTION re-capture: Phase A artifacts + the full capture chain "
            "are rebuilt under the custom ET windows (times only; names, timezone and "
            "trading-day boundary are fixed). All files are written under NEW atag/ctag "
            "names — the canonical capture is never modified. The 'custom session "
            "windows' control in the experiment form only re-stamps trades; THIS job "
            "changes the Asia/London/NY H/L levels the strategy interacts with."
        )
        windows: dict[str, tuple[str, str]] = {}
        for name in ("asia", "london", "ny"):
            default = charts.ENGINE_SESSION_WINDOWS[name]
            _seed_default(st, f"ifl_rc_{name}_start", default[0])
            _seed_default(st, f"ifl_rc_{name}_end", default[1])
            w1, w2 = st.columns(2)
            start = w1.text_input(f"{name} start (ET HH:MM)", key=f"ifl_rc_{name}_start")
            end = w2.text_input(f"{name} end (ET HH:MM)", key=f"ifl_rc_{name}_end")
            windows[name] = (start.strip(), end.strip())

        try:
            preview = recapture.config_for_windows(windows)
        except ValueError as exc:
            st.error(f"Invalid windows: {exc}")
            return
        atag, ctag = preview.artifacts_tag(), preview.capture_tag()
        is_default = ctag == IfvgCaptureConfig().capture_tag()
        st.caption(f"derived atag `{atag}` · ctag `{ctag}`")
        st.caption(
            "Cost: ~45-65 min at 4 workers; Phase A levels rebuild is decode-bound "
            "72-101s/day; capture adds minutes."
        )
        if is_default:
            st.info("These windows match the default profile — nothing to re-capture.")

        c1, c2 = st.columns(2)
        name_label = c1.text_input("Profile name (optional)", key="ifl_rc_name")
        _seed_default(st, "ifl_rc_workers", 4)
        workers = c2.number_input("Workers", 1, 16, key="ifl_rc_workers")
        confirm = st.checkbox(
            "I understand this launches a ~1h background job that rebuilds all "
            "per-day artifacts for these windows",
            key="ifl_rc_confirm",
        )
        if st.button(
            "Launch re-capture job",
            type="primary",
            key="ifl_rc_launch",
            disabled=not confirm or is_default,
        ):
            job_dir = recapture.launch_recapture_job(
                windows, name=name_label.strip() or None, workers=int(workers)
            )
            st.success(f"Job launched -> {job_dir}")

        status = recapture.read_job_status(recapture.job_dir_for(atag, ctag))
        if status is not None:
            st.markdown("**Job status (for the windows above)**")
            _render_job_status(st, status)
        st.button("Refresh status", key="ifl_rc_refresh")


# ══════════════════════════════════════════════════════════════════════════════
#  B2 — Replay / verifier
# ══════════════════════════════════════════════════════════════════════════════


def render_ifvg_replay_tab(st) -> None:
    profile = _profile_selector(st, key="ifl_rep_profile")
    cfg, symbol_dir, atag, ctag = _cfg(profile)
    st.markdown("### IFVG Lab — Replay / verifier")
    runs = list_experiments()
    run = _run_picker(st, runs, key="ifl_rep_run", none_label="(all captured trades)")
    run_keys = None
    loaded = None
    if run is not None:
        loaded = load_experiment(run["experiment_hash"])
        with st.expander("Run configuration", expanded=True):
            _render_config_panel(
                st,
                loaded["config"].model_dump(mode="json"),
                {
                    "experiment_hash": loaded["experiment_hash"],
                    "capture_tag": loaded["capture_tag"],
                    "created_utc": (loaded.get("meta") or {}).get("created_utc"),
                },
            )
        if loaded["capture_tag"] != ctag:
            st.warning(
                f"This run was saved on capture profile `{loaded['capture_tag']}` but the "
                f"selected profile is `{ctag}` — trade scoping and sealed replay are "
                "disabled here; switch to the run's profile to replay it."
            )
            run = None
            loaded = None
        else:
            trades = (
                ((loaded.get("result") or {}).get("trade_stats") or {}).get("trades") or []
            )
            run_keys = charts.run_trade_keys(trades)
            st.caption(f"Chart scoped to this run's {len(run_keys)} admitted trades.")

    days = _cached_replay_days(symbol_dir, atag, ctag, False)
    if not days:
        st.info("No pre-seal replay days with bars+levels+capture artifacts found.")
        return
    ds = _cached_entry_dataset(_dataset_path(cfg, ctag))

    nav = _trade_nav_entries(ds, run_keys, loaded, set(days))
    _trade_navigation(st, nav)
    _sanitize_select(st, "ifl_rep_day", days)
    day = st.selectbox("Day (pre-seal only)", days, index=len(days) - 1, key="ifl_rep_day")
    # The selected PROFILE's windows drive the engine-scheme bands.
    eng_windows = charts.scheme_windows(cfg.session_scheme)
    _render_day_chart(
        st, symbol_dir, atag, ctag, day, run_keys,
        key_prefix="ifl_rep", allow_sealed=False, ds=ds, engine_windows=eng_windows,
    )

    if charts.sealed_replay_available(run):
        _sealed_replay_section(
            st, symbol_dir, atag, ctag, run, loaded, engine_windows=eng_windows
        )


def _trade_nav_entries(
    ds: pd.DataFrame | None,
    run_keys: set | None,
    loaded: dict | None,
    replay_days: set[str],
) -> list[dict]:
    """Global jump-to-trade entries (pre-seal days with artifacts only)."""
    entries: list[dict] = []
    if run_keys is not None and loaded is not None:
        trades = ((loaded.get("result") or {}).get("trade_stats") or {}).get("trades") or []
        for t in trades:
            day = str(t.get("trading_day"))
            if day in replay_days:
                entries.append(
                    {
                        "day": day,
                        "label": f"{day} · {t['setup_id']} · {t['entry_family']}",
                    }
                )
    elif ds is not None:
        pre = ds[ds["trading_day"].astype(str) < SEALED_HOLDOUT_START]
        for _, row in pre.iterrows():
            day = str(row["trading_day"])
            if day not in replay_days:
                continue
            tag = "" if bool(row["selected"]) else " (dropped)"
            entries.append(
                {
                    "day": day,
                    "label": f"{day} · {row['setup_id']} · {row['entry_family']}{tag}",
                }
            )
    seen: set[str] = set()
    unique = []
    for e in sorted(entries, key=lambda e: e["label"]):
        if e["label"] not in seen:
            seen.add(e["label"])
            unique.append(e)
    return unique


def _trade_navigation(st, nav: list[dict]) -> None:
    if not nav:
        return
    labels = [e["label"] for e in nav]
    by_label = {e["label"]: e for e in nav}

    def _apply(label: str) -> None:
        entry = by_label[label]
        st.session_state["ifl_rep_day"] = entry["day"]
        st.session_state["ifl_rep_trade"] = " · ".join(label.split(" · ")[1:])

    def _jump() -> None:
        label = st.session_state.get("ifl_rep_jump")
        if label and label != "(jump to trade)":
            _apply(label)

    def _step(delta: int) -> None:
        current_day = st.session_state.get("ifl_rep_day")
        current_trade = st.session_state.get("ifl_rep_trade")
        current = None
        if current_day and current_trade:
            current = f"{current_day} · {current_trade}".replace(" (dropped)", "")
        idx = 0
        for i, label in enumerate(labels):
            if current and label.replace(" (dropped)", "").startswith(current):
                idx = i
                break
        _apply(labels[max(0, min(len(labels) - 1, idx + delta))])

    c1, c2, c3 = st.columns([1, 1, 6])
    c1.button("prev trade", key="ifl_rep_prev", on_click=_step, args=(-1,))
    c2.button("next trade", key="ifl_rep_next", on_click=_step, args=(1,))
    c3.selectbox(
        "Jump to trade (switches day automatically)",
        ["(jump to trade)", *labels],
        key="ifl_rep_jump",
        on_change=_jump,
    )


def _render_day_chart(
    st,
    symbol_dir: str,
    atag: str,
    ctag: str,
    day: str,
    run_keys: set | None,
    *,
    key_prefix: str,
    allow_sealed: bool,
    ds: pd.DataFrame | None,
    engine_windows: dict[str, tuple[str, str]] | None = None,
) -> None:
    payload = _cached_day_payload(symbol_dir, atag, ctag, day, allow_sealed)
    capture = payload["capture"]
    zones = charts.dedup_zones(capture)
    overlays = charts.scope_overlays(charts.trade_overlays(capture), run_keys)

    # trade/setup picker within the day.
    trade_labels: dict[str, dict] = {}
    for o in overlays:
        tag = "" if o["selected"] else " (dropped)"
        trade_labels[f"{o['setup_id']} · {o['entry_family']}{tag}"] = o
    setups = (
        sorted(
            {
                str(s)
                for s in capture["envelope_setup_id"].dropna().unique()
                if str(s).strip()
            }
        )
        if "envelope_setup_id" in capture.columns
        else []
    )
    covered = {o["setup_id"] for o in overlays}
    # Sealed replay is scoped to the validated run's trades ONLY — never list
    # unscoped setups there (their stage chains would exceed the scope).
    setup_only = (
        [] if allow_sealed else [f"{s} (setup only)" for s in setups if s not in covered]
    )
    options = [*trade_labels, *setup_only] or ["(no setups on this day)"]
    trade_key_name = f"{key_prefix}_trade"
    _sanitize_select(st, trade_key_name, options)
    picked = st.selectbox("Trade / setup", options, key=trade_key_name)
    overlay = trade_labels.get(picked)
    if overlay is not None:
        setup_id = overlay["setup_id"]
    elif picked.endswith(" (setup only)"):
        setup_id = picked[: -len(" (setup only)")]
    else:
        setup_id = None
    markers = charts.stage_markers(capture, setup_id) if setup_id else []

    # layer toggles + timeframe + scrub.
    t1, t2, t3, t4, t5 = st.columns(5)
    tf_label = t1.selectbox(
        "Timeframe", list(charts.TIMEFRAME_OPTIONS), key=f"{key_prefix}_tf"
    )
    tf = charts.TIMEFRAME_OPTIONS[tf_label]
    show_engine = t2.checkbox("engine sessions", value=True, key=f"{key_prefix}_beng")
    show_doc = t3.checkbox("doc sessions", value=False, key=f"{key_prefix}_bdoc")
    show_levels = t4.checkbox("levels", value=True, key=f"{key_prefix}_lvl")
    show_zones = t5.checkbox("FVG zones", value=True, key=f"{key_prefix}_zon")
    t6, t7, t8, t9, t10 = st.columns(5)
    show_recomputed = t6.checkbox(
        "all gaps (recomputed)", value=False, key=f"{key_prefix}_rec",
        help="Re-runs SC's FVG detector over the day's bars — audit overlay, "
        "drawn dotted grey.",
    )
    show_stages = t7.checkbox("stage markers", value=True, key=f"{key_prefix}_stg")
    show_trades = t8.checkbox("trades", value=True, key=f"{key_prefix}_trd")
    show_dropped = t9.checkbox("dropped candidates", value=True, key=f"{key_prefix}_drp")
    show_extra_tps = t10.checkbox("TP 1.5R/2R", value=True, key=f"{key_prefix}_tps")

    n_bars = int((payload["bars"]["timeframe_ticks"] == tf).sum())
    scrub_key = f"{key_prefix}_scrub_{day}_{tf}"
    max_idx = max(n_bars - 1, 0)
    scrub = st.slider(
        "Bar scrub (truncates the chart to bars[0..i]; stages appear in order)",
        0, max_idx, max_idx, key=scrub_key,
    ) if n_bars > 1 else max_idx

    recomputed = (
        _cached_recomputed_gaps(symbol_dir, atag, ctag, day, tf, allow_sealed)
        if show_recomputed
        else None
    )
    chart_col, panel_col = st.columns([4, 1])
    with chart_col:
        fig = charts.build_replay_figure(
            day=day,
            bars=payload["bars"],
            levels=payload["levels"],
            zones=zones,
            overlays=overlays,
            markers=markers,
            timeframe_seconds=tf,
            max_bar_index=scrub,
            show_engine_bands=show_engine,
            show_doc_bands=show_doc,
            show_levels=show_levels,
            show_zones=show_zones,
            show_stages=show_stages,
            show_trades=show_trades,
            show_dropped=show_dropped,
            show_extra_tps=show_extra_tps,
            recomputed_zones=recomputed,
            selected_setup_id=setup_id,
            engine_windows=engine_windows,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fig")
    with panel_col:
        _trade_side_panel(st, ds, day, overlay, markers)


def _trade_side_panel(
    st, ds: pd.DataFrame | None, day: str, overlay: dict | None, markers: list[dict]
) -> None:
    st.markdown("**Trade panel**")
    if overlay is None:
        st.caption("No trade selected (setup-only view).")
    else:
        st.caption(f"{overlay['setup_id']} · {overlay['entry_family']}")
        st.caption(
            f"{overlay['direction']} @ {overlay['entry_price']:.2f} · SL "
            f"{overlay['stop_price']:.2f}"
        )
        st.caption(
            f"risk {overlay['risk_ticks']:.0f}t / {overlay['risk_points']:.2f}pt"
        )
        if not overlay["selected"]:
            st.caption(f"DROPPED: {overlay['drop_reason']}")
        if overlay["resolution"] is not None:
            st.caption(
                f"outcome: {overlay['resolution']}"
                + (
                    f" · {overlay['bars_in_trade']} bars"
                    if overlay["bars_in_trade"] is not None
                    else ""
                )
            )
        row = None
        if ds is not None:
            match = ds[
                (ds["trading_day"].astype(str) == day)
                & (ds["setup_id"] == overlay["setup_id"])
                & (ds["entry_family"] == overlay["entry_family"])
            ]
            row = match.iloc[0] if len(match) else None
        if row is not None:
            st.markdown("labels (all R families)")
            fam_rows = [
                {
                    "family": fam,
                    "label": row.get(f"label_{fam}"),
                    "net R": _fmt(row.get(f"realized_r_net_{fam}"), "{:.2f}"),
                }
                for fam in ("r10", "r15", "r20")
            ]
            st.dataframe(pd.DataFrame(fam_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"sessions: engine {row.get('session_engine')} / doc "
                f"{row.get('session_doc')}"
            )
    if markers:
        st.markdown("stage chain")
        chain = pd.DataFrame(
            [
                {"stage": m["stage"], "ts (UTC)": str(pd.Timestamp(m["ts"]))[:19]}
                for m in markers
            ]
        )
        st.dataframe(chain, hide_index=True, use_container_width=True)


def _sealed_replay_section(
    st, symbol_dir: str, atag: str, ctag: str, run: dict, loaded: dict,
    *, engine_windows: dict[str, tuple[str, str]] | None = None,
) -> None:
    """The ONLY place sealed days can appear: an explicit, labeled expander for
    a saved run with ledgered sealed validations, scoped to that run."""
    seqs = sorted(run.get("sealed_validations") or [])
    title = f"Sealed replay — validated run #{seqs[-1]} ({len(seqs)} ledgered look(s))"
    with st.expander(title, expanded=False):
        st.error(
            "SEALED HOLDOUT DAYS. Replay is unlocked ONLY because this saved run has "
            "sealed validations on the append-only ledger; the chart is scoped to the "
            "trades that run's sealed validation admitted. Every look degrades the "
            "holdout."
        )
        sealed_days = _cached_replay_days(symbol_dir, atag, ctag, True)
        if not sealed_days:
            st.info("No sealed days with replay artifacts.")
            return
        sealed_keys: set = set()
        result_path = Path(loaded["run_dir"]) / f"sealed_result_{seqs[-1]}.json"
        if result_path.exists():
            sealed_result = json.loads(result_path.read_text(encoding="utf-8"))
            sealed_trades = (sealed_result.get("trade_stats") or {}).get("trades") or []
            sealed_keys = charts.run_trade_keys(sealed_trades)
            st.caption(f"scoped to {len(sealed_keys)} sealed-validation trades")
        else:
            st.warning("Sealed result file not found; no trades will be drawn.")
        _sanitize_select(st, "ifl_sealed_day", sealed_days)
        day = st.selectbox("Sealed day", sealed_days, key="ifl_sealed_day")
        _render_day_chart(
            st, symbol_dir, atag, ctag, day, sealed_keys,
            key_prefix="ifl_sealed", allow_sealed=True, ds=None,
            engine_windows=engine_windows,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  entry point (wired from dashboard.py)
# ══════════════════════════════════════════════════════════════════════════════


def render_ifvg_lab_tab() -> None:
    tab_exp, tab_replay = st.tabs(["🧪 Experiments", "🎬 Replay / Verifier"])
    with tab_exp:
        render_ifvg_experiments_tab(st)
    with tab_replay:
        render_ifvg_replay_tab(st)
