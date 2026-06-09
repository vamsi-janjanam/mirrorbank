"""Mirrorbank — Streamlit control panel."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import streamlit as st

# Ensure `src/` is importable when run directly via `streamlit run` (not just pytest).
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

st.set_page_config(
    page_title="Mirrorbank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🏦 Mirrorbank")
st.sidebar.caption("Differentially private synthetic financial data")
st.sidebar.divider()

INSTRUMENTS = ["ach", "check", "credit_card", "debit_card", "wire", "zelle"]
INSTRUMENT_LABELS = {
    "ach": "ACH",
    "check": "Check",
    "credit_card": "Credit Card",
    "debit_card": "Debit Card",
    "wire": "Wire",
    "zelle": "Zelle",
}
PRESETS = {
    "tight": {"epsilon": 1.0, "noise_multiplier": 1.5, "label": "Tight (ε=1) — strongest privacy"},
    "balanced": {"epsilon": 3.0, "noise_multiplier": 1.1, "label": "Balanced (ε=3) — recommended"},
    "loose": {"epsilon": 5.0, "noise_multiplier": 0.9, "label": "Loose (ε=5) — better utility"},
    "demo": {"epsilon": 10.0, "noise_multiplier": 0.7, "label": "Demo (ε=10) — explore only"},
}

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_schema, tab_profiler, tab_budget, tab_fidelity = st.tabs([
    "🔍 Instrument Explorer",
    "📊 Data Profiler",
    "🔒 Privacy Budget",
    "📐 Fidelity Evaluation",
])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Instrument Explorer
# ═══════════════════════════════════════════════════════════════════════════════

with tab_schema:
    st.header("Instrument Explorer")
    st.caption("Browse schema definitions for each payment instrument.")

    selected = st.selectbox(
        "Payment instrument",
        INSTRUMENTS,
        format_func=lambda k: INSTRUMENT_LABELS[k],
        key="explorer_instrument",
    )

    try:
        from mirrorbank.instruments.registry import get_schema
        schema = get_schema(selected)

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader(f"{schema.display_name} schema")
            if schema.fraud_label:
                st.info(f"Fraud label column: **{schema.fraud_label}**")

            rows = []
            for col in schema.columns:
                rows.append({
                    "Column": col.name,
                    "Kind": col.kind,
                    "PII": "⚠️ yes" if col.is_pii else "no",
                    "Nullable": "yes" if col.nullable else "no",
                    "Description": col.description or "—",
                })
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Kind": st.column_config.TextColumn(width="small"),
                    "PII": st.column_config.TextColumn(width="small"),
                    "Nullable": st.column_config.TextColumn(width="small"),
                },
            )

        with col_right:
            st.subheader("Column routing")
            train_cols = schema.training_columns()
            text_cols = schema.text_columns()
            ref_cols = [c.name for c in schema.reference_columns()]
            id_cols = schema.identifier_columns()
            pii_cols = schema.pii_columns()

            st.metric("Total columns", len(schema.columns))
            st.metric("Track A (DP generator)", len(train_cols))
            st.metric("Track B (vocab sampler)", len(text_cols))
            st.metric("Reference (post-hoc)", len(ref_cols))
            st.metric("Identifiers (excluded)", len(id_cols))
            st.metric("PII (excluded)", len(pii_cols))

            if train_cols:
                with st.expander("Track A columns"):
                    st.write(train_cols)
            if text_cols:
                with st.expander("Track B columns"):
                    st.write(text_cols)
            if pii_cols:
                with st.expander("PII columns"):
                    st.write(pii_cols)

    except Exception as e:
        st.error(f"Could not load schema: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Data Profiler
# ═══════════════════════════════════════════════════════════════════════════════

with tab_profiler:
    st.header("Data Profiler")
    st.caption("Upload a CSV and inspect inferred column kinds, PII flags, and statistics.")

    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"], key="profiler_upload")
    with p_col2:
        use_schema = st.checkbox("Apply instrument schema", value=False)
        if use_schema:
            schema_choice = st.selectbox(
                "Instrument",
                INSTRUMENTS,
                format_func=lambda k: INSTRUMENT_LABELS[k],
                key="profiler_instrument",
            )

    if uploaded:
        try:
            df = pl.read_csv(uploaded)
            st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

            from mirrorbank.profile.schema_profiler import SchemaProfiler
            from mirrorbank.instruments.registry import get_schema as _get_schema

            schema_obj = _get_schema(schema_choice) if use_schema else None
            profiler = SchemaProfiler(schema=schema_obj)

            with st.spinner("Profiling…"):
                profile = profiler.fit(df)

            st.subheader("Dataset overview")
            ov1, ov2, ov3, ov4 = st.columns(4)
            ov1.metric("Rows", f"{profile.n_rows:,}")
            ov2.metric("Columns", profile.n_cols)
            ov3.metric("Training cols (Track A)", len(profile.training_columns()))
            ov4.metric("Text cols (Track B)", len(profile.text_columns()))

            st.subheader("Column profiles")
            profile_rows = []
            for cp in profile.columns:
                row = {
                    "Column": cp.name,
                    "Kind": cp.kind,
                    "Dtype": cp.dtype,
                    "PII": "⚠️ yes" if cp.is_pii else "no",
                    "Unique": f"{cp.n_unique:,}",
                    "Null %": f"{cp.null_frac:.1%}",
                }
                # Append key stats inline
                for stat_key in ("mean", "std", "min", "max", "median"):
                    if stat_key in cp.stats:
                        v = cp.stats[stat_key]
                        row[stat_key] = f"{v:.4g}" if isinstance(v, float) else str(v)
                    else:
                        row[stat_key] = "—"
                profile_rows.append(row)

            st.dataframe(profile_rows, use_container_width=True, hide_index=True)

            st.subheader("Kind breakdown")
            kind_counts: dict[str, int] = {}
            for cp in profile.columns:
                kind_counts[cp.kind] = kind_counts.get(cp.kind, 0) + 1
            st.bar_chart(kind_counts)

        except Exception as e:
            st.error(f"Profiling failed: {e}")
    else:
        st.info("Upload a CSV to begin profiling.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Privacy Budget
# ═══════════════════════════════════════════════════════════════════════════════

with tab_budget:
    st.header("Privacy Budget")
    st.caption("Configure DP-SGD privacy parameters and calibrate the noise multiplier.")

    b_left, b_right = st.columns([1, 1])

    with b_left:
        st.subheader("Configuration")

        mode = st.radio("Mode", ["Preset", "Manual"], horizontal=True)

        if mode == "Preset":
            preset_key = st.selectbox(
                "Preset",
                list(PRESETS.keys()),
                index=1,
                format_func=lambda k: PRESETS[k]["label"],
            )
            target_epsilon = PRESETS[preset_key]["epsilon"]
            default_sigma = PRESETS[preset_key]["noise_multiplier"]
            st.info(f"ε = {target_epsilon}  |  σ (default) = {default_sigma}")
        else:
            target_epsilon = st.slider("Target ε (epsilon)", 0.5, 20.0, 3.0, 0.5)
            default_sigma = None

        delta = st.selectbox("δ (delta)", [1e-5, 1e-6, 1e-4], index=0, format_func=lambda x: f"{x:.0e}")

        st.divider()
        st.subheader("Training parameters")
        dataset_size = st.number_input("Dataset size (rows)", min_value=1000, max_value=50_000_000, value=100_000, step=1000)
        batch_size = st.number_input("Batch size", min_value=32, max_value=65536, value=4096, step=32)
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=30, step=1)
        max_grad_norm = st.number_input("Max gradient norm (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

        calibrate = st.button("Calibrate noise multiplier", type="primary")

    with b_right:
        st.subheader("Results")

        if calibrate:
            try:
                from mirrorbank.privacy.budget import calibrate_noise_multiplier, BudgetConfig, PrivacyBudget

                with st.spinner("Binary-searching for optimal σ…"):
                    sigma = calibrate_noise_multiplier(
                        target_epsilon=target_epsilon,
                        delta=delta,
                        n_rows=int(dataset_size),
                        batch_size=int(batch_size),
                        epochs=int(epochs),
                    )

                st.success(f"Noise multiplier σ = **{sigma:.4f}**")

                config = BudgetConfig(
                    epsilon=target_epsilon,
                    delta=delta,
                    noise_multiplier=sigma,
                    max_grad_norm=max_grad_norm,
                    batch_size=int(batch_size),
                    dataset_size=int(dataset_size),
                )
                budget = PrivacyBudget(config)

                n_steps = max(1, int(dataset_size / batch_size * epochs))
                st.metric("Training steps", f"{n_steps:,}")
                st.metric("Sampling rate q", f"{batch_size/dataset_size:.4%}")
                st.metric("Noise multiplier σ", f"{sigma:.4f}")
                st.metric("Max gradient norm C", max_grad_norm)

                st.divider()
                st.subheader("Budget simulation")
                st.caption("ε consumed vs training steps (sample every 1% of steps)")

                sample_every = max(1, n_steps // 100)
                eps_trace = []
                step_trace = []

                try:
                    for step in range(1, n_steps + 1):
                        budget.charge_step()
                        if step % sample_every == 0:
                            eps_trace.append(budget.current_epsilon())
                            step_trace.append(step)
                except Exception:
                    # BudgetExhausted — chart up to where it stopped
                    pass

                if eps_trace:
                    chart_data = {"step": step_trace, "ε spent": eps_trace}
                    st.line_chart({"ε spent": eps_trace}, x_label="Step (sampled)", y_label="ε")
                    final_eps = eps_trace[-1]
                    st.metric("Final ε", f"{final_eps:.4f}", delta=f"target {target_epsilon}")

            except Exception as e:
                st.error(f"Calibration failed: {e}")
        else:
            st.info("Set parameters on the left and click **Calibrate** to run.")

            st.subheader("Preset reference")
            preset_rows = [
                {"Preset": k, "ε": v["epsilon"], "σ (default)": v["noise_multiplier"], "Use case": v["label"].split("—")[1].strip()}
                for k, v in PRESETS.items()
            ]
            st.dataframe(preset_rows, use_container_width=True, hide_index=True)
            st.caption("δ = 1e-5 for all presets. Calibration overwrites σ to exactly hit ε.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Fidelity Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

with tab_fidelity:
    st.header("Fidelity Evaluation")
    st.caption("Upload real and synthetic CSVs to run the KS-test battery and correlation check.")

    f_col1, f_col2, f_col3 = st.columns([2, 2, 1])

    with f_col1:
        real_file = st.file_uploader("Real dataset (CSV)", type=["csv"], key="fidelity_real")
    with f_col2:
        synth_file = st.file_uploader("Synthetic dataset (CSV)", type=["csv"], key="fidelity_synth")
    with f_col3:
        use_instrument = st.checkbox("Apply schema", value=True)
        if use_instrument:
            eval_instrument = st.selectbox(
                "Instrument",
                INSTRUMENTS,
                format_func=lambda k: INSTRUMENT_LABELS[k],
                key="fidelity_instrument",
            )
        ks_threshold = st.number_input("KS p-threshold", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

    run_eval = st.button("Run fidelity evaluation", type="primary", disabled=not (real_file and synth_file))

    if run_eval and real_file and synth_file:
        try:
            real_df = pl.read_csv(real_file)
            synth_df = pl.read_csv(synth_file)

            st.success(f"Real: {real_df.shape[0]:,} rows  |  Synthetic: {synth_df.shape[0]:,} rows")

            from mirrorbank.evaluate.gauntlet import run_gauntlet
            from mirrorbank.instruments.registry import get_schema as _get_schema

            schema_obj = _get_schema(eval_instrument) if use_instrument else None

            with st.spinner("Running gauntlet…"):
                report = run_gauntlet(real_df, synth_df, schema=schema_obj)

            # ── Overall verdict ────────────────────────────────────────────────
            if report.pass_overall:
                st.success("✅ Gauntlet PASSED")
            else:
                st.error("❌ Gauntlet FAILED")

            # ── Summary metrics ────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            fid = report.fidelity
            m1.metric("KS pass rate", f"{fid.ks_pass_rate:.1%}", delta="target ≥ 80%")
            m2.metric("Correlation distance", f"{fid.corr_distance:.4f}", delta="target < 0.15")
            m3.metric("Instrument errors", len(fid.instrument_errors))
            m4.metric("Columns tested", len(fid.ks_results))

            # ── KS results table ───────────────────────────────────────────────
            st.subheader("Per-column KS tests")
            ks_rows = []
            for col, result in sorted(fid.ks_results.items()):
                ks_rows.append({
                    "Column": col,
                    "KS statistic": f"{result['statistic']:.4f}",
                    "p-value": f"{result['p_value']:.4f}",
                    "Pass": "✅" if result["pass"] else "❌",
                })
            st.dataframe(ks_rows, use_container_width=True, hide_index=True)

            # ── Instrument errors ──────────────────────────────────────────────
            if fid.instrument_errors:
                st.subheader("Instrument business-rule violations")
                for err in fid.instrument_errors:
                    st.warning(err)
            else:
                st.subheader("Instrument business rules")
                st.success("No violations detected.")

            # ── KS bar chart ───────────────────────────────────────────────────
            st.subheader("KS statistic by column")
            ks_chart = {row["Column"]: float(row["KS statistic"]) for row in ks_rows}
            st.bar_chart(ks_chart, x_label="Column", y_label="KS statistic")

        except Exception as e:
            st.error(f"Evaluation failed: {e}")

    elif not real_file or not synth_file:
        st.info("Upload both a real and a synthetic CSV, then click **Run fidelity evaluation**.")
