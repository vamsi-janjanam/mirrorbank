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

tab_overview, tab_generate, tab_schema, tab_profiler, tab_budget, tab_fidelity = st.tabs([
    "🏠 Overview",
    "✨ Generate",
    "🔍 Instrument Explorer",
    "📊 Data Profiler",
    "🔒 Privacy Budget",
    "📐 Fidelity Evaluation",
])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 0 — Overview
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Welcome to Mirrorbank")
    st.caption("Differentially private synthetic financial data generator")

    st.markdown(
        """
Mirrorbank takes a sensitive transaction dataset (ACH, check, Zelle, wire,
credit card, or debit card) and produces:

- a **synthetic CSV** that looks and behaves like the real data,
- a **privacy certificate** (ε/δ values + membership-inference audit result), and
- a **utility scorecard** (statistical fidelity + fraud-model performance comparison).

The goal: let teams share or model on transaction data **without exposing real
customer records**, while proving — with numbers — that the synthetic data is
both *safe* (low re-identification risk) and *useful* (a fraud model trained on
it performs close to one trained on the real thing).
        """
    )

    st.success(
        "🚀 **Try the demo in 30 seconds — no data needed.** "
        "Open the **📊 Data Profiler** or **📐 Fidelity Evaluation** tab and "
        "click *“Load a built-in example”* to run the tool on bundled sample "
        "transactions for any instrument."
    )

    st.divider()
    st.subheader("Why this matters")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("**🔐 Privacy**")
        st.markdown(
            "Every generation run is trained under a formal "
            "(ε, δ)-differential-privacy guarantee. The privacy budget is "
            "tracked per instrument and never silently exceeded."
        )
    with w2:
        st.markdown("**📊 Utility**")
        st.markdown(
            "Synthetic data is checked against the real data with statistical "
            "tests (KS tests, correlation distance) and downstream model "
            "performance (train-on-synthetic, test-on-real)."
        )
    with w3:
        st.markdown("**🏦 Domain-aware**")
        st.markdown(
            "Each payment instrument has its own schema with realistic "
            "business rules — e.g. wires have zero weekend volume, Zelle "
            "caps transfers at $2,500."
        )

    st.divider()
    st.subheader("How to use this app")
    st.markdown(
        """
Work through the tabs left to right:

1. **🔍 Instrument Explorer** — Start here. Pick a payment instrument
   (ACH, check, Zelle, wire, credit card, debit card) and see exactly which
   columns it expects, which are PII, and how each column gets routed
   (structured fields → DP model, free text → vocab sampler, identifiers →
   regenerated post-hoc).
2. **📊 Data Profiler** — Upload your own CSV (or click *“Load a built-in
   example”*) to see how Mirrorbank would classify each column (continuous /
   categorical / datetime / free text / identifier), detect PII, and compute
   summary statistics. Useful for sanity-checking a dataset *before* generation.
3. **🔒 Privacy Budget** — Choose a privacy preset (or set ε manually),
   enter your dataset size and training plan (batch size, epochs), and
   calibrate the noise multiplier σ needed to hit that ε. Visualize how the
   privacy budget gets consumed step by step.
4. **📐 Fidelity Evaluation** — Compare a real dataset against a synthetic one
   to run the evaluation gauntlet: per-column KS tests, correlation-matrix
   distance, and instrument-specific business-rule checks. Bring your own pair
   of CSVs, or click *“Load a built-in real + synthetic pair”* to try it now.
        """
    )

    st.info(
        "💡 No dataset of your own? Click *“Load a built-in example”* on the "
        "Profiler or Fidelity tab. The same samples live in `data/sample/` "
        "(regenerate with `make sample-data`) and as `tests/conftest.py` fixtures."
    )

    st.divider()
    st.subheader("How to integrate this into your application")

    st.markdown("**Option 1 — Python API**")
    st.code(
        """\
from mirrorbank.instruments.registry import get_schema, detect_instrument
from mirrorbank.profile.schema_profiler import SchemaProfiler
from mirrorbank.privacy.budget import BudgetConfig, calibrate_noise_multiplier
from mirrorbank.evaluate.gauntlet import run_gauntlet
import polars as pl

# 1. Profile your data
df = pl.read_csv("transactions.csv")
schema = get_schema("wire")  # or detect_instrument(df)
profile = SchemaProfiler(schema=schema).fit(df)

# 2. Calibrate a privacy budget for your dataset
sigma = calibrate_noise_multiplier(
    target_epsilon=3.0, delta=1e-5,
    n_rows=profile.n_rows, batch_size=4096, epochs=30,
)

# 3. (once Stage 2 is implemented) generate synthetic data
# synth_df = generate(df, schema=schema, config=BudgetConfig(epsilon=3.0, ...))

# 4. Evaluate fidelity against the real data
report = run_gauntlet(df, synth_df, schema=schema)
report.print_summary()""",
        language="python",
    )

    st.markdown("**Option 2 — CLI** *(scaffolded, not yet implemented)*")
    st.code(
        """\
# Planned usage once src/mirrorbank/cli.py is written:
mirrorbank profile data/transactions.csv --instrument wire
mirrorbank generate --config configs/wire.yaml
mirrorbank evaluate --real data/real.csv --synth outputs/synth.csv --instrument wire""",
        language="bash",
    )

    st.markdown("**Option 3 — Config files**")
    st.markdown(
        "Dataset-specific hyperparameters live in `configs/*.yaml` "
        "(`tabformer.yaml`, `ach.yaml`, `wire.yaml`, `zelle.yaml`) — "
        "instrument, target/exclude columns, generation model + training "
        "params, privacy epsilon/preset, and evaluation targets all live "
        "there so runs are reproducible."
    )

    st.markdown("**Option 4 — This UI**")
    st.markdown(
        "Use the tabs above interactively for exploration, ad-hoc profiling, "
        "and one-off fidelity checks — no code required."
    )

    st.divider()
    st.subheader("Current build status")
    status_rows = [
        {"Stage": "1. Schema profiler", "Status": "✅ done", "Tab": "Instrument Explorer / Data Profiler"},
        {"Stage": "2. Dual-track generation (DP-CTGAN + vocab sampler)", "Status": "🔲 not yet implemented", "Tab": "—"},
        {"Stage": "3. Privacy budget accountant (RDP)", "Status": "✅ done", "Tab": "Privacy Budget"},
        {"Stage": "4a. Fidelity evaluation (KS tests, correlation)", "Status": "✅ done", "Tab": "Fidelity Evaluation"},
        {"Stage": "4b. Utility evaluation (TSTR AUC)", "Status": "🔲 not yet implemented", "Tab": "—"},
        {"Stage": "4c. Privacy audit (shadow-model MIA)", "Status": "🔲 not yet implemented", "Tab": "—"},
        {"Stage": "5. Release bundling (CSV + cert + scorecard)", "Status": "🔲 not yet implemented", "Tab": "—"},
    ]
    st.dataframe(status_rows, use_container_width=True, hide_index=True)
    st.caption(
        "Tabs for not-yet-implemented stages aren't shown yet — this Overview "
        "will be updated as Stage 2 (generation) and the rest of Stage 4 land."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab Generate — Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════

with tab_generate:
    st.header("Generate Synthetic Data")
    st.caption(
        "Runs the Stage 2 pipeline: a DP-statistical structured-column model "
        "(charged against a privacy budget) plus a vocab sampler for free-text "
        "fields (zero privacy budget). Not yet the full DP-CTGAN/DP-TabDDPM "
        "models from the roadmap."
    )

    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        gen_instrument = st.selectbox(
            "Payment instrument",
            INSTRUMENTS,
            format_func=lambda k: INSTRUMENT_LABELS[k],
            key="generate_instrument",
        )

        data_source = st.radio(
            "Data source",
            ["Use built-in sample", "Upload a CSV"],
            horizontal=True,
            key="generate_source",
        )

        gen_upload = None
        if data_source == "Upload a CSV":
            gen_upload = st.file_uploader("Real dataset (CSV)", type=["csv"], key="generate_upload")

    with g_col2:
        gen_preset_key = st.selectbox(
            "Privacy preset",
            list(PRESETS.keys()),
            index=1,
            format_func=lambda k: PRESETS[k]["label"],
            key="generate_preset",
        )
        n_rows = st.number_input(
            "Rows to generate",
            min_value=10,
            max_value=1_000_000_000,
            value=200,
            step=10,
        )
        st.caption(
            "Fit runs once on the real data (charges ε); sampling rows is "
            "privacy-free, so output size is unlimited. Runs above 250k stream "
            "to disk in chunks (~10M rows ≈ 15s, flat memory). For billions, "
            "call `generate_to_csv(...)` from the Python API."
        )

    # ── Resolve the real dataset ────────────────────────────────────────────────
    real_df = None
    if data_source == "Upload a CSV":
        if gen_upload is not None:
            real_df = pl.read_csv(gen_upload)
            st.session_state["generate_real_df"] = real_df
        elif "generate_real_df" in st.session_state:
            real_df = st.session_state["generate_real_df"]
    else:
        from mirrorbank.sample_data import sample_dataset
        real_df = sample_dataset(gen_instrument)

    has_real = real_df is not None
    run_generate = st.button("Generate synthetic data", type="primary", disabled=not has_real)

    if not has_real:
        st.info("Upload a real CSV — or use the built-in sample — to enable generation.")

    if run_generate and has_real:
        try:
            from mirrorbank.instruments.registry import get_schema as _get_schema
            from mirrorbank.privacy.budget import BudgetConfig

            schema_obj = _get_schema(gen_instrument)
            budget_config = BudgetConfig.from_preset(gen_preset_key, dataset_size=real_df.height)
            n = int(n_rows)
            target_epsilon = PRESETS[gen_preset_key]["epsilon"]

            def _show_metrics(eps, delta, rows):
                m1, m2, m3 = st.columns(3)
                m1.metric("ε spent", f"{eps:.4f}", delta=f"target {target_epsilon}")
                m2.metric("δ", f"{delta:.0e}")
                m3.metric("Rows generated", f"{rows:,}")

            _STREAM_THRESHOLD = 250_000

            if n <= _STREAM_THRESHOLD:
                # Small enough to build and serve in memory.
                from mirrorbank.generate.pipeline import generate

                with st.spinner(f"Generating {n:,} synthetic rows…"):
                    result = generate(real_df, schema_obj, n_rows=n, budget_config=budget_config)

                st.success(
                    f"Generated {result.n_rows:,} synthetic rows for "
                    f"**{INSTRUMENT_LABELS[gen_instrument]}**"
                )
                _show_metrics(result.epsilon_spent, result.delta, result.n_rows)
                st.subheader("Preview")
                st.dataframe(result.synthetic.head(50).to_pandas(), use_container_width=True)
                st.download_button(
                    "Download CSV",
                    result.synthetic.write_csv(),
                    file_name=f"{gen_instrument}_synthetic.csv",
                    mime="text/csv",
                )
            else:
                # Large run — stream to disk in bounded-memory chunks.
                import os
                import tempfile

                from mirrorbank.generate.pipeline import generate_to_csv

                tmp = tempfile.NamedTemporaryFile(
                    prefix=f"{gen_instrument}_synth_", suffix=".csv", delete=False
                )
                tmp.close()
                with st.spinner(f"Streaming {n:,} synthetic rows to disk…"):
                    res = generate_to_csv(
                        real_df,
                        schema_obj,
                        n_rows=n,
                        budget_config=budget_config,
                        path=tmp.name,
                        chunk_size=500_000,
                    )

                size_mb = os.path.getsize(tmp.name) / 1e6
                st.success(
                    f"Generated {res.n_rows:,} rows → {size_mb:,.0f} MB on disk "
                    f"(streamed, flat memory)"
                )
                _show_metrics(res.epsilon_spent, res.delta, res.n_rows)
                st.subheader("Preview (first 50 rows)")
                st.dataframe(pl.read_csv(tmp.name, n_rows=50).to_pandas(), use_container_width=True)

                if size_mb <= 500:
                    with open(tmp.name, "rb") as fh:
                        st.download_button(
                            "Download CSV",
                            fh.read(),
                            file_name=f"{gen_instrument}_synthetic.csv",
                            mime="text/csv",
                        )
                else:
                    st.info(
                        "File is too large to stream through the browser. "
                        f"It's saved on disk at:\n\n`{tmp.name}`"
                    )

        except Exception as e:
            st.error(f"Generation failed: {e}")


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

    # ── Try-it shortcut: load a built-in example ───────────────────────────────
    with st.expander("🚀 No data? Load a built-in example", expanded=False):
        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            sample_choice = st.selectbox(
                "Sample instrument",
                INSTRUMENTS,
                format_func=lambda k: INSTRUMENT_LABELS[k],
                key="profiler_sample_instrument",
            )
        with s_col2:
            st.write("")  # vertical spacer to align the button
            if st.button("Load sample data", key="profiler_load_sample", use_container_width=True):
                from mirrorbank.sample_data import sample_dataset
                st.session_state["profiler_df"] = sample_dataset(sample_choice)
                st.session_state["profiler_df_instrument"] = sample_choice

    # ── Resolve the data source: upload takes priority over a loaded sample ─────
    df = None
    schema_name = None
    source = None
    if uploaded:
        df = pl.read_csv(uploaded)
        source = "upload"
        if use_schema:
            schema_name = schema_choice
    elif "profiler_df" in st.session_state:
        df = st.session_state["profiler_df"]
        schema_name = st.session_state.get("profiler_df_instrument")
        source = "sample"

    if df is not None:
        try:
            if source == "sample":
                st.success(
                    f"Loaded sample **{INSTRUMENT_LABELS.get(schema_name, schema_name)}** "
                    f"data — {df.shape[0]:,} rows × {df.shape[1]} columns "
                    f"(instrument schema applied automatically)"
                )
            else:
                st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

            from mirrorbank.instruments.registry import get_schema as _get_schema
            from mirrorbank.profile.schema_profiler import SchemaProfiler

            schema_obj = _get_schema(schema_name) if schema_name else None
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
        st.info("Upload a CSV — or load a built-in example above — to begin profiling.")


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
                from mirrorbank.privacy.budget import (
                    BudgetConfig,
                    PrivacyBudget,
                    calibrate_noise_multiplier,
                )

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

    # ── Try-it shortcut: load a built-in real + synthetic pair ─────────────────
    with st.expander("🚀 No data? Load a built-in real + synthetic pair", expanded=False):
        d_col1, d_col2 = st.columns([2, 1])
        with d_col1:
            demo_choice = st.selectbox(
                "Sample instrument",
                INSTRUMENTS,
                format_func=lambda k: INSTRUMENT_LABELS[k],
                key="fidelity_sample_instrument",
            )
        with d_col2:
            st.write("")  # vertical spacer to align the button
            if st.button("Load sample pair", key="fidelity_load_sample", use_container_width=True):
                from mirrorbank.sample_data import sample_dataset
                st.session_state["fidelity_real_df"] = sample_dataset(demo_choice, seed=42)
                st.session_state["fidelity_synth_df"] = sample_dataset(demo_choice, seed=7)
                st.session_state["fidelity_sample_instrument"] = demo_choice

    # ── Resolve sources: uploads take priority over a loaded sample pair ───────
    real_df = synth_df = None
    schema_name = None
    source = None
    if real_file and synth_file:
        real_df = pl.read_csv(real_file)
        synth_df = pl.read_csv(synth_file)
        source = "upload"
        if use_instrument:
            schema_name = eval_instrument
    elif "fidelity_real_df" in st.session_state and "fidelity_synth_df" in st.session_state:
        real_df = st.session_state["fidelity_real_df"]
        synth_df = st.session_state["fidelity_synth_df"]
        schema_name = st.session_state.get("fidelity_sample_instrument")
        source = "sample"

    has_pair = real_df is not None and synth_df is not None
    run_eval = st.button("Run fidelity evaluation", type="primary", disabled=not has_pair)

    if run_eval and has_pair:
        try:
            if source == "sample":
                st.success(
                    f"Sample **{INSTRUMENT_LABELS.get(schema_name, schema_name)}** — "
                    f"Real: {real_df.shape[0]:,} rows  |  Synthetic: {synth_df.shape[0]:,} rows"
                )
            else:
                st.success(f"Real: {real_df.shape[0]:,} rows  |  Synthetic: {synth_df.shape[0]:,} rows")

            from mirrorbank.evaluate.gauntlet import run_gauntlet
            from mirrorbank.instruments.registry import get_schema as _get_schema

            schema_obj = _get_schema(schema_name) if schema_name else None

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

    elif not has_pair:
        st.info(
            "Upload both a real and a synthetic CSV — or load a built-in pair above "
            "— then click **Run fidelity evaluation**."
        )
