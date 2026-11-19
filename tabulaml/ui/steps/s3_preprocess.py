"""Step 3: data-quality report, Automatic / Custom preprocessing, pipeline preview."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from tabulaml.core.types import ImbalanceStrategy, ModelProfile, ProblemType
from tabulaml.data.profiler import profile_dataset
from tabulaml.data.validation import check_supported, detect_leakage
from tabulaml.evaluation import plots
from tabulaml.preprocessing.builder import describe_pipeline
from tabulaml.preprocessing.cleaning import DataCleaner, clean_target
from tabulaml.preprocessing.recommender import recommend_config
from tabulaml.ui.components import issues_view, nav_buttons
from tabulaml.ui.helpers import OPTIONS, config_to_editor, editor_to_config
from tabulaml.ui.state import AppState, get_state, reset_from, set_state


def _analyse(state: AppState) -> AppState:
    """Profile once per dataset/target and create the Automatic config."""
    if state.config is not None:
        return state
    with st.spinner("Profiling the dataset and recommending a pipeline..."):
        clean = DataCleaner().fit_transform(state.raw_df)
        schema = profile_dataset(clean)
        leakage = tuple(detect_leakage(clean, state.target, state.problem_type))
        supported = tuple(check_supported(clean, schema, state.target))
        config = recommend_config(
            schema, state.target, state.problem_type, leakage, y=state.raw_df[state.target]
        )
    return set_state(
        clean_df=clean, schema=schema, leakage=leakage, supported=supported, config=config
    )


def _quality_tab(state: AppState) -> None:
    schema = state.schema
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{schema.n_rows:,}")
    c2.metric("Columns", len(schema.columns))
    c3.metric("Exact duplicate rows", schema.n_duplicate_rows)
    issues_view(state.supported + state.leakage)
    table = pd.DataFrame(
        [
            {
                "column": c.name,
                "detected type": c.inferred_type.value,
                "missing": f"{c.missing_ratio:.1%}",
                "unique": c.n_unique,
                "skew": None if c.skewness is None else round(c.skewness, 2),
                "outliers": None if c.outlier_ratio is None else f"{c.outlier_ratio:.1%}",
                "examples": ", ".join(c.sample_values[:3]),
                "warnings": ", ".join(c.warnings),
            }
            for c in schema.columns
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    left, right = st.columns(2)
    missing = pd.Series({c.name: c.missing_ratio for c in schema.columns})
    if (missing > 0).any():
        left.plotly_chart(plots.missingness_fig(missing), width="stretch")
    y = clean_target(state.raw_df[state.target], state.problem_type == ProblemType.REGRESSION)
    if state.problem_type == ProblemType.CLASSIFICATION:
        right.plotly_chart(plots.class_distribution_fig(y.dropna()), width="stretch")
    else:
        right.plotly_chart(
            plots.target_histogram_fig(y.dropna().rename(state.target)), width="stretch"
        )


def _global_controls(state: AppState) -> None:
    cfg = state.config
    c1, c2, c3 = st.columns(3)
    indicator = c1.checkbox("Add missing-value indicators", cfg.missing_indicator, key="g_ind")
    case = c1.checkbox(
        "Merge category spelling variants", cfg.normalize_category_case, key="g_case"
    )
    rare = c2.slider(
        "Group categories rarer than",
        0.0,
        0.1,
        cfg.rare_category_min_frequency,
        0.005,
        format="%.3f",
        key="g_rare",
    )
    high = c2.number_input(
        "High-cardinality threshold (target encoding above)",
        2,
        500,
        cfg.high_cardinality_threshold,
        key="g_high",
    )
    changes = {
        "missing_indicator": indicator,
        "normalize_category_case": case,
        "rare_category_min_frequency": rare,
        "high_cardinality_threshold": int(high),
    }
    if state.problem_type == ProblemType.CLASSIFICATION:
        options = [s.value for s in ImbalanceStrategy]
        imb = c3.selectbox(
            "Class imbalance", options, options.index(cfg.imbalance.value), key="g_imb"
        )
        changes["imbalance"] = ImbalanceStrategy(imb)
    else:
        tt = c3.selectbox(
            "Target transform",
            ["none", "log1p"],
            ["none", "log1p"].index(cfg.target_transform),
            key="g_tt",
        )
        changes["target_transform"] = tt
    new = replace(cfg, **changes)
    if new != cfg:
        reset_from(4, config=replace(new, mode="custom"))


def _preprocessing_tab(state: AppState) -> None:
    mode = st.radio(
        "Mode",
        ["Automatic", "Custom"],
        horizontal=True,
        index=0 if state.config.mode == "auto" else 1,
        key="pp_mode",
    )
    if state.card is not None and state.card.recommended_exclusions:
        st.caption(
            "Dataset card recommends excluding: "
            + ", ".join(f"`{c}`" for c in state.card.recommended_exclusions)
        )
    st.markdown("\n".join(f"- {line}" for line in state.config.explanations))
    editor = config_to_editor(state.config)
    if mode == "Automatic":
        st.dataframe(editor, hide_index=True, width="stretch")
        return
    _global_controls(state)
    column_config = {
        k: st.column_config.SelectboxColumn(k, options=v, required=True) for k, v in OPTIONS.items()
    }
    column_config["include"] = st.column_config.CheckboxColumn("include")
    edited = st.data_editor(
        editor,
        hide_index=True,
        width="stretch",
        key="pp_editor",
        disabled=["column", "reason"],
        column_config=column_config,
    )
    new = editor_to_config(get_state().config, edited)
    if new != get_state().config:
        reset_from(4, config=new)


def _preview_tab(state: AppState) -> None:
    st.write(
        "Each model family gets its own pipeline. Every step is fitted on training folds only."
    )
    left, right = st.columns(2)
    left.markdown("**Tree models** (RF, XGBoost, LightGBM, CatBoost, ...)")
    left.dataframe(
        describe_pipeline(state.config, ModelProfile.TREE), hide_index=True, width="stretch"
    )
    right.markdown("**Scale-sensitive models** (linear, SVM, KNN, ANN)")
    right.dataframe(
        describe_pipeline(state.config, ModelProfile.SCALE_SENSITIVE),
        hide_index=True,
        width="stretch",
    )


def _run_settings(state: AppState) -> None:
    cfg = state.config
    with st.expander("Evaluation settings"):
        c1, c2, c3 = st.columns(3)
        test = c1.slider("Test set share", 0.1, 0.4, cfg.test_size, 0.05, key="rs_test")
        folds = c2.slider("Cross-validation folds", 2, 10, cfg.cv_folds, key="rs_folds")
        seed = c3.number_input("Random seed", 0, 10_000, cfg.random_state, key="rs_seed")
    new = replace(cfg, test_size=test, cv_folds=folds, random_state=int(seed))
    if new != cfg:
        reset_from(4, config=new)


def render(state: AppState) -> None:
    st.header("3. Data quality and preprocessing")
    state = _analyse(state)
    quality, preprocessing, preview = st.tabs(
        ["Data quality report", "Preprocessing", "Pipeline preview"]
    )
    with quality:
        _quality_tab(state)
    with preprocessing:
        _preprocessing_tab(state)
    _run_settings(get_state())
    with preview:
        _preview_tab(get_state())
    blocked = any(i.level == "error" for i in state.supported)
    has_features = any(c.include for c in get_state().config.columns)
    if not has_features:
        st.error("Include at least one feature column.")
    nav_buttons(back=2, next_step=4, next_enabled=not blocked and has_features)
