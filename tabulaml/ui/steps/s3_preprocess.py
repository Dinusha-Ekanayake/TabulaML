"""Step 3: data-quality report, Automatic / Custom preprocessing, pipeline preview."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from tabulaml.core.types import ImbalanceStrategy, ModelProfile, ProblemType
from tabulaml.evaluation import plots
from tabulaml.preprocessing.builder import describe_pipeline
from tabulaml.preprocessing.cleaning import clean_target
from tabulaml.training.splits import describe
from tabulaml.ui.components import issues_view, nav_buttons
from tabulaml.ui.helpers import OPTIONS, config_to_editor, editor_to_config
from tabulaml.ui.state import AppState, get_state, reset_from, set_state
from tabulaml.ui.steps.s3_explore import explore_tab
from tabulaml.ui.steps.s3_features import features_card
from tabulaml.ui.steps.s3_missing import missing_values_section
from tabulaml.ui.steps.s3_rows import row_settings, rows_card
from tabulaml.workflow import Roles, auto_configure

IMBALANCE_HELP = {
    ImbalanceStrategy.NONE: ("None", "Train on the data as it is."),
    ImbalanceStrategy.CLASS_WEIGHT: (
        "Class weights",
        "Mistakes on the rare class count more. No new rows.",
    ),
    ImbalanceStrategy.RANDOM_OVERSAMPLE: (
        "Random oversampling",
        "Copies rare-class rows in the training folds.",
    ),
    ImbalanceStrategy.SMOTE: (
        "SMOTE",
        "Creates new rare-class rows between neighbours; "
        "category columns keep real values (SMOTE-NC).",
    ),
    ImbalanceStrategy.UNDERSAMPLE: (
        "Random undersampling",
        "Drops common-class rows in the training folds.",
    ),
}


def _settings() -> tuple[float, int, int, int | None]:
    """Evaluation settings live in widget state so changing them re-runs the analysis."""
    cap = int(st.session_state.get("rs_cap", 0) or 0)
    return (
        float(st.session_state.get("rs_test", 0.2)),
        int(st.session_state.get("rs_folds", 5)),
        int(st.session_state.get("rs_seed", 42)),
        cap or None,
    )


def _analyse(state: AppState) -> AppState:
    """Holdout first, then recommendations from the training rows only (workflow)."""
    if state.config is not None and state.auto is not None:
        return state
    test, folds, seed, cap = _settings()
    drop_dups, missing_rows = row_settings()
    roles = Roles(
        target=state.target,
        problem_type=state.problem_type,
        validation=state.validation,
        group_column=state.group_column,
        order_column=state.order_column,
        test_size=test,
        cv_folds=folds,
        random_state=seed,
        decimal=state.decimal,
        positive_class=state.positive_class,
        max_rows=cap,
        extra_targets=state.extra_targets,
        drop_duplicates=drop_dups,
        missing_rows=missing_rows,
        forecast=state.forecast if state.forecasting else None,
    )
    with st.spinner("Holding out the test rows, profiling and recommending a pipeline..."):
        auto = auto_configure(state.raw_df, roles)
    return set_state(
        auto=auto,
        schema=auto.schema_all,
        leakage=auto.leakage,
        supported=auto.supported,
        config=auto.config,
        leakage_decisions={},
    )


def _quality_tab(state: AppState) -> None:
    schema = state.schema
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{schema.n_rows:,}")
    c2.metric("Columns", len(schema.columns))
    c3.metric("Exact duplicate rows", schema.n_duplicate_rows)
    auto = state.auto
    st.info(
        f"{describe(state.config)} Training (dev) rows: {auto.n_dev:,}; "
        f"held-out test rows: {auto.n_test:,}. Recommendations use the training rows only.",
        icon=":material/verified:",
    )
    issues_view(state.supported + auto.split_issues + auto.format_issues)
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
    if state.problem_type == ProblemType.CLASSIFICATION and cfg.is_multi_output:
        c3.caption(
            "Class imbalance: class weights and resampling need a single target, so they "
            "are off when you predict several targets."
        )
    elif state.problem_type == ProblemType.CLASSIFICATION:
        options = list(ImbalanceStrategy)
        imb = c3.selectbox(
            "Class imbalance",
            options,
            options.index(cfg.imbalance),
            format_func=lambda s: IMBALANCE_HELP[s][0],
            key="g_imb",
        )
        c3.caption(IMBALANCE_HELP[imb][1])
        changes["imbalance"] = imb
    else:
        low = min(clean_target(state.raw_df[t], regression=True).min() for t in cfg.targets)
        allowed = ["none", "yeo_johnson"] + (["log1p"] if low > -1 else [])
        current = cfg.target_transform if cfg.target_transform in allowed else "none"
        tt = c3.selectbox(
            "Target transform",
            allowed,
            allowed.index(current),
            key="g_tt",
            help="log1p suits positive, right-skewed targets (counts, prices); "
            "Yeo-Johnson also works with negative values.",
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
    features_card(get_state())
    if mode == "Automatic":
        st.dataframe(config_to_editor(state.config), hide_index=True, width="stretch")
        st.caption(
            "To choose the imputation method, encoding, scaling and more per column, "
            "switch the mode to **Custom**."
        )
        return
    _global_controls(state)
    rows_card(get_state())
    missing_values_section(get_state())
    st.markdown("##### Per-column settings")
    column_config = {
        k: st.column_config.SelectboxColumn(k, options=v, required=True) for k, v in OPTIONS.items()
    }
    column_config["include"] = st.column_config.CheckboxColumn("include")
    column_config["impute"] = st.column_config.TextColumn(
        "impute", help="Set in Missing values (imputation) above."
    )
    edited = st.data_editor(
        config_to_editor(get_state().config),
        hide_index=True,
        width="stretch",
        key="pp_editor",
        disabled=["column", "reason", "impute"],
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


def _run_settings(n_rows: int) -> None:
    """Changing these re-runs the analysis (a new holdout means new recommendations)."""
    with st.expander("Evaluation settings (test share, folds, seed, row cap)"):
        c1, c2, c3, c4 = st.columns(4)
        c1.slider("Test set share", 0.1, 0.4, 0.2, 0.05, key="rs_test", on_change=_restart_analysis)
        c2.slider("Cross-validation folds", 2, 10, 5, key="rs_folds", on_change=_restart_analysis)
        c3.number_input("Random seed", 0, 10_000, 42, key="rs_seed", on_change=_restart_analysis)
        c4.number_input(
            "Row cap (0 = all rows)",
            0,
            max(n_rows, 1),
            0,
            step=1000,
            key="rs_cap",
            on_change=_restart_analysis,
            help="Train on at most this many rows to save time. Time-ordered data keeps the "
            "latest rows; grouped data keeps whole groups; otherwise a random sample.",
        )
        st.caption(
            "Changing these re-creates the held-out test set and re-runs the "
            "recommendations; custom edits are reset."
        )


def _restart_analysis() -> None:
    reset_from(3)


def _leakage_tab(state: AppState) -> bool:
    """P16: the user decides every flagged column. Returns True when all are answered."""
    flagged = [i for i in state.auto.leakage if i.column]
    if not flagged:
        st.success("No column looks like target leakage.", icon=":material/check_circle:")
        return True
    st.write(
        "These columns are almost perfectly related to the target. That can be genuine, or the "
        "value may only be known AFTER the outcome (leakage). They are excluded until you "
        "decide."
    )
    decisions = dict(state.leakage_decisions)
    for issue in flagged:
        with st.container(border=True):
            st.markdown(f"**{issue.column}**")
            st.caption(issue.message)
            answer = st.radio(
                "Will this value be known at the moment you make a prediction?",
                ["No: keep it excluded", "Yes: use it as a feature"],
                index=None if issue.column not in decisions else int(decisions[issue.column]),
                key=f"leak_{issue.column}",
                horizontal=True,
            )
            if answer is not None:
                decisions[issue.column] = answer.startswith("Yes")
    if decisions != state.leakage_decisions:
        _apply_decisions(state, decisions)
    return all(i.column in decisions for i in flagged)


def _apply_decisions(state: AppState, decisions: dict[str, bool]) -> None:
    config = state.config
    for column, keep in decisions.items():
        current = next((c for c in config.columns if c.name == column), None)
        if current is None:
            continue
        reason = (
            "Included: you confirmed it is known at prediction time."
            if keep
            else "Excluded: you confirmed it is not known at prediction time."
        )
        config = config.with_column(replace(current, include=keep, reason=reason))
    explanations = tuple(e for e in config.explanations if not e.startswith("Leakage review:"))
    summary = "Leakage review: " + "; ".join(
        f"'{c}' {'kept' if k else 'excluded'}" for c, k in decisions.items()
    )
    reset_from(
        4,
        config=replace(config, explanations=(*explanations, summary)),
        leakage_decisions=decisions,
    )


def _keep_tab() -> None:
    """Tab selection is tracked in session state, so re-runs keep the open tab."""


def render(state: AppState) -> None:
    st.header("3. Data quality and preprocessing")
    st.caption(
        "Check the data, review possible leakage, and see exactly how each model "
        "family will prepare it."
    )
    _run_settings(len(state.raw_df))
    state = _analyse(get_state())
    n_flagged = len([i for i in state.auto.leakage if i.column])
    quality, explore, leakage, preprocessing, preview = st.tabs(
        key="step3_tabs",
        on_change=_keep_tab,
        tabs=[
            "Data quality report",
            "Explore data",
            f"Leakage review ({n_flagged})" if n_flagged else "Leakage review",
            "Preprocessing",
            "Pipeline preview",
        ],
    )
    with quality:
        _quality_tab(state)
    with explore:
        explore_tab(state)
    with leakage:
        answered = _leakage_tab(state)
    with preprocessing:
        _preprocessing_tab(get_state())
    with preview:
        _preview_tab(get_state())
    blocked = any(i.level == "error" for i in state.supported + state.auto.split_issues)
    has_features = any(c.include for c in get_state().config.columns)
    if not has_features:
        st.error("Include at least one feature column.")
    if not answered:
        st.warning("Answer the leakage review before choosing models.", icon=":material/report:")
    nav_buttons(back=2, next_step=4, next_enabled=not blocked and has_features and answered)
