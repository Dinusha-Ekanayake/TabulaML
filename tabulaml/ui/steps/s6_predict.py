"""Step 6: use a trained model on one record or a batch file; download the bundle."""

from __future__ import annotations

from typing import Any

import streamlit as st

from tabulaml.core.types import ProblemType
from tabulaml.data.loader import DataLoadError, read_tabular_file
from tabulaml.evaluation import plots
from tabulaml.inference.persistence import bundle_bytes
from tabulaml.inference.predictor import Predictor, make_input_spec
from tabulaml.ui.components import issues_view, nav_buttons
from tabulaml.ui.state import AppState, set_state


def _input_spec(state: AppState) -> AppState:
    if state.input_spec:
        return state
    spec = make_input_spec(state.raw_df[list(state.run.feature_columns)], state.config)
    return set_state(input_spec=tuple(spec))


def _field(spec: dict[str, Any], col) -> Any:  # noqa: ANN001
    name, kind, default = spec["name"], spec["type"], spec["default"]
    key = f"in_{name}"
    if kind == "numeric":
        return col.number_input(
            name,
            value=default,
            key=key,
            help=f"range in data: {spec.get('min')} to {spec.get('max')}",
        )
    if kind == "boolean":
        return col.checkbox(name, value=bool(default), key=key)
    if kind == "datetime":
        return col.date_input(name, value=default, key=key)
    options = spec["options"] or []
    if options:
        value = col.selectbox(name, options + ["(other value)"], key=key)
        if value == "(other value)":
            return col.text_input(f"{name} (type a value)", key=f"{key}_other")
        return value
    return col.text_input(name, value=default or "", key=key)


def _single(predictor: Predictor, state: AppState) -> None:
    with st.form("single_record"):
        cols = st.columns(3)
        record = {s["name"]: _field(s, cols[i % 3]) for i, s in enumerate(state.input_spec)}
        submitted = st.form_submit_button("Predict", type="primary")
    if not submitted:
        return
    out = predictor.predict_one(record)
    if state.problem_type == ProblemType.CLASSIFICATION:
        st.success(f"Prediction: **{out['prediction']}**")
        if out["probabilities"]:
            st.plotly_chart(plots.probability_fig(out["probabilities"]), width="stretch")
    else:
        st.success(f"Prediction: **{out['prediction']:,.3f}**")


def _batch(predictor: Predictor) -> None:
    st.caption(
        "The file needs these columns: " + ", ".join(f"`{c}`" for c in predictor.feature_columns)
    )
    file = st.file_uploader(
        "CSV or Excel file to score", type=["csv", "txt", "xlsx", "xls"], key="batch_file"
    )
    if file is None:
        return
    try:
        df = read_tabular_file(file, file.name)
    except DataLoadError as exc:
        st.error(str(exc))
        return
    issues = predictor.validate_input(df)
    issues_view(issues)
    if any(i.level == "error" for i in issues):
        return
    quality = predictor.quality_report(df)
    if quality.empty:
        st.success(
            "The file looks like the training data: no parsing problems, no values "
            "outside the training range, no new categories.",
            icon=":material/check:",
        )
    else:
        st.warning(
            "Some values differ from the training data. Predictions are still made, "
            "but check these columns:",
            icon=":material/warning:",
        )
        st.dataframe(quality, hide_index=True, width="stretch")
    scored = predictor.predict_batch(df)
    st.dataframe(scored.head(200), hide_index=True, width="stretch")
    st.download_button(
        "Download predictions (CSV)",
        scored.to_csv(index=False).encode(),
        "predictions.csv",
        "text/csv",
        key="dl_pred",
    )


def render(state: AppState) -> None:
    st.header("6. Use a model")
    state = _input_spec(state)
    run = state.run
    ok = {r.name: r for r in run.results if r.status == "ok"}
    names = list(ok)
    choice = st.selectbox(
        "Model",
        names,
        index=names.index(run.best().name),
        key="predict_model",
        help="Only fully validated models are listed.",
    )
    result = ok[choice]
    threshold = state.thresholds.get(result.key)
    predictor = Predictor(
        result.pipeline,
        run.config,
        run.class_labels,
        state.input_spec,
        threshold=threshold,
        positive_index=run.positive_index,
    )
    if threshold is not None:
        st.caption(
            f"Using your decision threshold {threshold:.2f} for "
            f"'{run.class_labels[run.positive_index]}' (set in step 5)."
        )
    single, batch, export = st.tabs(["Single record", "Batch file", "Download model"])
    with single:
        _single(predictor, state)
    with batch:
        _batch(predictor)
    with export:
        st.warning(
            "Model bundles are Python pickle files. Only load bundles you created "
            "yourself; never open one from an untrusted source.",
            icon="\U0001f512",
        )
        _downloads(result, run, state, threshold)
    nav_buttons(back=5, next_step=None)


LOAD_SNIPPET = """from tabulaml.inference.persistence import load_bundle

predictor = load_bundle("{file}")          # works for .joblib and .pkl
predictions = predictor.predict_batch(new_data_frame)"""


def _downloads(result, run, state: AppState, threshold: float | None) -> None:  # noqa: ANN001
    """Both formats: .joblib (compressed, scikit-learn convention) and .pkl (standard pickle).

    Files are built once per model and kept for the session, so a download click (which
    reruns the page) does not make the buttons disappear.
    """
    cache_key = f"bundles_{result.key}_{threshold}"
    if cache_key not in st.session_state:
        if not st.button("Prepare model files", key="prep_bundle", type="primary"):
            st.caption(
                "Builds a .joblib and a .pkl file containing the full pipeline "
                "(cleaning, preprocessing, model), its settings and the threshold."
            )
            return
        with st.spinner("Packaging the trained pipeline..."):
            st.session_state[cache_key] = {
                fmt: bundle_bytes(result, run, state.input_spec, threshold, fmt=fmt)
                for fmt in ("joblib", "pickle")
            }
    files = st.session_state[cache_key]
    left, right = st.columns(2)
    left.download_button(
        f"Download .joblib ({len(files['joblib']) / 1e6:.1f} MB)",
        files["joblib"],
        f"tabulaml_{result.key}.joblib",
        "application/octet-stream",
        key="dl_bundle_joblib",
        width="stretch",
        icon=":material/download:",
    )
    right.download_button(
        f"Download .pkl ({len(files['pickle']) / 1e6:.1f} MB)",
        files["pickle"],
        f"tabulaml_{result.key}.pkl",
        "application/octet-stream",
        key="dl_bundle_pickle",
        width="stretch",
        icon=":material/download:",
    )
    st.caption(
        ".joblib is compressed and the usual format for scikit-learn models; .pkl is "
        "a standard Python pickle. Both hold the same model."
    )
    st.code(LOAD_SNIPPET.format(file=f"tabulaml_{result.key}.joblib"), language="python")
