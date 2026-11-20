"""Drive the real Streamlit app headlessly through the whole wizard."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")
TIMEOUT = 300


pytestmark = pytest.mark.usefixtures("fast_models")


def _no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    problems = [e.value for e in at.error if "Something went wrong" in str(e.value)]
    assert not problems, problems


def _click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click()
    return at.run(timeout=TIMEOUT)


def _train(at: AppTest) -> AppTest:
    """Press Train & Predict, wait for the background job, then let the app pick it up."""
    at = _click(at, "train")
    job = at.session_state["tml_training_job"]["job"]
    assert job.wait(TIMEOUT).state in ("done", "cancelled", "failed")
    return at.run(timeout=TIMEOUT)


def _answer_leaks(at: AppTest) -> AppTest:
    for radio in at.radio:
        if str(radio.key or "").startswith("leak_"):
            radio.set_value("No: keep it excluded")
    return at.run(timeout=TIMEOUT)


def _to_step3(problem: str, sample_title_part: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    _no_errors(at)
    at = _click(at, f"choose_{problem}")
    select = at.selectbox(key="sample_choice")
    select.set_value(next(o for o in select.options if sample_title_part in o))
    at = select.run() if hasattr(select, "run") else at.run()
    at = _click(at, "load_sample")
    _no_errors(at)
    at = _click(at, "next_3")
    _no_errors(at)
    return at


def test_classification_flow_end_to_end():
    at = _to_step3("classification", "Tea Auction")
    # The real structural leak ('category') must be decided by the user before step 4.
    assert at.button(key="next_4").disabled
    at = _answer_leaks(at)
    assert not at.button(key="next_4").disabled
    at = _click(at, "next_4")
    for key in ("m_logreg", "m_rf"):
        at.toggle(key=key).set_value(True)
    at = at.run()
    at = _train(at)
    _no_errors(at)
    assert at.session_state["tml_state"].step == 5
    assert at.session_state["tml_state"].run.best().status == "ok"
    # Holdout lock: test columns hidden until the user reveals them on purpose.
    assert not at.session_state["tml_state"].test_revealed
    at = _click(at, "reveal_test")
    assert at.session_state["tml_state"].test_revealed
    at = _click(at, "next_6")
    _no_errors(at)
    at.button(key="FormSubmitter:single_record-Predict").click()
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    assert any("Prediction" in str(s.value) for s in at.success)


def test_regression_flow_on_real_dengue_data():
    at = _to_step3("regression", "dengue")
    at.radio(key="pp_mode").set_value("Custom")
    at = at.run()
    _no_errors(at)
    at = _click(at, "next_4")
    at.toggle(key="m_ridge").set_value(True)
    at.toggle(key="m_rf").set_value(True)
    at = at.run()
    at = _train(at)
    _no_errors(at)
    run = at.session_state["tml_state"].run
    assert {r.status for r in run.results} == {"ok"}


def test_changing_problem_type_resets_downstream_state():
    at = _to_step3("classification", "Tea Auction")
    assert at.session_state["tml_state"].config is not None
    at = _click(at, "nav_1")
    at = _click(at, "choose_regression")
    state = at.session_state["tml_state"]
    assert state.raw_df is None and state.config is None and state.step == 2


def _footer_html(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown if "tml-footer" in str(m.value))


def test_footer_on_every_step_with_dataset_attribution():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert "No dataset loaded" in _footer_html(at) or "no dataset loaded" in _footer_html(at)
    at = _to_step3("regression", "dengue")
    html = _footer_html(at)
    assert "CC0" in html and "colmozzie" in html and "GitHub" in html
    assert "&copy;" in html and "Dinusha Ekanayake" in html


def test_model_cards_and_selection_bar():
    at = _answer_leaks(_to_step3("classification", "Tea Auction"))
    at = _click(at, "next_4")
    _no_errors(at)
    at = _click(at, "rec_set")
    assert at.toggle(key="m_rf").value and at.toggle(key="m_catboost").value
    assert any("5 selected" in str(m.value) for m in at.markdown)
    at = _click(at, "none_Boosting")
    assert not at.toggle(key="m_xgb").value


def test_custom_mode_lets_user_choose_imputation_per_column():
    at = _to_step3("regression", "dengue")
    at.radio(key="pp_mode").set_value("Custom")
    at = at.run(timeout=TIMEOUT)
    at.toggle(key="imp_show_all").set_value(True)
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    box = next(
        s for s in at.selectbox if str(s.key or "").startswith("imp_") and "bulk" not in s.key
    )
    column = box.key.removeprefix("imp_")
    box.set_value("mean")
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    config = at.session_state["tml_state"].config
    chosen = next(c for c in config.columns if c.name == column)
    assert chosen.impute == "mean" and config.mode == "custom"


def test_several_targets_flow():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at = _click(at, "choose_classification")
    select = at.selectbox(key="sample_choice")
    select.set_value(next(o for o in select.options if "Tea Auction" in o))
    at = at.run()
    at = _click(at, "load_sample")
    at.toggle(key="multi_toggle").set_value(True)
    at = at.run(timeout=TIMEOUT)
    at.multiselect(key="extra_targets").set_value(["tier"])
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    assert at.session_state["tml_state"].extra_targets == ("tier",)
    at = _click(at, "next_3")
    at = _answer_leaks(at)
    _no_errors(at)
    config = at.session_state["tml_state"].config
    assert config.targets == ("catalogue", "tier")
    at = _click(at, "next_4")
    at.toggle(key="m_dt").set_value(True)
    at = at.run(timeout=TIMEOUT)
    at = _train(at)
    _no_errors(at)
    run = at.session_state["tml_state"].run
    assert run.is_multi_output and run.best().per_target
    at = _click(at, "reveal_test")
    _no_errors(at)
    at = _click(at, "next_6")
    at.button(key="FormSubmitter:single_record-Predict").click()
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    assert {m.label for m in at.metric} >= {"catalogue", "tier"}


def test_forecasting_flow_on_real_dengue_data():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at = _click(at, "choose_forecasting")
    assert at.session_state["tml_state"].forecasting
    at = _click(at, "load_sample")
    _no_errors(at)
    at.number_input(key="fc_season").set_value(0)
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    state = at.session_state["tml_state"]
    assert state.forecast is not None and state.order_column == "week_start"
    at = _click(at, "next_3")
    _no_errors(at)
    assert at.session_state["tml_state"].config.forecast is not None
    at = _click(at, "next_4")
    for key in ("m_naive", "m_ridge", "m_rf_arima"):
        at.toggle(key=key).set_value(True)
    at = at.run(timeout=TIMEOUT)
    at = _train(at)
    _no_errors(at)
    run = at.session_state["tml_state"].run
    assert {r.status for r in run.results} == {"ok"}, [r.error for r in run.results]
    at = _click(at, "reveal_test")
    _no_errors(at)
    at = _click(at, "next_6")
    _no_errors(at)
    at = _click(at, "fc_go")
    _no_errors(at)
    assert any(df.value.shape[0] == run.config.forecast.horizon for df in at.dataframe)


def test_grid_search_from_the_settings_card():
    at = _to_step3("regression", "dengue")
    at = _click(at, "next_4")
    at.toggle(key="m_ridge").set_value(True)
    at = at.run(timeout=TIMEOUT)
    at.radio(key="mode_ridge").set_value("Grid search")
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    assert any("combinations" in str(c.value) for c in at.caption)
    at = _train(at)
    _no_errors(at)
    ridge = at.session_state["tml_state"].run.result_for("ridge")
    assert ridge.status == "ok" and len(ridge.tuning_history) == 3  # alpha: 3 values


def test_rows_card_options_recreate_the_split():
    at = _to_step3("regression", "dengue")
    at.radio(key="pp_mode").set_value("Custom")
    at = at.run(timeout=TIMEOUT)
    before = at.session_state["tml_state"].auto.n_dev
    at.radio(key="rs_missing_rows").set_value("drop")
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    state = at.session_state["tml_state"]
    assert state.config.missing_rows == "drop" and state.auto.n_dev < before
    at.toggle(key="rs_dups").set_value(False)
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    assert at.session_state["tml_state"].config.drop_duplicates is False


def test_explore_data_tab_shows_distributions_and_relations():
    at = _to_step3("classification", "Tea Auction")
    at.selectbox(key="eda_column").set_value("price_mid_lkr")
    at = at.run(timeout=TIMEOUT)
    _no_errors(at)
    labels = {m.label for m in at.metric}
    assert {"Mean", "Median", "Skew"} <= labels


def test_results_page_explains_when_no_model_is_validated(monkeypatch):
    import tabulaml.training.trainer as trainer

    def broken(model, X, y_like):
        raise ValueError("fold exploded")

    monkeypatch.setattr(trainer, "_predictions", broken)
    at = _to_step3("regression", "dengue")
    at = _click(at, "next_4")
    at.toggle(key="m_ridge").set_value(True)
    at = at.run(timeout=TIMEOUT)
    at = _train(at)
    _no_errors(at)
    texts = " ".join(str(e.value) for e in at.error) + " ".join(str(m.value) for m in at.markdown)
    assert "fold exploded" in texts or "fold exploded" in " ".join(str(w.value) for w in at.warning)
    assert "No model was fully validated" in texts
