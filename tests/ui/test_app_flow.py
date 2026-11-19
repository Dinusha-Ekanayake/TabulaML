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
    at = _click(at, "train")
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
    at = _click(at, "train")
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
    assert "CC0" in html and "colmozzie" in html and "GitHub repository" in html


def test_model_cards_and_selection_bar():
    at = _answer_leaks(_to_step3("classification", "Tea Auction"))
    at = _click(at, "next_4")
    _no_errors(at)
    at = _click(at, "rec_set")
    assert at.toggle(key="m_rf").value and at.toggle(key="m_catboost").value
    assert any("5 selected" in str(m.value) for m in at.markdown)
    at = _click(at, "none_Boosting")
    assert not at.toggle(key="m_xgb").value
