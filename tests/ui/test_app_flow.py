"""Drive the real Streamlit app headlessly through the whole wizard."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tabulaml.models import registry

APP = str(Path(__file__).resolve().parents[2] / "app.py")
TIMEOUT = 300


@pytest.fixture(autouse=True)
def _fast_models(monkeypatch):
    original = registry.get_model

    def small(key):
        spec = original(key)

        def factory(ptype, rs, _f=spec.factory):
            est = _f(ptype, rs)
            for name in ("n_estimators", "iterations"):
                if name in est.get_params():
                    est.set_params(**{name: 30})
            return est

        return spec.__class__(**{**spec.__dict__, "factory": factory})

    monkeypatch.setattr("tabulaml.training.pipeline_factory.get_model", small)


def _no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    problems = [e.value for e in at.error if "Something went wrong" in str(e.value)]
    assert not problems, problems


def _click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click()
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
    at = _to_step3("classification", "synthetic")
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "disconnection_date" in warnings  # planted leak surfaced in the quality report
    at = _click(at, "next_4")
    for key in ("m_logreg", "m_rf"):
        at.checkbox(key=key).check()
    at = at.run()
    at = _click(at, "train")
    _no_errors(at)
    assert at.session_state["tml_state"].step == 5
    assert at.session_state["tml_state"].run.best().status == "ok"
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
    at.checkbox(key="m_ridge").check()
    at.checkbox(key="m_rf").check()
    at = at.run()
    at = _click(at, "train")
    _no_errors(at)
    run = at.session_state["tml_state"].run
    assert {r.status for r in run.results} == {"ok"}


def test_changing_problem_type_resets_downstream_state():
    at = _to_step3("classification", "synthetic")
    assert at.session_state["tml_state"].config is not None
    at = _click(at, "nav_1")
    at = _click(at, "choose_regression")
    state = at.session_state["tml_state"]
    assert state.raw_df is None and state.config is None and state.step == 2
