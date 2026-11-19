"""Bundled samples are genuine, publicly available Sri Lankan datasets with complete cards."""

import pytest

from tabulaml.core.types import ProblemType
from tabulaml.data.samples import list_samples, load_sample

BUNDLED = ["lk_colombo_dengue", "lk_tea_auction_catalogue", "lk_tea_auction_price"]


@pytest.mark.parametrize("key", BUNDLED)
def test_bundled_sample_is_public_and_card_matches_data(key):
    df, card = load_sample(key)
    assert not card.is_synthetic
    assert card.license.startswith(("CC0", "CC-BY"))
    assert "Public dataset" in card.provenance
    assert card.target in df.columns
    assert set(card.features) == set(df.columns)
    assert set(card.recommended_exclusions) <= set(df.columns)


def test_no_synthetic_samples_are_shipped():
    assert all(not c.is_synthetic for c in list_samples())


def test_list_samples_filters_by_problem_type():
    cls = {c.key for c in list_samples(ProblemType.CLASSIFICATION)}
    reg = {c.key for c in list_samples(ProblemType.REGRESSION)}
    assert "lk_tea_auction_catalogue" in cls and not cls & reg
    assert {"lk_tea_auction_price", "lk_colombo_dengue"} <= reg


def test_two_tasks_share_one_tea_file():
    a, _ = load_sample("lk_tea_auction_catalogue")
    b, _ = load_sample("lk_tea_auction_price")
    assert a.shape == b.shape == (12233, 26)


def test_unknown_sample_raises():
    with pytest.raises(KeyError):
        load_sample("nope")
