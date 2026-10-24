import optuna
import pytest
from nba_ou.modeling.optuna_error_line import build_xgb_params_error_line
from nba_ou.modeling.optuna_total_points import build_xgb_params_total_points
from optuna.samplers import TPESampler
from pydantic import ValidationError

from training_pipeline.config import (
    UPSTREAM_SEARCH_SPACE,
    FloatRange,
    IntRange,
    SearchSpaceConfig,
)
from training_pipeline.tuning import build_xgb_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sample_params(builder, n_trials: int = 5) -> list[dict]:
    """Draw params from a seeded sampler so results are exactly reproducible."""
    captured: list[dict] = []
    study = optuna.create_study(sampler=TPESampler(seed=16))

    def objective(trial):
        captured.append(builder(trial))
        return 0.0

    study.optimize(objective, n_trials=n_trials)
    return captured


# `device` is a static XGBoost param added by build_xgb_params for GPU support;
# the upstream builders never set it. Strip it before comparing so the equality
# check keeps testing what it is meant to: parameter names, ranges, log flags,
# and suggest_* call order.
def _drop_device(params_list: list[dict]) -> list[dict]:
    return [{k: v for k, v in p.items() if k != "device"} for p in params_list]


def test_upstream_space_constant_reproduces_upstream_total_points_exactly():
    """UPSTREAM_SEARCH_SPACE must stay a faithful transcription of the
    hardcoded upstream space: same parameter names, ranges, log flags AND call
    order. A seeded TPE sampler draws per-parameter in call order, so identical
    draws prove all four match. This is what keeps pre-existing results
    reproducible now that the *defaults* deliberately differ.
    """
    upstream = _sample_params(
        lambda t: build_xgb_params_total_points(t, objective="reg:squarederror")
    )
    mine = _sample_params(
        lambda t: build_xgb_params(
            t, UPSTREAM_SEARCH_SPACE, objective_name="reg:squarederror"
        )
    )
    assert _drop_device(mine) == upstream


def test_upstream_space_constant_reproduces_upstream_error_line_exactly():
    upstream = _sample_params(
        lambda t: build_xgb_params_error_line(t, objective="reg:squarederror")
    )
    mine = _sample_params(
        lambda t: build_xgb_params(
            t, UPSTREAM_SEARCH_SPACE, objective_name="reg:squarederror"
        )
    )
    assert _drop_device(mine) == upstream


def test_defaults_deliberately_differ_from_upstream():
    """The defaults are tuned for p/n ~ 0.6-0.8 with a very weak signal, which
    the legacy space was not. Guard the two changes that matter most.
    """
    space = SearchSpaceConfig()
    assert space != UPSTREAM_SEARCH_SPACE

    # A depth-4 tree makes ~15 splits; choosing them from ~720 candidates
    # (the old 0.35 floor on ~2000 features) mines noise.
    assert space.colsample_bytree.low < UPSTREAM_SEARCH_SPACE.colsample_bytree.low
    assert space.colsample_bytree.low <= 0.05

    # For reg:squarederror with residual sigma ~19.5, chance split gains are
    # O(100s) -- the old 0.1-3.0 gamma range pruned nothing.
    assert space.gamma.high >= 100
    assert space.gamma.log is True

    # A leaf holding 5 of 2500 games is noise.
    assert space.min_child_weight.low >= 20
    # Stumps cannot manufacture spurious interactions.
    assert space.max_depth.low == 1


def test_default_space_stays_within_valid_xgboost_bounds():
    space = SearchSpaceConfig()
    assert 0 < space.colsample_bytree.low <= space.colsample_bytree.high <= 1.0
    assert 0 < space.subsample.low <= space.subsample.high <= 1.0
    assert space.max_depth.low >= 1
    for name in ("gamma", "min_child_weight", "reg_alpha", "reg_lambda"):
        assert getattr(space, name).low > 0, name


def test_widening_the_space_actually_changes_sampled_values():
    """Regression against dead config: the ranges must be honored, not
    decorative. A deeper max_depth range must eventually produce a depth
    outside the default 2..4.
    """
    space = SearchSpaceConfig(max_depth=IntRange(low=8, high=12))
    drawn = _sample_params(
        lambda t: build_xgb_params(t, space, objective_name="reg:squarederror"),
        n_trials=10,
    )
    depths = {p["max_depth"] for p in drawn}
    assert depths, "no params drawn"
    assert all(8 <= d <= 12 for d in depths)


def test_search_space_controls_n_estimators_and_early_stopping():
    space = SearchSpaceConfig(n_estimators=250, early_stopping_rounds=15)
    params = _sample_params(
        lambda t: build_xgb_params(t, space, objective_name="reg:squarederror"),
        n_trials=1,
    )[0]
    assert params["n_estimators"] == 250
    assert params["early_stopping_rounds"] == 15


def test_objective_name_is_passed_through():
    params = _sample_params(
        lambda t: build_xgb_params(
            t, SearchSpaceConfig(), objective_name="reg:pseudohubererror"
        ),
        n_trials=1,
    )[0]
    assert params["objective"] == "reg:pseudohubererror"


def test_ranges_reject_inverted_bounds():
    with pytest.raises(ValidationError):
        IntRange(low=5, high=2)
    with pytest.raises(ValidationError):
        FloatRange(low=1.0, high=0.5)


def test_log_scaled_range_rejects_non_positive_low():
    """log=True with low<=0 is mathematically invalid and Optuna would raise
    at sample time -- catch it at config construction instead.
    """
    with pytest.raises(ValidationError, match="log-scaled"):
        FloatRange(low=0.0, high=1.0, log=True)


def test_search_space_round_trips_through_json():
    space = SearchSpaceConfig(
        max_depth=IntRange(low=3, high=9),
        learning_rate=FloatRange(low=0.01, high=0.2, log=True),
    )
    assert SearchSpaceConfig.model_validate_json(space.model_dump_json()) == space


# --- recency-weighting search space ------------------------------------------


def test_default_lambda_bounds_are_not_aggressive():
    """lambda above ~0.005 gives a sub-5-month half-life, which shrinks a
    2500-game window to a few weeks of effective data.
    """
    from training_pipeline.config import SampleWeightConfig

    low, high = SampleWeightConfig().lambda_bounds
    assert (low, high) == (0.0005, 0.005)
    # Oldest game in a ~770-day window keeps a meaningful share of its weight.
    import math

    assert math.exp(-high * 770) > 0.02
    assert math.exp(-low * 770) < 0.75


def test_optuna_can_choose_not_to_weight_at_all():
    """allow_unweighted turns "off" into a real option the sampler evaluates,
    rather than something unreachable at the bottom of a log-uniform range.
    """
    from training_pipeline.config import SampleWeightConfig
    from training_pipeline.tuning import (
        USE_SAMPLE_WEIGHT_PARAM,
        _resolve_trial_sample_weight_lambda,
    )

    sw = SampleWeightConfig(enabled=True, tune_lambda=True, allow_unweighted=True)
    seen = set()
    study = optuna.create_study(sampler=TPESampler(seed=3))

    def objective(trial):
        seen.add(_resolve_trial_sample_weight_lambda(trial, sw) is None)
        return 0.0

    study.optimize(objective, n_trials=25)

    assert True in seen, "should sometimes decline weighting"
    assert False in seen, "should sometimes apply weighting"
    for trial in study.trials:
        if trial.params.get(USE_SAMPLE_WEIGHT_PARAM) is False:
            # Conditional space: no decay rate sampled when weighting is off.
            assert "sample_weight_lambda" not in trial.params
        else:
            lam = trial.params["sample_weight_lambda"]
            assert 0.0005 <= lam <= 0.005


def test_disabling_allow_unweighted_always_weights():
    from training_pipeline.config import SampleWeightConfig
    from training_pipeline.tuning import (
        USE_SAMPLE_WEIGHT_PARAM,
        _resolve_trial_sample_weight_lambda,
    )

    sw = SampleWeightConfig(enabled=True, tune_lambda=True, allow_unweighted=False)
    study = optuna.create_study(sampler=TPESampler(seed=3))
    study.optimize(
        lambda t: (_resolve_trial_sample_weight_lambda(t, sw), 0.0)[1], n_trials=8
    )

    for trial in study.trials:
        assert USE_SAMPLE_WEIGHT_PARAM not in trial.params
        assert "sample_weight_lambda" in trial.params
