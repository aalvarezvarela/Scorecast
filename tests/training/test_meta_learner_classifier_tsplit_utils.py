import optuna

from lab.meta_learner.meta_learner_classifier_tsplit_utils import (
    select_best_trial_lexicographic_classifier,
    select_best_trial_min_log_loss_classifier,
)


def test_select_best_trial_min_log_loss_classifier_ignores_accuracy_tiebreaking() -> None:
    study = optuna.create_study(direction="minimize")

    low_loss_trial = study.ask()
    low_loss_trial.set_user_attr("mean_log_loss", 0.68)
    low_loss_trial.set_user_attr("mean_accuracy", 0.52)
    low_loss_trial.set_user_attr("mean_balanced_accuracy", 0.52)
    low_loss_trial.set_user_attr("mean_brier", 0.24)
    study.tell(low_loss_trial, 0.68)

    high_accuracy_trial = study.ask()
    high_accuracy_trial.set_user_attr("mean_log_loss", 0.689)
    high_accuracy_trial.set_user_attr("mean_accuracy", 0.61)
    high_accuracy_trial.set_user_attr("mean_balanced_accuracy", 0.60)
    high_accuracy_trial.set_user_attr("mean_brier", 0.25)
    study.tell(high_accuracy_trial, 0.689)

    selected_min_loss = select_best_trial_min_log_loss_classifier(study)
    selected_lexicographic = select_best_trial_lexicographic_classifier(study)

    assert selected_min_loss.number == low_loss_trial.number
    assert selected_lexicographic.number == high_accuracy_trial.number
