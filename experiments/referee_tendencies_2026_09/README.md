# Referee tendency comparisons on schema 2_5

The 12 cells use the same merged injury/referee dataset and vary only the
referee columns retained for training within each prediction strategy. The
exclusion lists contain the exact feature names; their intended coverage is
checked by `tests/test_referee_campaign_configs.py`.

The configs point to `data/train_data/training_data_2_5_20260704.csv`. Generate
that file before running the campaign, then set `data.expected_checksum` in
each YAML file to the checksum printed by `create_train_data.py`. No checksum
has been recorded yet because the merged dataset has not been regenerated.
