# Yahoo percentage features, schema 2_5 rebuild

Closing historical builds and same-day prediction builds retain exactly 36
Yahoo percentage inputs. The contract lives in `nba_ou.config.yahoo_features`.

- Twelve raw percentages: totals over/under, spread home/away and moneyline
  home/away, for both tickets (`pct_bets`) and money (`pct_money`).
- Twenty-four history features: previous-five-game average and trend for
  totals-over, team spread and team moneyline tickets/money, separately for
  each team. Each history calculation excludes the current game.

No Yahoo under history, season means/stds, location-relative statistics or
home-minus-away differences are emitted. Raw observations are preserved; missing
values stay NaN. The compact set is protected through missingness, constant,
duplicate, correlation and seasonal-availability pruning. Both row-NA limits
ignore these optional inputs. Training fails if cleaning loses any of them, but
an explicit exclusion still wins: `cleaning.exclude_cols_containing: [pct_bets,
pct_money]` removes the whole compact block, as the "without Yahoo" ablation
configs expect, and `exclude_cols` can drop individual columns. Protection
applies only to the complete compact schema; archived full Yahoo schemas keep
their prior policy.

Intermediate builds retain only the 24 historical inputs. The current-game
percentages are not timestamped per prediction horizon, so the existing leakage
gate continues to exclude them. They must not be copied from closing data into
earlier prediction snapshots.

Yahoo's fill of missing SBR BetMGM total lines, spreads and moneylines is
unchanged, as are all other market-derived features. Missing raw Yahoo columns
are tolerated when scheduled odds are validated against historical odds.

## Rollout

1. Rebuild historical datasets with the existing creation scripts. They stay
   schema `2_5` and write Parquet, overwriting the earlier 2_5 `.parquet`
   files; the earlier 2_5 CSVs (full Yahoo family) are left untouched.
2. Point new experiment configurations at those files, update their data version
   and checksum, and retrain. Compare against the existing model on the same
   games; no performance improvement is presumed.
3. Verify saved feature schemas contain all 36 closing / 24 intermediate inputs.
   Exercise prediction with Yahoo present and absent before promotion.
4. Promote the new models through the existing registry workflow. Old models
   needing removed Yahoo features cannot use the new prediction frame; retain
   the old checkout/data for those models until the replacement is ready.

The code change does not rebuild data, fit models or change production channels.
