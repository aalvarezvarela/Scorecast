"""Out-of-fold factorization machine on the monthly folds (phase 6, step 6a).

For each fold (``pretraining_oof.monthly_folds``): train on the stints (and,
with ``edges``, the attacker-games) dated before the checkpoint, warm-started
from the previous fold's model; predict the fold's stints and attacker-games;
store the predictions (``oof/<name>/``, ``edges_oof/<name>/``) and the
checkpoint (``fm/<name>/fold=YYYY-MM.pt``). A fold already stored is skipped
(its checkpoint is the next fold's warm start), so a run resumes.

Variants (the plan's ladder): ``additive`` (no interaction term), ``no_matchup``
(chemistry and style), ``matchup`` (all stint terms), ``matchup_edges`` (plus
the time-attributed edge targets).

``res_*`` (decided 2026-10-15): the same models learning the **residual over
the daily as-of RAPM** (``pretraining_daily_rapm.py``) for points per
possession and pace, ``y - base`` with the base of each stint's own date
(training and prediction alike), predicting ``base + residual``; the other
four targets keep their own heads. Stints without a base (before 2016-12-01)
give those two targets no weight.

``int_*`` (decided 2026-10-15): interaction-only residuals over the same
base: no per-player additive term for points and pace (RAPM holds them) and
interaction vectors centered on their running mean, so no interaction term
can act as a hidden rating; ``int_control`` keeps only the global intercept
and home term for them.

``gnn_*`` (step 6b, decided 2026-10-15): the relational GNN (``gnn.py``,
width 32, two layers) as an interaction-only residual over the same base, with
the additive absorber (trained, dropped at prediction): ``gnn_control``
(intercept / home only), ``gnn_node`` (no message passing),
``gnn_no_guard`` (teammate and opponent edges), ``gnn_full`` (+ guard edges);
the last three have identical parameters.

``edge_*`` (the edge-head ablation): edge-only models, no stint loss,
predicting the attacker-game targets from the attacker alone
(``edge_attacker``), plus the defenders' terms (``edge_defender``), plus the
matchup product (``edge_matchup``); the capacity control ``edge_capacity``
(attacker x a learned defender-independent vector) and the defenders' terms
on top of it (``edge_capacity_defender``, ``edge_capacity_matchup``). Their
folds start where training edges exist (2017-11).

Hyperparameters are the untuned defaults of ``FMConfig`` (dimension 16).
Design runs stay before 2019-07-01 (``--until``): 2_7's evaluation starts in
2019-20. This script trains small models; it is not a long run unless asked
for many folds.

    # tiny CPU smoke run: three folds, two epochs, a training subsample
    python scripts/player_graph/fm_oof.py --variant matchup_edges \\
        --name smoke_matchup_edges --until 2017-03-01 --epochs 2 --max-train 20000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nba_ou.data_processing.player_graph.edge_targets import TARGETS, to_edge_batch
from nba_ou.data_processing.player_graph.fm import (
    FactorizationMachine,
    FMConfig,
    PlayerVocab,
    edge_tensors,
    finalize_centering,
    fit_standardization,
    load_checkpoint,
    predict,
    predict_edges,
    save_checkpoint,
    stint_tensors,
    train,
    warm_start,
)
from nba_ou.data_processing.player_graph.gnn import GNN, GNNConfig
from nba_ou.data_processing.player_graph.pretraining_data import (
    FIRST_MATCHUP_SEASON,
    NODE_COLUMNS,
    ROOT,
    VERSION,
    load_stints,
    season_path,
)
from nba_ou.data_processing.player_graph.pretraining_oof import (
    PredictionStore,
    check_fold,
    monthly_folds,
    residual_stints,
)
from nba_ou.data_processing.player_graph.stint_graph import LABELS

_ADDITIVE = FMConfig(chemistry=False, style=False, matchup=False, edges=False)
_EDGE_ONLY = replace(_ADDITIVE, edges=True, stint_loss=False)
VARIANTS = {
    "additive": _ADDITIVE,
    "no_matchup": FMConfig(matchup=False, edges=False),
    "matchup": FMConfig(edges=False),
    "matchup_edges": FMConfig(),
    "edge_attacker": replace(_EDGE_ONLY, edge_defender=False, edge_matchup=False),
    "edge_defender": replace(_EDGE_ONLY, edge_matchup=False),
    "edge_matchup": _EDGE_ONLY,
}
VARIANTS |= {
    f"res_{k}": VARIANTS[k]
    for k in ("additive", "no_matchup", "matchup", "matchup_edges")
}
RESIDUAL_TARGETS = ("pts_per_poss", "poss_per_48")
BASE = "rapm_daily"
# Interaction-only residuals (decided 2026-10-15): no per-player additive
# term for the residual targets, interaction vectors centered.
_NO_ADDITIVE = tuple(LABELS.index(t) for t in RESIDUAL_TARGETS)
VARIANTS |= {
    "int_control": replace(_ADDITIVE, no_additive=_NO_ADDITIVE),
    "int_chemistry_style": FMConfig(
        matchup=False, edges=False, no_additive=_NO_ADDITIVE, center_interactions=True
    ),
    "int_matchup": FMConfig(
        edges=False, no_additive=_NO_ADDITIVE, center_interactions=True
    ),
    # Edge-head capacity control: the attacker x learned-vector product
    # without the defenders, then the defenders' terms on top of it.
    "edge_capacity": replace(
        _EDGE_ONLY, edge_defender=False, edge_matchup=False, edge_capacity=True
    ),
    "edge_capacity_defender": replace(
        _EDGE_ONLY, edge_matchup=False, edge_capacity=True
    ),
    "edge_capacity_matchup": replace(_EDGE_ONLY, edge_capacity=True),
}
# GNN ladder (decided 2026-10-15): B control, C node-only, D teammate /
# opponent, E full; C-E capacity-matched; all with the additive absorber.
_GNN = GNNConfig(no_additive=_NO_ADDITIVE, absorb_additive=True)
VARIANTS |= {
    "gnn_control": replace(_GNN, relations=(), readout=False),
    "gnn_node": replace(_GNN, relations=()),
    "gnn_no_guard": replace(_GNN, relations=("teammate", "opponent")),
    "gnn_full": _GNN,
}
RESIDUAL_PREFIXES = ("res_", "int_", "gnn_")
EVALUATION_START = pd.Timestamp("2019-07-01")


def load_edges(root: Path, seasons):
    parts = []
    for season in seasons:
        if season < FIRST_MATCHUP_SEASON:
            continue
        edges = pd.read_parquet(season_path(root, "edges", season))
        nodes = pd.read_parquet(season_path(root, "nodes", season))
        batch, _ = to_edge_batch(edges, nodes, NODE_COLUMNS)
        parts.append(batch)
    if not parts:
        return None
    from nba_ou.data_processing.player_graph.edge_targets import EdgeBatch

    offset = np.cumsum([0, *[len(p) for p in parts[:-1]]])
    return EdgeBatch(
        **{
            name: np.concatenate([getattr(p, name) for p in parts])
            for name in EdgeBatch.__dataclass_fields__
            if name != "pair_game_index"
        },
        pair_game_index=np.concatenate(
            [p.pair_game_index + o for p, o in zip(parts, offset, strict=True)]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--name", help="Output name (default: fm_<variant>)")
    parser.add_argument(
        "--dim", type=int, help="Embedding width (default: the variant's own)"
    )
    parser.add_argument("--from", dest="start", default="2016-12-01")
    parser.add_argument("--until", default=str(EVALUATION_START.date()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-train", type=int, help="Smoke: subsample training stints"
    )
    parser.add_argument("--allow-evaluation-seasons", action="store_true")
    args = parser.parse_args()
    until = pd.Timestamp(args.until)
    if until > EVALUATION_START and not args.allow_evaluation_seasons:
        raise SystemExit(f"Design runs stop before {EVALUATION_START.date()}")
    name = args.name or f"fm_{args.variant}"
    config = VARIANTS[args.variant]
    if args.dim is not None:
        config = replace(config, dim=args.dim)
    root = args.local_root / ROOT / VERSION
    seasons = range(2016, until.year + 1)
    torch.manual_seed(args.seed)

    stints = load_stints(root, seasons, until=until)
    residual = args.variant.startswith(RESIDUAL_PREFIXES)
    edge_only = not config.stint_loss
    base = None
    if residual:
        base = PredictionStore(root / "oof").read(BASE, stints.game_id, stints.seg_idx)
        stints = residual_stints(stints, base, RESIDUAL_TARGETS)
    edges = load_edges(root, seasons) if config.edges else None
    folds = [
        f for f in monthly_folds(stints.game_date, first=args.start) if f.until <= until
    ]
    store = PredictionStore(root / "oof")
    edge_dir = root / "edges_oof" / name
    ckpt_dir = root / "fm" / name
    rng = np.random.default_rng(args.seed)
    model = vocab = None
    print(
        f"{name}: {asdict(config)}\n{len(folds)} folds, {len(stints):,} stints",
        flush=True,
    )
    for fold in folds:
        ckpt = ckpt_dir / f"fold={fold.label}.pt"
        done = (
            (edge_dir / f"fold={fold.label}.parquet").exists()
            if edge_only
            else store.has(name, fold)
        )
        if done and ckpt.exists():
            model, vocab, _, _ = load_checkpoint(ckpt)
            continue
        started = time.time()
        train_mask, predict_mask = fold.masks(stints.game_date)
        train_rows = np.flatnonzero(train_mask)
        if args.max_train and len(train_rows) > args.max_train:
            train_rows = np.sort(rng.choice(train_rows, args.max_train, replace=False))
        train_stints = stints.subset(train_rows)
        check_fold(fold, train_stints.game_date, stints.game_date[predict_mask])
        train_edges = predict_edges_batch = None
        if edges is not None:
            edge_train, edge_predict = fold.masks(edges.game_date)
            train_edges = edges.subset(edge_train) if edge_train.any() else None
            predict_edges_batch = (
                edges.subset(edge_predict) if edge_predict.any() else None
            )
        if edge_only and (train_edges is None or predict_edges_batch is None):
            continue
        std = fit_standardization(train_stints, train_edges)
        new_vocab = PlayerVocab(train_stints.players.reshape(-1))
        if train_edges is not None:
            new_vocab.extend([*train_edges.attacker, *train_edges.defender])
        if vocab is not None:
            grown = PlayerVocab(vocab.ids)
            grown.extend(new_vocab.ids)
            new_vocab = grown
        # Seeded per fold, so a resumed run equals an uninterrupted one.
        fold_seed = args.seed * 10_000 + int(fold.checkpoint.strftime("%y%m"))
        torch.manual_seed(fold_seed)
        model_class = GNN if isinstance(config, GNNConfig) else FactorizationMachine
        new_model = model_class(config, len(new_vocab))
        if model is not None:
            warm_start(new_model, model, new_vocab, vocab)
        model, vocab = new_model, new_vocab
        graph = isinstance(config, GNNConfig)
        tensors = (
            None if edge_only else stint_tensors(train_stints, vocab, std, guard=graph)
        )
        edge_t = (
            edge_tensors(train_edges, vocab, std) if train_edges is not None else None
        )
        history = train(
            model,
            tensors,
            edge_t,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=fold_seed,
            device=args.device,
        )
        if tensors is not None:
            # Exact training-set centering mean (2026-10-15), from this fold's
            # training stints only.
            model.to("cpu")
            finalize_centering(model, tensors)
        target = stints.subset(predict_mask)
        predictions = None
        uncorrected: list[str] = []
        if not edge_only:
            predictions = predict(
                model,
                stint_tensors(target, vocab, std, guard=graph),
                std,
                device=args.device,
            )
            if residual:  # base + learned residual
                for t in RESIDUAL_TARGETS:
                    k = LABELS.index(t)
                    if train_stints.label_weight[:, :, k].sum() == 0:
                        # No training residual yet (the first fold): no
                        # correction, the base alone.
                        predictions[:, :, k] = 0.0
                        uncorrected.append(t)
                    predictions[:, :, k] += base[predict_mask][:, :, k]
        model.to("cpu")
        meta = {
            "variant": args.variant,
            "residual_over": BASE if residual else None,
            "residual_targets": list(RESIDUAL_TARGETS) if residual else [],
            "config": asdict(config),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "max_train": args.max_train,
            "dataset": str(root),
        }
        # The fold's marker (the prediction store's fold file, or the edge file
        # of an edge-only model) is written last, so an interrupted fold is
        # redone whole on the next run.
        log = {
            **meta,
            "fold": fold.label,
            "history": history,
            "seconds": time.time() - started,
            "train_stints": 0 if edge_only else len(train_stints),
            "train_attacker_games": len(train_edges) if train_edges is not None else 0,
            "players": len(vocab),
            "predicted_stints": 0 if edge_only else len(target),
            "uncorrected_residual_targets": uncorrected,
            "centering": (
                "exact training mean"
                if getattr(config, "center_interactions", False)
                else None
            ),
        }
        save_checkpoint(ckpt, model, vocab, std, log)
        (ckpt_dir / f"fold={fold.label}.json").write_text(
            json.dumps(log, indent=2, default=str) + "\n"
        )
        if predict_edges_batch is not None and train_edges is not None:
            values = predict_edges(
                model, edge_tensors(predict_edges_batch, vocab, std), std
            )
            frame = pd.DataFrame(
                {
                    "game_id": predict_edges_batch.game_id,
                    "off_player_id": predict_edges_batch.attacker,
                }
            )
            for k, t in enumerate(TARGETS):
                frame[t] = values[:, k]
            edge_dir.mkdir(parents=True, exist_ok=True)
            tmp = edge_dir / f"fold={fold.label}.tmp"
            frame.to_parquet(tmp, index=False)
            if not edge_only:
                tmp.replace(edge_dir / f"fold={fold.label}.parquet")
        if not edge_only:
            store.write(name, fold, target.game_id, target.seg_idx, predictions, meta)
        else:
            tmp.replace(edge_dir / f"fold={fold.label}.parquet")
        last = history[-1]
        print(
            f"{fold.label}: train {0 if edge_only else len(train_stints):,} stints"
            + (
                f" + {len(train_edges):,} attacker-games"
                if train_edges is not None
                else ""
            )
            + f", predict {len(target):,}; loss stint {last['stint']:.4f}"
            + (f" edge {last['edge']:.4f}" if config.edges else "")
            + f" ({time.time() - started:.0f}s)",
            flush=True,
        )
    (root / "fm" / name).mkdir(parents=True, exist_ok=True)
    (root / "fm" / name / "run.json").write_text(
        json.dumps(
            {
                "args": vars(args) | {"local_root": str(args.local_root)},
                "folds": [f.label for f in folds],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
