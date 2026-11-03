"""Qwen embeddings and within-document comparisons of team perspectives.

Uses the official Transformers recipe: last non-padding token, L2 normalization,
and optional task instructions in the ``Instruct: ...\nQuery:...`` format.
https://huggingface.co/Qwen/Qwen3-Embedding-4B
"""

from __future__ import annotations

import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from lab.text_to_features.team_perspective import document_perspectives

MODEL_ID = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
PARAMETERS = 4_021_774_336
# Pinned revision and parameter count of each supported model.
MODELS = {
    MODEL_ID: (MODEL_REVISION, PARAMETERS),
    "Qwen/Qwen3-Embedding-0.6B": (
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        595_776_512,
    ),
}
VARIANTS = ("raw", "home_target", "away_target")
PAIRS = (("raw", "home_target"), ("raw", "away_target"), ("home_target", "away_target"))
GIB = 1024**3


def embedding_input(text: str, instruction: str | None = None) -> str:
    """Apply Qwen's instruction format; a missing instruction preserves the text."""
    if instruction is None:
        return text
    if not instruction.strip():
        raise ValueError("instruction must be nonempty or None")
    return f"Instruct: {instruction}\nQuery:{text}"


def gpu_report(parameters: int = PARAMETERS) -> dict:
    """Inspect actual free VRAM; weight estimates exclude activation overhead."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with access to the NVIDIA GPU.")
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "free_gib": free / GIB,
        "total_gib": total / GIB,
        "fp16_weights_min_gib": parameters * 2 / GIB,
        "fp32_weights_min_gib": parameters * 4 / GIB,
    }


class QwenEmbedder:
    """One complete document at a time; reject overlength inputs, never truncate.

    Auto mode uses NF4 when unquantized weights plus 1.5 GiB headroom cannot fit.
    Pascal GPUs use FP32 compute (no BF16 support and slow native FP16).
    Every layer stays on the GPU; there is no implicit CPU offload.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        quantization: str = "auto",
        model_id: str = MODEL_ID,
    ):
        if quantization not in ("auto", "nf4", "none"):
            raise ValueError("quantization must be auto, nf4, or none")
        if model_id not in MODELS:
            raise ValueError(f"model_id must be one of {sorted(MODELS)}")
        revision, parameters = MODELS[model_id]
        self.hardware = gpu_report(parameters)
        pascal = self.hardware["compute_capability"][0] < 7
        dtype = torch.float32 if pascal else torch.float16
        required = self.hardware[
            "fp32_weights_min_gib" if pascal else "fp16_weights_min_gib"
        ]
        if quantization == "auto":
            quantization = (
                "nf4" if required + 1.5 > self.hardware["free_gib"] else "none"
            )
        if quantization == "none" and required + 1.5 > self.hardware["free_gib"]:
            raise RuntimeError("Unquantized weights exceed available VRAM; use NF4.")
        kwargs = {}
        if quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            padding_side="left",
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            dtype=dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
            **kwargs,
        ).eval()
        self.model.config.use_cache = False
        torch.cuda.synchronize()
        self.metadata = {
            **self.hardware,
            "model_id": model_id,
            "revision": revision,
            "quantization": quantization,
            "compute_dtype": str(dtype),
            "pooling": "last non-padding token; L2 normalization in FP32",
            "instruction": None,
            "batch_size": 1,
            "max_tokens": 32768,
            "truncation": False,
            "load_seconds": time.perf_counter() - started,
            "model_footprint_gib": self.model.get_memory_footprint() / GIB,
            "versions": {
                name: version(name)
                for name in ("torch", "transformers", "bitsandbytes", "accelerate")
            },
        }

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        *,
        instruction: str | None = None,
        progress_every: int = 1,
    ) -> tuple[np.ndarray, list[int]]:
        """``progress_every`` prints every n-th text and the last; 0 is silent."""
        self.metadata.update(
            {
                "instruction": instruction,
                "instruction_format": "Instruct: {instruction}\\nQuery:{text}"
                if instruction
                else None,
            }
        )
        embeddings, lengths = [], []
        for index, text in enumerate(texts):
            batch = self.tokenizer(
                [embedding_input(text, instruction)],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            length = int(batch.attention_mask.sum())
            if length > self.metadata["max_tokens"]:
                raise ValueError(
                    f"Text {index} has {length} tokens; refusing to truncate."
                )
            batch = batch.to("cuda")
            output = self.model(**batch, use_cache=False)
            # Left padding means the last position is the last real token.
            vector = torch.nn.functional.normalize(
                output.last_hidden_state[:, -1].float(), dim=1
            )
            if not torch.isfinite(vector).all() or vector.norm() == 0:
                raise RuntimeError(f"Invalid embedding for text {index}")
            embeddings.append(vector[0].cpu().numpy())
            lengths.append(length)
            del output, batch, vector
            done = index + 1
            if progress_every and (done % progress_every == 0 or done == len(texts)):
                print(f"Embedded {done}/{len(texts)}: {length} tokens", flush=True)
        return np.stack(embeddings), lengths


def compare_perspectives(
    documents, players, embedder, *, label_rule="any", instruction=None
):
    """Return three rows per document plus embeddings and exact transformed texts."""
    if not documents:
        raise ValueError("No documents to compare")
    records, texts = [], []
    for doc in documents:
        perspectives = document_perspectives(
            doc, players=players[doc["game_id"]], label_rule=label_rule
        )
        records.append(
            {
                "source": doc["source"],
                "game_id": doc["game_id"],
                "date": doc["date"],
                "home": doc["home"],
                "away": doc["away"],
                "texts": perspectives,
                "model_inputs": {
                    name: embedding_input(perspectives[name], instruction)
                    for name in VARIANTS
                },
            }
        )
        texts.extend(perspectives[name] for name in VARIANTS)
    started = time.perf_counter()
    vectors, lengths = embedder.encode(texts, instruction=instruction)
    torch.cuda.synchronize()
    embedder.metadata.update(
        {
            "encode_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
            "label_rule": label_rule,
            "perspective_options": {"strip": True, "intro": True, "name_teams": True},
        }
    )
    vectors = vectors.reshape(len(documents), len(VARIANTS), -1)
    rows = []
    for index, record in enumerate(records):
        record["tokens"] = dict(
            zip(VARIANTS, lengths[index * 3 : index * 3 + 3], strict=True)
        )
        matrix = vectors[index] @ vectors[index].T
        for left, right in PAIRS:
            rows.append(
                {
                    **{
                        k: v
                        for k, v in record.items()
                        if k not in ("texts", "tokens", "model_inputs")
                    },
                    "pair": f"{left} / {right}",
                    "cosine_similarity": float(
                        matrix[VARIANTS.index(left), VARIANTS.index(right)]
                    ),
                    "tokens_left": record["tokens"][left],
                    "tokens_right": record["tokens"][right],
                }
            )
    return pd.DataFrame(rows), vectors, records


def plot_similarities(results, *, instruction=None):
    """Annotated heatmap, with the complete cosine range retained in the scale."""
    import matplotlib.pyplot as plt

    groups = results.groupby(["source", "game_id"], sort=False)
    labels, values = [], []
    for (source, game_id), rows in groups:
        first = rows.iloc[0]
        labels.append(f"{source} · {game_id}\n{first['away']} @ {first['home']}")
        values.append(
            rows.set_index("pair").loc[
                [f"{a} / {b}" for a, b in PAIRS], "cosine_similarity"
            ]
        )
    values = np.array(values)
    fig, ax = plt.subplots(
        figsize=(10, max(3.5, 0.85 * len(values) + 1.7)), layout="constrained"
    )
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3), [f"{a}\n↔ {b}" for a, b in PAIRS])
    ax.set_yticks(range(len(labels)), labels)
    subtitle = (
        "English instruction: Target Team / Opponent Team roles"
        if instruction
        else "Full texts · no instruction"
    )
    ax.set_title(
        f"Qwen3-Embedding-4B · cosine similarity between perspectives\n{subtitle}",
        pad=15,
    )
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                f"{values[row, col]:.4f}",
                ha="center",
                va="center",
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "pad": 4,
                },
            )
    fig.colorbar(image, ax=ax, label="Cosine similarity (1 = same direction)")
    return fig
