"""Export TMS SPD runs to a JSON blob the interactive page can read.

Reads every SPD run under ``spd/experiments/tms/out/`` (any dir containing a
``model_*.pth`` checkpoint), pulls the target-model weights, the component
A/B matrices, the linear2 bias, and the GateMLP parameters straight out of the
checkpoint (no wandb / no network), and writes ``tms_interactive/data.js``:

    window.TMS_DATA = { runs: { "<run_name>": {...}, ... } };

Loading a ``.js`` file via a <script> tag works from ``file://`` (unlike
``fetch``), so the page works by double-clicking it.

Usage:
    uv run python tms_interactive/export_data.py
"""

import json
import re
from pathlib import Path

import torch

OUT_DIR = Path(__file__).parent.parent / "spd" / "experiments" / "tms" / "out"
DEST = Path(__file__).parent / "data.js"

# Component module name -> target model state_dict prefix for its weight.
LAYER_ORDER = ["linear1", "hidden_layers-0", "linear2"]
MODEL_WEIGHT_KEY = {
    "linear1": "model.linear1.weight",
    "hidden_layers-0": "model.hidden_layers.0.weight",
    "linear2": "model.linear2.weight",
}


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the model_<step>.pth with the largest step, or None."""
    best: tuple[int, Path] | None = None
    for p in run_dir.glob("model_*.pth"):
        m = re.search(r"model_(\d+)\.pth", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, p)
    return best[1] if best else None


def lst(t: torch.Tensor) -> list:
    return t.detach().cpu().float().tolist()


def export_run(run_dir: Path) -> dict | None:
    ckpt_path = latest_checkpoint(run_dir)
    if ckpt_path is None:
        return None  # not an SPD run (e.g. plain TMS training dir)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Discover which component layers are present (keeps this generic).
    present = [name for name in LAYER_ORDER if f"components.{name}.A" in sd]
    if not present:
        return None

    layers = {}
    for name in present:
        A = sd[f"components.{name}.A"]  # (d_in, C)
        B = sd[f"components.{name}.B"]  # (C, d_out)
        layer = {
            "d_in": A.shape[0],
            "d_out": B.shape[1],
            "C": A.shape[1],
            "A": lst(A),  # d_in x C
            "B": lst(B),  # C x d_out
            "W_model": lst(sd[MODEL_WEIGHT_KEY[name]]),  # d_out x d_in
            "gate": {
                "mlp_in": lst(sd[f"gates.{name}.mlp_in"]),  # C x N
                "in_bias": lst(sd[f"gates.{name}.in_bias"]),  # C x N
                "mlp_out": lst(sd[f"gates.{name}.mlp_out"]),  # C x N
                "out_bias": lst(sd[f"gates.{name}.out_bias"]),  # C
            },
        }
        layers[name] = layer

    first = layers[present[0]]
    last = layers[present[-1]]
    return {
        "name": run_dir.name,
        "checkpoint": ckpt_path.name,
        "n_features": first["d_in"],
        "n_hidden": first["d_out"],
        "C": first["C"],
        "n_ci_mlp_neurons": len(first["gate"]["mlp_in"][0]),
        "order": present,
        "out_bias": lst(sd["model.linear2.bias"]),  # final additive bias (5,)
        "layers": layers,
    }


def main() -> None:
    runs = {}
    for run_dir in sorted(OUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        data = export_run(run_dir)
        if data is not None:
            runs[run_dir.name] = data
            print(f"exported {run_dir.name}")
        else:
            print(f"skipped  {run_dir.name} (no SPD checkpoint)")

    payload = {"runs": runs}
    DEST.write_text("window.TMS_DATA = " + json.dumps(payload) + ";\n")
    print(f"\nWrote {len(runs)} runs to {DEST}")


if __name__ == "__main__":
    main()
