# TMS SPD component explorer

Interactive page for poking at SPD decompositions of the TMS toy model — the
GUI version of the `recon_masked(...)` forward-pass experiments from
`analysis.ipynb`.

## Use

```bash
# 1. export run data (only needed once, or when you add/retrain runs)
uv run python tms_interactive/export_data.py

# 2. open the page (no server needed)
xdg-open tms_interactive/index.html      # or just double-click it
```

`export_data.py` scans `spd/experiments/tms/out/` for any dir with a
`model_*.pth` checkpoint, pulls the target weights, component A/B matrices,
the linear2 bias, and the GateMLP params straight out of the checkpoint (no
wandb, no network), and writes `data.js`. The page loads `data.js` via a
`<script>` tag, so it runs entirely in the browser from `file://`.

## Features

- **run selector** — pick any exported SPD run.
- **input** — type each feature value, or use the `e0…e4` / `0` preset buttons.
- **scale** — slider (0–1, step 0.01) plus a number box for typing any value;
  multiplies the whole input vector.
- **per-layer component tables** (`linear1`, `hidden_layers-0`, `linear2`):
  - `ci` = causal importance `lower_leaky_relu(gate(u · A_c))`, computed from the
    *target model* activations (matching `calc_causal_importances`).
  - per-layer **auto** dropdown: `manual` / `=ci` (pin masks to `clamp01(ci)`) /
    `=1[ci>0]` (binary keep/drop). In an auto mode the masks re-pin live as you
    change input/scale and the sliders are disabled.
  - click the **ci** header to sort components by importance.
  - **mask** slider sets each component's mask manually (value shown alongside).
  - per-layer `=ci` / `=1` / `=0` buttons, plus global `set masks = ci` /
    `reset masks = 1`.
- **forward pass chain** (right → left, like `relu(W₂·Wₕ·W₁·x + b)`) — alternating
  vector blocks (model `m` / recon `r` columns) and weight blocks. Each weight block
  shows the recon `Ŵ` next to the model `W` plus its **faithfulness loss**
  `Σ(Ŵ−W)²/n` (per-element mse and total sse), and the header line reports the
  output MSE and per-layer / total faithfulness — i.e. how well the *active masked
  components* reconstruct the real weights. Matches `calc_faithfulness_loss`.
- **hidden-space quiver** (right column, always visible; when `n_hidden == 2`) —
  feature-direction arrows (columns of the effective weight) for the model's L1,
  the reconstructed L1, and the reconstructed post-hidden map, plus per-stage
  toggles for the current input's resulting point (model/recon, after L1 / after
  hidden).
- **output bar plot** (below the quiver) — per-feature grouped bars comparing the
  model output against the masked-recon output.
- **hidden layer reconstruction** (below the output bars) — the reconstructed `Ŵₕ`
  from the current `hidden_layers-0` masks next to the model's `Wₕ` (≈ identity).

## Notes

- The component reconstruction is `W ≈ Σ_c mask_c · A_{:,c} ⊗ B_{c,:}`; with all
  masks = 1 the recon reproduces the model up to the small `AB ≈ W`
  approximation error (visible in the output MSE).
- The additive `linear2` bias is **not** part of the components — it lives on
  the target model and is added back in the final stage.
- Math (GELU/erf, gate, ci, forward) is validated against the torch
  implementation to 4 decimals.
