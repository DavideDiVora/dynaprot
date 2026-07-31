# dynaprot

**Learning residue-level protein dynamics with multiscale Gaussians**

Mihir Bafna, Bowen Jing, Bonnie Berger — CSAIL, MIT

DynaProt predicts descriptors of protein dynamics from a single static structure, treating an
ensemble as a multivariate Gaussian over Cα coordinates:

- **DynaProt-M** — per-residue **3×3 marginal covariances**: anisotropic "Gaussian blobs"
  capturing local flexibility and its direction. RMSF is `sqrt(Tr(Σᵢ))`. **955K params.**
- **DynaProt-J** — an **N×N coupling matrix**: how residue *i*'s motion relates to residue *j*'s.

Composing the two gives an approximate **3N×3N** joint covariance, from which ensembles are
sampled in a fraction of a second — ~70,000× faster than generative ensemble predictors, at
2.86M total parameters.

## Install

```bash
git clone https://github.com/MihirBafna/dynaprot.git
cd dynaprot
uv sync
```

Then `uv run dynaprot ...`, or activate with `source .venv/bin/activate`. `pip install -e .`
also works. Extras: `nma` (ProDy baseline), `preprocess`, `viz`, `train`, `dev`.

```bash
uv sync --extra nma --extra dev      # pip: pip install -e ".[nma,dev]"
```

The OpenFold code DynaProt needs is vendored in `dynaprot/utils/openfold/` (Apache 2.0).

## Predict

```bash
dynaprot predict my_protein.pdb -o out/ --num-samples 250
```

| File | Contents |
|---|---|
| `prediction.npz` | the predicted arrays (below) |
| `rmsf.csv` | per-residue RMSF |
| `rmsf_bfactor.pdb` | input structure with RMSF in the B-factor column (`spectrum b` in PyMOL) |
| `ensemble.pdb` / `.xtc` | sampled conformational ensemble |
| `metadata.json` | checkpoint SHA-256s, sequence, seed, timings |

`prediction.npz` (`N` residues, Å and Å²):

| Array | Shape | Meaning |
|---|---|---|
| `marginal_covariances` | `(N,3,3)` | per-residue covariance, global frame |
| `rmsf` | `(N,)` | `sqrt(Tr(Σᵢ))` |
| `coupling` | `(N,N)` | residue–residue coupling covariance |
| `correlation` | `(N,N)` | unit-diagonal form of `coupling` |
| `ca_coords` | `(N,3)` | input Cα coordinates — the Gaussian means |
| `residue_index` | `(N,)` | numbering from the input PDB |
| `sequence` | — | one-letter sequence |

```python
from dynaprot.inference import load_prediction

pred = load_prediction("out/")   # sequence comes back as a str
pred["rmsf"].shape               # (431,)
```

### Choosing outputs

`--outputs` selects descriptors; only the models needed are loaded and run.

```bash
dynaprot predict p.pdb -o out/ --outputs rmsf              # DynaProt-M only
dynaprot predict p.pdb -o out/ --outputs marginals coupling
```

| `--outputs` | Needs | Writes |
|---|---|---|
| `rmsf` | M | `rmsf` in the npz, `rmsf.csv`, `rmsf_bfactor.pdb` |
| `marginals` | M | `marginal_covariances` in the npz |
| `coupling` | J | `coupling` + `correlation` in the npz |
| `ensemble` | M + J | `ensemble.pdb` / `.xtc` |

RMSF derives from the marginals, so requesting it still computes them and the prediction object
keeps them.

Other flags: `--chain A` (required for multi-chain PDBs), `--num-samples`, `--format both`,
`--all-atom` (PULCHRA must be on `PATH`), `--device`, `--seed`.

### From Python

```python
from dynaprot.inference import DynaProtPredictor

predictor  = DynaProtPredictor.from_pretrained(device="cuda:0")
prediction = predictor.predict("my_protein.pdb", chain_id="A", outputs="rmsf")

prediction.rmsf                  # (N,)     Å
prediction.marginal_covariances  # (N,3,3)  global frame, Å²
prediction.coupling              # (N,N)
prediction.joint_covariance      # (3N,3N)  composed via Eq. 5
prediction.sample_ensemble(250)  # (250,N,3)
```

### Visualizing the marginals

`dynaprot visualize` renders the marginals as PyMOL ellipsoids — each the one-sigma surface of a
residue's predicted positional distribution (the paper's Figure 4).

```bash
dynaprot visualize out/ --structure my_protein.pdb   # writes out/ellipsoids.py
pymol my_protein.pdb out/ellipsoids.py
```

The generated script needs nothing installed here. With `pymol` importable, `--session` and
`--image` save a `.pse` or a ray-traced PNG directly.

## Checkpoints

Hosted at [`mihirbafna/dynaprot`](https://huggingface.co/mihirbafna/dynaprot); they download on
first use. For local copies, set `DYNAPROT_CKPT_DIR` to a directory of checkpoint folders or
pass `--marginal-ckpt` / `--joint-ckpt`.

| Name | Model |
|---|---|
| `dynaprot-M` | DynaProt-M, ATLAS/AlphaFlow split, 955K params |
| `dynaprot-J` | DynaProt-J, same split, 1.9M params |

Each folder holds `model.ckpt` plus the exact `model_config.yaml` and `data_config.yaml` it must
be loaded with — these checkpoints do not carry their hyperparameters internally. Split-specific
and ablation checkpoints are retained by the authors but not published.

## Training

Trains on [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) MD trajectories (3 replicates × 100 ns per
protein, Cα only). Build the labels once, then train:

```bash
export DYNAPROT_DATA_DIR=/path/to/datasets

python -m dynaprot.data.preprocessing.preprocess_atlas \
    --inpath  $DYNAPROT_DATA_DIR/atlas \
    --outpath $DYNAPROT_DATA_DIR/atlas_dynamics_labels

dynaprot-train \
    --data_config  configs/data/atlas_config_randomcropping.yaml \
    --model_config configs/model/dynaprot_marginal_nosinusoidal_1layer.yaml \
    --name my-run --logger none
```

See `configs/README.md` for which config goes with which checkpoint.

> The released checkpoints were trained at a constant learning rate; the warmup+cosine schedule
> in the configs is opt-in via `train_params.use_lr_schedule: true`.

## Citation

```bibtex
@inproceedings{bafna2026dynaprot,
  title     = {Learning residue level protein dynamics with multiscale Gaussians},
  author    = {Bafna, Mihir and Jing, Bowen and Berger, Bonnie},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Vendored OpenFold code in `dynaprot/utils/openfold/` remains under
Apache 2.0.
