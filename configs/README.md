# Configs

Two YAMLs are composed at runtime: a **data** config (dataset location, split list, cropping)
and a **model** config (architecture, losses, training params). `dynaprot.train` attaches the former
to the latter as `cfg["data_config"]`.

Paths use `${DYNAPROT_DATA_DIR}`, expanded by `dynaprot.config.load_config`. Split lists are
resolved against the copies shipped inside the package, so they work from any directory.

## Which config goes with which checkpoint

`DynaProt.__init__` never called `save_hyperparameters()`, so **a checkpoint cannot be loaded
without its exact model config** — and the checkpoint filename does not imply the config (e.g.
`dynaprot-J` is named `joint2head6` but needs `dynaprot_joint_pairattention_bigger.yaml`). The
released checkpoint folders each carry a verbatim copy of both YAMLs; this table records the
same mapping.

| Checkpoint | Model config | Data config | Paper result |
|---|---|---|---|
| `dynaprot-M` | `dynaprot_marginal_nosinusoidal_1layer.yaml` | `atlas_config_randomcropping.yaml` | marginal Gaussians, RMSF, ensembles |
| `dynaprot-J` | `dynaprot_joint_pairattention_bigger.yaml` | `atlas_config_randomcropping_256res.yaml` | residue coupling, joint reconstruction |
| `dynaprot-M-flexpert` | `dynaprot_marginal_nosinusoidal_2layer.yaml` | `atlas_config_flexpertsplits.yaml` | RMSF *r* vs FlexPert-3D |
| `dynaprot-M-ablate-mse` | `dynaprot_marginal_ablatemseloss.yaml` | `atlas_config_randomcropping.yaml` | ablation: no log-Frobenius |
| `dynaprot-MJ` | `dynaprot_M+J_testing.yaml` | `atlas_config_randomcropping.yaml` | jointly trained M+J |

The `256res` and `512res` data configs differ only in `max_num_residues`, which affects training
crops; at test time `DynaProtDataset` returns unpadded, uncropped proteins, so the two are
interchangeable for evaluation.

## Model config keys worth knowing

- `train_params.out_type` selects the readout — `marginal`, `joint_pairattention`, or
  `both_marginal+joint`. Substring matching drives the branches in `architecture.py`.
- `eval_params.loss_weights` gates every loss term: `null` means "don't compute", `0.0` means
  "compute and log but don't optimise". `log_frob_norm: 1.0` is the paper's objective.
- `train_params.use_lr_schedule` (default `false`) opts into the warmup+cosine schedule. The
  released checkpoints were trained without it — see the note in the top-level README.
- `model_params.use_blocks` (residual IPA blocks) is **not** exercised by any released model and
  the code path is known to be broken. One config misspells it as `use_block`, so its
  `prelayernorm` setting never took effect either.

## Known issue

`atlas_config_flexpertsplits.yaml` points `protein_chains_path` at `flexpert_chains.npy`, which
does not exist — only `flexpert_chains_{train,val,test}.npy` do. This works for any real split
because `DynaProtDataset` appends `_{split}`, but `split="all"` will fail.
