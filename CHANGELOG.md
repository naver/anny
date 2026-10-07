# Changelog

All notable changes to this project.

## Unreleased

### Added

- `SMPLH` model class (`anny.models.smpl.SMPLH`) for SMPL+H (SMPL body with MANO hands),
  reading the SMPL+H `.pkl` or the extended 16-beta `.npz` files.
- `phenotype_blendshapes` argument of `Anny(...)`: `"compact"` (default, see below),
  `"original"` for the MakeHuman phenotype blendshapes as loaded, or a positive integer to
  truncate the compact basis to this rank (an approximation). See
  `anny.models.model_transforms.compact_phenotype_blendshapes`.

### Changed

- **Phenotype blendshapes are compacted by default**: replaced by an orthonormal basis of the
  rest states the phenotypes can reach around the template (136 rows instead of 624 for the
  default model). The template and the model outputs are unchanged up to round-off for given
  parameters, but below the phenotype parameters: phenotype `blendshape_labels` become
  `phenotype_basis:NNN`, raw blendshape coefficients are specific to a model configuration
  (topology, rig, phenotype set), and compacted model data is restricted to its phenotype
  set (`ModelMetadata.phenotype_labels`). Pass `phenotype_blendshapes="original"` for the
  previous behavior. Cached model data is regenerated automatically (data version bump).

### Changed

- Development versions count towards the next minor release (`0.7.0.devN` after `v0.6.1`),
  and towards the next patch release only on release branches (`0.6.x`, `release-0.6`):
  setuptools-scm's `semver-pep440-release-branch` scheme (setuptools-scm >= 10 to build).

### Fixed

- Models retopologized onto another mesh (e.g. `SMPL(topology="anny")`) mixed float32 and
  float64 tensors, so that their forward pass failed until an explicit `.to(dtype=...)`.
- `SMPLX` joints now follow the expression blendshapes, as in the `smplx` reference
  implementation.
- The weight, muscle and proportions tables of `SimpleShapeDistribution` were indexed
  by age in years but looked up with the anny `age` phenotype (in [0, 1]), so they were
  always read between their newborn and 1-year entries. Their anchors are now expressed
  in anny age, like the height table. This changes the weight and muscle values sampled
  by `SimpleShapeDistribution.sample()`, and the phenotype prior of `AnnyInverter`.
- `SMPL`, `SMPLH` and `SMPLX` produced NaN gradients for their shape and pose parameters
  (`betas` and every non-root pose: body, hands, jaw, eyes, expression; only `global_orient`
  and `transl` were unaffected). Their synthetic bone tails are aligned with the y axis, so
  the rotation axis of these bones was computed as 0/0; the forward pass already replaced it
  by the degenerate rotation, but the NaN still propagated in the backward pass.

## v0.6.1 - 2026-09-28

### Fixed

- Alternative topologies obtained by retopology (e.g. `smplx`) are now left/right
  symmetric: target vertices on the midline are projected onto the intersection of the
  source mesh with its symmetry plane, rather than onto the closest surface point, which
  was ambiguous over concave midline regions. Cached model data is regenerated
  automatically (data version bump).
- Phenotype parameters lying exactly on an anchor point, in particular internal ones
  such as the default `weight=0.5`, now get a consistent non-zero gradient. The
  interpolation weight was previously clamped with `torch.clamp`, whose gradient at the
  bounds is inconsistent across torch versions and could vanish there.
- SOMA assets are pinned to the Hugging Face revision matching `py-soma-x==0.1.0`
  (`anny.models.soma.SOMA_ASSETS_REVISION`), since later assets are incompatible.
- `KeypointsRegressor` now loads its keypoint data directly on the model device, fixing
  device mismatches with models not on the CPU.

## v0.6 - 2026-08-06

Highlights:

- **Facial actions.** Facial expressions can now be driven by the MakeHuman face
  units, e.g. `Anny(facial_actions="all")` then
  `model(facial_actions={"jawOpen": 0.8})`. See the dedicated tutorial.
- **New `anny` rig**, now the default. Its bone orientations are much more stable
  across shape changes, and its root bone is aligned with the pelvis.
- **Baked rig orientations.** Rest bone orientations are anchored to the mesh: each bone
  follows the vertices it skins, which is what keeps it stable across shapes. The
  per-bone cross-covariance matrices behind that fit are linear in the blend shape
  coefficients, so they are precomputed and shipped (`data/cached/{anny,soma}.pth`) and
  the orientations are recovered at runtime by a single batched Procrustes alignment.
- **New `Anny()` API.** `create_fullbody_model()` and the part-model factories are
  deprecated in favor of a single `Anny(...)` constructor.
- **Improved SOMA compatibility**, including SOMA with alternative topologies.
- **`torch.compile` support**, for significant speedups.

### Added

- Facial actions, exposed through the `facial_actions` argument of `Anny(...)` and of
  the model call.
- The `anny` rig, a pruned rig with shape-stable bone orientations obtained by
  Procrustes alignment.
- A `"cached"` bone orientation mode, now used by the `anny` and `soma` rigs, backed by
  precomputed orientation data generated by `scripts/precompute_rig_caches.py`.
- Cross-model pose transfer, via `anny.utils.pose.transfer_pose_parameters`
  (see `tutorials/pose_transfer.py`).
- Compatibility with `torch.compile` and `torch.export` (TorchDynamo).
- Bone subtree selection for the `anny` rig.

### Changed

- **`Anny()` now defaults to the `anny` rig with `pose_parameterization="local-ref"`**
  instead of `"local-bone"`. Pose parameters are therefore not interchangeable with
  those of previous versions; the deprecated `create_fullbody_model()` keeps the old
  defaults.
- Improved SOMA support: bone orientation blend shapes, an orientation refiner, and
  faces reproducing those of the `soma` package.
- The default skinning weights replace the raw MakeHuman ones, with symmetrization and
  island cleanup now baked in offline rather than applied at model creation.
- The model class hierarchy was flattened to single inheritance; `Anny`, `SMPL` and
  `SMPLX` all derive directly from `RiggedModelWithLinearBlendShapes`.
- Phenotype targets moved out of the `mpfb2` directory.
- Tutorials converted from Jupyter notebooks to jupytext `py:percent` sources.
- Dynamic version numbers.
- Improved regressor of Anny parameters (`ParametersRegressor` renamed into `AnnyInverter`).

### Deprecated

- `create_fullbody_model()`, `create_hand_model()` and `create_head_model()` — use
  `Anny(...)` with the corresponding topology instead.
- Passing `local_changes` as a bool; pass `"none"`, `"default"`, `"all"`, or a sequence
  of local change labels.

### Removed

- The `default_no_toes` and `legacy_no_toes` rig presets, and the `symmetric` rig
  modifier.
- The skinning-weight cleanup arguments, the weights being pre-cleaned now.

### Fixed

- A degeneracy in the separating-axis self-intersection test.
- Phenotype broadcasting to the pose batch size.
- Compatibility with torch < 2.5, and a missing `chumpy` dependency for SMPL.

## v0.5 - 2026-06-03

Code refactoring (one can now use the `anny.Anny` syntax). Support for the
[soma](https://github.com/NVlabs/SOMA-X) rig and topology. SMPL-X wrapper with `anny`
topology support.

*There was no v0.4 release: `0.4a` was the in-development version number for this cycle,
renamed to 0.5 before release.*

## v0.3.1 - 2026-02-13

Fixed the rest joint position computation.

## v0.3 - 2026-02-04

`smplx` topology available for interoperability with
[SMPL-X](https://smpl-x.is.tue.mpg.de/) (non-commercial use only). Nipple blend shapes
excluded from the default settings (use `local_changes="all"` for backward
compatibility).

## v0.2 - 2025-11-21

Support for different mesh topologies.

## v0.1 - 2025-11-05

Initial release.
