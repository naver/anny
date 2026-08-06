# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

**Anny** is a differentiable human body mesh model in PyTorch that covers all ages (infants to elders) with a common topology and parameter space. Based on MakeHuman assets, it provides full-body, hand, and face models.

## Commands

### Setup
```bash
uv sync --extra examples  # full install with demo dependencies
```

### Testing
```bash
uv run python -m unittest discover     # run all tests
uv run python -m unittest test.test_various  # run a single test file
```

### Documentation
```bash
bash build_doc.bash  # build HTML docs from the jupytext py:percent tutorials in tutorials/*.py
```

## Architecture

### Entry Points

`src/anny/models/__init__.py` exports the public API:
- `Anny(...)` — the full-body model. `Anny` is a class (cf. `anny.SMPLX`); calling
  `anny.Anny(...)` builds a model and `isinstance(model, Anny)` holds for any Anny model.
  Accepts `rig`, `topology`, `pose_parameterization`, `all_phenotypes`, and skinning options.
- `create_fullbody_model(...)` — deprecated legacy full-body factory. It preserves the old
  default rig preset (`rig="default"`) and old full-body defaults; prefer `Anny(...)`.
- `create_hand_model()` / `create_head_model()` — isolated part models

### Core Class Hierarchy

- **`RiggedModelWithLinearBlendShapes`** (`models/rigged_model.py`) — base class; holds template vertices/faces/blend shapes, implements forward kinematics and LBS. The `model_type` parameter (`"tail"` or `"procrustes"`) selects bone orientation strategy internally.
- **`Anny`** (`models/phenotype.py`) — inherits directly from `RiggedModelWithLinearBlendShapes`; adds the 9 phenotype dimensions (gender, age, muscle, weight, height, proportions, race, cupsize, firmness) and computes blend shape coefficients from these semantic scalars. 
- **`SMPL`** / **`SMPLX`** (`models/smpl.py`) — first-class model types that also inherit directly from `RiggedModelWithLinearBlendShapes`; wrap the `smplx` library and follow the same initialization pattern as `Anny`, but accept `betas` + pose parameters instead of phenotype dimensions. Require the optional `smplx` package (`uv sync --extra smpl`).

### Rigs & Topologies

**Rigs** (`anny`, `makehuman`, `cmu_mb`, `game_engine`, `mixamo`, `soma`): bone hierarchies defined as JSON in `src/anny/data/mpfb2/rigs/`. `anny` is the pruned procrustes Anny default, equivalent to the MakeHuman source rig with `notongue`, `noexpression`, and `pruned` modifiers. `default` is a legacy preset accepted only by `create_fullbody_model(...)` and preserves the old full MakeHuman rig defaults. `makehuman` is the full MakeHuman rig with tail/blender orientation and `root_identity_orientation=True`. Rig orientation is part of rig resolution; public constructors do not accept a separate `bone_orientation` argument.

**Topologies** (`default`/`makehuman` ≈16K verts, `smplx` 6890 verts, `soma`): alternative meshes are produced by retopology matrices in `src/anny/data/topology/`. SMPL-X is non-commercial only.

### Key Subsystems

| Subsystem | Location | Purpose |
|-----------|----------|---------|
| Forward kinematics | `utils/kinematics.py` | Tree traversal with parallel propagation fronts |
| Skinning | `skinning/skinning.py` | LBS and dual-quaternion skinning |
| GPU skinning | `skinning/warp_skinning.py` | `warp-lang` accelerated variant (optional) |
| Collision | `utils/collision.py` | Self-intersection detection; warp-accelerated when available |
| Model data | `models/model_data.py` | `ModelData` / `ModelMetadata` dataclasses; bundle template mesh, blend shapes, and rig data; safetensors serialization for caching |
| Model transforms | `models/model_transforms.py` | `ModelData` → `ModelData` operations: retopology (from a mesh, or from linear combinations of template vertices), bone orientation conversion, mesh/skinning cleanups |
| Parameter regression | `parameters_regressor.py` | Iterative pose+shape fitting to a target mesh |
| Anthropometry | `anthropometry.py` | Computes body measurements (height, volume, mass) from mesh |

### Phenotype System

Phenotypes are blended linearly between discrete anchor states defined in `src/anny/data/mpfb2/targets/`. Default mode omits race, cupsize, and firmness; pass `phenotypes="all"` to enable them. Blend shape data is computed at model creation and cached in `~/.cache/anny/`. Set the `ANNY_CACHE_DIR` environment variable to use a different location.

### Pose Parameterization

Five built-in variants: `local-ref` (the `Anny()` default), `local-bone`, `local-bone-world`, `world`, `world-orient`. Selected via the `pose_parameterization` argument to `Anny()`. The deprecated `create_fullbody_model(...)` preserves the old `local-bone` default.

### Optional Dependencies

- `smplx` — required for `SMPL` and `SMPLX` model classes; install via `uv sync --extra smpl`
- `trimesh`, `gradio`, `jsonargparse`, `requests` — needed only for examples and parameter regression tests
