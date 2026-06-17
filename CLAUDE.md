# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Anny** is a differentiable human body mesh model in PyTorch that covers all ages (infants to elders) with a common topology and parameter space. Based on MakeHuman assets, it provides full-body, hand, and face models.

## Commands

### Setup
```bash
uv sync --extra warp --extra examples  # full install with GPU acceleration and demo dependencies
```

### Testing
```bash
uv run python -m unittest discover     # run all tests
uv run python -m unittest test.test_full_model  # run a single test file
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
- `create_fullbody_model(...)` — the factory `Anny(...)` delegates to (identical arguments);
  still public and equivalent.
- `create_hand_model()` / `create_head_model()` — isolated part models

### Core Class Hierarchy

- **`RiggedModelWithLinearBlendShapes`** (`models/rigged_model.py`) — base class; holds template vertices/faces/blend shapes, implements forward kinematics and LBS
- **`Anny`** (`models/phenotype.py`) — common base of every phenotype model (full-body, hand, head); adds the 9 phenotype dimensions (gender, age, muscle, weight, height, proportions, race, cupsize, firmness) and computes blend shape coefficients from these semantic scalars. Its `_AnnyMeta` metaclass also makes `Anny(...)` itself build a full-body model via `create_fullbody_model`.
- **`RiggedModelWithPhenotypeParameters`** / **`RiggedModelWithProcrustesAndPhenotypeParameters`** — concrete pose+shape models (tail- vs procrustes-based bone orientation); `model_from_model_data` selects between them.

### Rigs & Topologies

**Rigs** (`default`, `cmu_mb`, `game_engine`, `mixamo`, `soma`): bone hierarchies defined as YAML in `src/anny/data/mpfb2/rigs/`. The `default` rig supports modifiers via underscore suffixes (e.g. `default_no_toes`).

**Topologies** (`default`/`makehuman` ≈16K verts, `smplx` 6890 verts, `soma`): alternative meshes are produced by retopology matrices in `src/anny/data/topology/`. SMPL-X is non-commercial only.

### Key Subsystems

| Subsystem | Location | Purpose |
|-----------|----------|---------|
| Forward kinematics | `utils/kinematics.py` | Tree traversal with parallel propagation fronts |
| Skinning | `skinning/skinning.py` | LBS and dual-quaternion skinning |
| GPU skinning | `skinning/warp_skinning.py` | `warp-lang` accelerated variant (optional) |
| Collision | `utils/collision.py` | Self-intersection detection; warp-accelerated when available |
| Parameter regression | `parameters_regressor.py` | Iterative pose+shape fitting to a target mesh |
| Anthropometry | `anthropometry.py` | Computes body measurements (height, volume, mass) from mesh |

### Phenotype System

Phenotypes are blended linearly between discrete anchor states defined in `src/anny/data/mpfb2/targets/`. Default mode omits race, cupsize, and firmness; pass `all_phenotypes=True` to enable them. Blend shape data is computed at model creation and cached in `~/.cache/anny/`. Set the `ANNY_CACHE_DIR` environment variable to use a different location.

### Pose Parameterization

Three built-in variants: `local-bone` (default), `root_relative_world`, `root_relative`. Selected via `pose_parameterization` argument to `Anny()`.

### Optional Dependencies

- `warp-lang` — enables GPU-accelerated skinning and collision detection; code degrades gracefully without it
- `trimesh`, `gradio`, `jsonargparse`, `requests` — needed only for examples and parameter regression tests
