<h1 style="text-align: center;">Anny Body</h1>

<img src="docs/figures/anny_teaser.jpg" alt="Anny" style="display:block;max-width:100%;max-height:24em;margin:auto"/>

Anny is a differentiable human body mesh model written in PyTorch.
Anny models a large variety of human body shapes, from infants to elders, using a common topology and parameter space.

[![ArXiv](https://img.shields.io/badge/arXiv-2511.03589-33cb56)](https://arxiv.org/abs/2511.03589)
[![Demo](https://img.shields.io/badge/Demo-33cb56)](http://anny-demo.europe.naverlabs.com/)
[![Blogpost](https://img.shields.io/badge/Blogpost-33cb56)](https://europe.naverlabs.com/blog/anny-a-free-to-use-3d-human-parametric-model-for-all-ages/)

### Features
- Anny is based on the tremendous work of the [MakeHuman](https://static.makehumancommunity.org/) community, which offers plenty of opportunities for extensions.
- We provide both full body and part-specific models for hands and faces.
- Anny is open-source and free.

### News
 - **2026-06-03**: v0.5: code refactoring (one can now use "anny.Anny" syntax). Support for ["soma"](https://github.com/NVlabs/SOMA-X) rig and topology. SMPLX wrapper with "anny" topology support.
 - **2026-02-04**: v0.3: "smplx" topology available for interoperability with [SMPL-X](https://smpl-x.is.tue.mpg.de/) (non-commercial use only). Nipple blend shapes excluded from default settings (use `local_changes="all"` for backward compatibility).
 - **2025-11-21**: v0.2: support for different mesh topologies.
 - **2025-11-05**: v0.1: initial release.

### Installation

```bash
pip install anny[warp,smpl,examples] # Full install (non-free dependencies).
pip install anny[warp,examples] # Free install.
pip install anny # Minimal install (use more memory for large batch sizes).
# Note that the free install may download non-commercial only assets when needed.
pip install anny[warp,examples]@git+https://github.com/naver/anny.git # latest sources.
```

### Quickstart example
```python
import torch, anny, trimesh
model = anny.Anny(local_changes=True, facial_actions=True).to(dtype=torch.float32)
# The model accept both dictionnary and stacked tensor inputs.
# Skeletal rig pose parameters (see model.bone_labels).
pose_parameters = torch.eye(4)[None, None].repeat(1, model.bone_count, 1, 1)
# High-level shape parameters (within [0,1], see model.phenotype_labels):
phenotype_kwargs = {key : 0.5 for key in model.phenotype_labels}
# Local shape changes (within [-1,1], see model.local_change_labels):
local_changes = {'stomach-pregnant-incr': 1.}
# Facial expression changes (within [0,1], see model.facial_action_labels):
facial_actions = {"jawOpen": 0.8, "mouthSmileLeft": 0.4}
# Export default mesh output
output = model(
      pose_parameters=pose_parameters,
      phenotype_kwargs=phenotype_kwargs,
      local_changes_kwargs=local_changes,
      facial_actions=facial_actions
      )
trimesh.Trimesh(vertices = output["vertices"].squeeze(dim=0).numpy(), faces=model.faces).export("anny_output.ply")
```

### Default `anny` rig

By default, `anny.Anny()` uses the compact `anny` rig with 104 bones and Procrustes bone orientations. This is the recommended default for most full-body use cases: it keeps the main body, hand, and head articulation while removing facial expression, eye, tongue, and other zero-weight/pruned bones that are present in the full MakeHuman rig. For comparison, `Anny(rig="makehuman")` exposes the full 163-bone MakeHuman rig with the old blender/root-identity orientation. Choose `rig="anny"` for a smaller, stable default skeleton; choose `rig="makehuman"` if you need exact compatibility with old models or direct access to the removed face/tongue/eye bones. Facial action blendshapes remain available separately with `facial_actions=True`.

### Default `anny` topology

By default, `anny.Anny()` uses the `anny` topology: a MakeHuman-derived full-body mesh with Anny's minor nudity-related mesh edits, unattached vertices removed, and triangular faces. This is the recommended topology for new full-body models because every output vertex is referenced by the mesh connectivity and the triangulated faces work directly with downstream tools such as anthropometry and most mesh processing libraries. Use `topology="anny-quads"` if you need the original quad faces, `topology="anny-full"` if you need to keep unattached vertices and disable the nudity-related mesh edits, or `topology="makehuman"` for the unedited quad MakeHuman body mesh convention. Alternative retopologies such as `smplx`, `smpl`, and `soma` are available when interoperability with those ecosystems is more important than using Anny's native mesh.

### Migration from legacy defaults

`anny.Anny()` now defaults to the new pruned procrustes Anny model (`rig="anny"`, `topology="anny"`), whereas the legacy full-body defaults were exposed through `create_fullbody_model(rig="default", topology="default", bone_orientation="blender-rootidentity", remove_unattached_vertices=True, triangulate_faces=False)`. For exact legacy behavior, keep using `anny.create_fullbody_model(...)` while migrating; for new `anny.Anny(...)` calls, replace `rig="default"` with `rig="makehuman-symmetric-blender-rootidentity"` when you need the old full MakeHuman rig and orientation, or with `rig="anny"` for the new default. Replace `topology="default"` with `topology="anny"`, and encode the old mesh flags in the topology string: add `-quads` for `triangulate_faces=False` and add `-full` for `remove_unattached_vertices=False` (for example, old `topology="default", triangulate_faces=False` maps to `topology="anny-quads"`).

## Caching

Anny parses MakeHuman assets and caches pre-computed blend shape data to avoid recomputation on subsequent runs.
The first instantiation of a model can take a few minutes. 
By default the cache is stored in `~/.cache/anny/`. To use a different location, set the `ANNY_CACHE_DIR` environment variable:

```bash
export ANNY_CACHE_DIR=/path/to/cache
```
## Tutorials

To get started with Anny, you can have a look at the different tutorials in the `tutorials` directory:
- [Shape parameterization](https://naver.github.io/anny/build/shape_parameterization.html)
- [Pose parameterization](https://naver.github.io/anny/build/pose_parameterization.html)
- [Texture coordinates](https://naver.github.io/anny/build/texture.html)
- [Alternative models](https://naver.github.io/anny/build/alternative_models.html)

## Interactive demo

We provide a simple Gradio demo enabling to interact with the model easily:
```bash
python -m anny.examples.interactive_demo
```

<img src="docs/figures/interactive_demo.jpg" alt="Interactive demo" style="display:block;max-width:100%;max-height:24em;margin:auto"/>


## License

The code of Anny, Copyright (c) 2025 NAVER Corp., is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

**data/mpfb2**: *Anny* relies on [MakeHuman](https://static.makehumancommunity.org/) assets adapted from [MPFB2](https://github.com/makehumancommunity/mpfb2/) that are licensed under the [CC0 1.0 Universal](src/anny/data/mpfb2/LICENSE.md) License.

**data/faceunits01**: Facial actions of *Anny* rely on [Face Units asset pack](https://static.makehumancommunity.org/assets/assetpacks/index.html#functional-asset-packs) by Mika Suominen, licensed under the [CC0 1.0 Universal](src/anny/data/mpfb2/LICENSE.md) License.

**data/soma**: *Anny* provide a "soma" topology adapted from [SOMA-X](https://github.com/NVlabs/SOMA-X) which is licenced under the [Apache 2.0](https://github.com/NVlabs/SOMA-X/blob/main/LICENSE) license.

**smplx**: A "smplx" topology can be downloaded for non-commercial use only, allowing interoperability with [SMPL-X](https://smpl-x.is.tue.mpg.de/). See LICENSE.txt and NOTICE.txt files in http://download.europe.naverlabs.com/humans/Anny/noncommercial.zip for more information.

## Citation

```
@misc{br\’egier2025humanmeshmodelinganny,
      title={Human Mesh Modeling for Anny Body}, 
      author={Romain Br\’egier and Gu\’enol\’e Fiche and Laura Bravo-S\’anchez and Thomas Lucas and Matthieu Armando and Philippe Weinzaepfel and Gr\’egory Rogez and Fabien Baradel},
      year={2025},
      eprint={2511.03589},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.03589}, 
}
```
