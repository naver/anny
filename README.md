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

## Installation

Full installation (depends on warp-lang, which may require some manual work to install):
```bash
pip install anny[warp,examples]
```

Minimal dependency installation (will use more memory with large batch sizes):
```bash
pip install anny
```

Installation from latest sources:
```bash
pip install anny[warp,examples]@git+https://github.com/naver/anny.git
```

## Caching

Anny parses MakeHuman assets and caches pre-computed blend shape data to avoid recomputation on subsequent runs.
The first instantiation of a model can take a few minutes. 
By default the cache is stored in `~/.cache/anny/`. To use a different location, set the `ANNY_CACHE_DIR` environment variable:

```bash
export ANNY_CACHE_DIR=/path/to/cache
```

## Face units

Full-body and head Anny models can optionally load face units:

```python
import torch
import anny

model = anny.Anny(face_units=True)
out = model(face_units={"jawOpen": 0.8, "mouthSmileLeft": 0.4})

values = torch.zeros(1, len(model.face_unit_labels), dtype=model.dtype, device=model.device)
values[:, model.face_unit_labels.index("jawOpen")] = 0.8
out = model(face_units=values)
```

The default is `face_units=False`, which keeps the existing model behavior. Face unit values are normalized in `[0, 1]`; tensor input is ordered by `model.face_unit_labels`.

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

**data/soma**: *Annny* provide a "soma" topology adapted from [SOMA-X](https://github.com/NVlabs/SOMA-X) which is licenced under the [Apache 2.0](https://github.com/NVlabs/SOMA-X/blob/main/LICENSE) license.

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
