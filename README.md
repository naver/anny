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

## Installation

Full installation:
```bash
pip install anny[warp,examples]@git+ssh:git@github.com:naver/anny.git
```

Minimal dependency installation:
```bash
pip install anny@git+ssh:git@github.com:naver/anny.git
```

## Tutorials

To get started with Anny, you can have a look at the different notebooks in the `tutorials` repository.

## Interactive demo

We provide a simple Gradio demo enabling to interact with the model easily:
```bash
python -m anny.examples.interactive_demo
```

<img src="docs/figures/interactive_demo.jpg" alt="Interactive demo" style="display:block;max-width:100%;max-height:24em;margin:auto"/>


## License

The code of Anny, Copyright (c) 2025 NAVER Corp., is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

**data/mpfb2**: *Anny* relies on [MakeHuman](https://static.makehumancommunity.org/) assets adapted from [MPFB2](https://github.com/makehumancommunity/mpfb2/) that are licensed under the [CC0 1.0 Universal](src/anny/data/mpfb2/LICENSE.md) License.
