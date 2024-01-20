# ClipBased-SyntheticImageDetection

[![Github](https://img.shields.io/badge/Github%20page-222222.svg?style=for-the-badge&logo=github)](https://grip-unina.github.io/ClipBased-SyntheticImageDetection/)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2312.00195)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

This is the official repository of the paper:
[Raising the Bar of AI-generated Image Detection with CLIP](https://arxiv.org/abs/2312.00195)

Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva.

## Overview

Aim of this work is to explore the potential of pre-trained vision-language models (VLMs) for universal detection of AI-generated images. We develop a lightweight detection strategy based on CLIP features and study its performance in a wide variety of challenging scenarios.
We find that, unlike previous belief, it is neither necessary nor convenient to use a large domain-specific dataset for training.
On the contrary, by using only a handful of example images from a single generative model, a CLIP-based detector exhibits a surprising generalization ability and high robustness across several different architectures, including recent commercial tools such as Dalle-3, Midjourney v5, and Firefly.
We match the SoTA on in-distribution data, and improve largely above it in terms of generalization to out-of-distribution data (+6% in terms of AUC) and robustness to impaired/laundered data (+13%).

## Code

coming soon

## Bibtex 

```
@misc{cozzolino2023raising,
  author={Davide Cozzolino and Giovanni Poggi and Riccardo Corvi and Matthias Nießner and Luisa Verdoliva},
  title={Raising the Bar of AI-generated Image Detection with CLIP}, 
  journal={arXiv preprint arXiv:2312.00195},
  year={2023},
}
```
