# Unofficial implementation of phase reconstruction method based on recurrent phase unwrapping with DNNs

This repository provides an unofficial implementation of phase reconstruction based on RPU with DNNs [1].

Additionally, a weighted RPU is also implemented [2].

## Licence
MIT licence.

Copyright (C) 2024 Akira Tamamori

## Dependencies
We tested the implemention on Ubuntu 22.04. The verion of Python was `3.10.12`. The following modules are required:

- hydra-core
- joblib
- librosa
- numpy
- progressbar2
- pydub
- pypesq
- pyroomacoustics
- pystoi
- scikit-learn
- scipy
- soundfile
- torch


## Datasets
You need to prepare the following two datasets from [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).

   - basic5000: for training

   - onomatopee300: for evaluation

## Recipes

1. Download the two datasets. Put those in /root_dir/trainset_dir and /root_dir/evalset_dir/, respectively.

2. Modify `config.yaml` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths according to your environment.

3. Run `preprocess.py`. It performs preprocessing steps.

4. Run `training.py`. It performs model training.

5. Run `evaluate_scores.py`. It generates reconstructed audio data and computes objective scores (PESQ, STOI, LSC). In this script, the function `compute_rpu` implements RPU and weighted RPU.

6. Run `evaluate_scores_zerophase.py`. It also generates reconstructed audio data and computes objective scores (PESQ, STOI, LSC), where phase spectrum is assumed to be zero (zero-phase).

7. Run `evaluate_scores_randomphase.py`. It also generates reconstructed audio data and computes objective scores (PESQ, STOI, LSC), where phase spectrum is assumed to be sampled uniformly between $-\pi$ and $\pi$.

8. Run `plot_boxplot.py`. It plots boxplot of objective scores.

## References

[1] Y. Masuyama, K. Yatabe, Y. Koizumi, Y. Oikawa and N. Harada, "Phase reconstruction based on recurrent phase unwrapping with deep neural networks," IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2020.

[2] N. B. Thien, Y. Wakabayashi, K. Iwai and T. Nishiura, "Inter-Frequency Phase Difference for Phase Reconstruction Using Deep Neural Networks and Maximum Likelihood," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 1667-1680, 2023, doi: 10.1109/TASLP.2023.3268577.
