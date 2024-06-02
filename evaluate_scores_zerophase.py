# -*- coding: utf-8 -*-
"""Evaluation script for sound quality based on segmental PESQ, STOI and LSC.

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import glob
import os

import joblib
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from pesq import pesq
from progressbar import progressbar as prg
from pystoi import stoi
from scipy import signal


def get_wavname(cfg, basename):
    """Return dirname of wavefile to be evaluated.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        wav_dir (str): dirname of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_name = wav_name.split("-")[0]
    if cfg.demo.gla is False:
        demo_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.demo_dir, "zero", "0")
    else:
        demo_dir = os.path.join(
            cfg.RPU.root_dir, cfg.RPU.demo_dir, "zero", f"{cfg.feature.gla_iter}"
        )
    wav_file = os.path.join(demo_dir, wav_name + ".wav")
    return wav_file


def compute_pesq(cfg, basename):
    """Compute PESQ and wideband PESQ.

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: PESQ (or wideband PESQ).
    """
    eval_wav, _ = sf.read(get_wavname(cfg, basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]
    wav_dir = os.path.join(
        cfg.RPU.root_dir, cfg.RPU.data_dir, cfg.RPU.evalset_dir, cfg.RPU.resample_dir
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return pesq(rate, reference, eval_wav, cfg.demo.pesq_band)


def compute_stoi(cfg, basename):
    """Compute STOI or extended STOI (ESTOI).

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: STOI (or ESTOI).
    """
    eval_wav, _ = sf.read(get_wavname(cfg, basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]
    wav_dir = os.path.join(
        cfg.RPU.root_dir, cfg.RPU.data_dir, cfg.RPU.evalset_dir, cfg.RPU.resample_dir
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return stoi(reference, eval_wav, rate, extended=cfg.demo.stoi_extended)


def compute_lsc(cfg, basename):
    """Compute log-spectral convergence (LSC).

    Args:
        cfg (DictConfig): configuration.

    Returns:
        float: log-spectral convergence.
    """
    eval_wav, _ = sf.read(get_wavname(cfg, basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]
    wav_dir = os.path.join(
        cfg.RPU.root_dir, cfg.RPU.data_dir, cfg.RPU.evalset_dir, cfg.RPU.resample_dir
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
        hop=cfg.feature.hop_length,
        fs=rate,
        mfft=cfg.feature.n_fft,
    )
    ref_spec = stfft.stft(reference).T
    eval_spec = stfft.stft(eval_wav).T
    lsc = np.linalg.norm(np.abs(ref_spec) - np.abs(eval_spec))
    lsc = lsc / np.linalg.norm(np.abs(ref_spec))
    lsc = 20 * np.log10(lsc)
    return lsc


def compensate_phase(phase, win_len, n_frame):
    """Compensate uniform linear phases.

    Args:
        phase (ndarray): phase spectrum.
        win_len (int): length of window.
        n_frame (int): length of frame.

    Returns:
        phase (ndarray): compensated phase spectrum.
    """
    k = np.arange(0, win_len // 2 + 1)
    angle_freq = (2 * np.pi / win_len) * k * (win_len - 1) / 2
    angle_freq = np.tile(np.expand_dims(angle_freq, 0), [n_frame, 1])
    phase = phase + np.angle(np.exp(1j * (angle_freq)))
    return phase


def reconst_waveform(cfg, logabs_path, scaler, device):
    """Reconstruct audio waveform only from the amplitude spectrum.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (tuple): tuple of DNN params (nn.Module).
        logamp_path (str): path to the log-amplitude spectrum.
        scaler (StandardScaler): standard scaler.
        device: device info.

    Returns:
        None.
    """
    logabs_feats = np.load(logabs_path)
    abs_feats = np.exp(scaler.inverse_transform(logabs_feats))
    logabs_feats = np.pad(
        logabs_feats, ((cfg.model.win_range, cfg.model.win_range), (0, 0)), "constant"
    )
    logabs_feats = torch.from_numpy(logabs_feats).float().unsqueeze(0).to(device)
    logabs_feats = logabs_feats.unfold(1, 2 * cfg.model.win_range + 1, 1)
    _, n_frame, _, _ = logabs_feats.shape
    logabs_feats = logabs_feats.reshape(1, n_frame, -1)
    if cfg.demo.gla is True:
        audio = pra.phase.griffin_lim(
            abs_feats,
            hop=cfg.feature.hop_length,
            analysis_window=signal.get_window(
                cfg.feature.window, cfg.feature.win_length
            ),
            n_iter=cfg.feature.gla_iter,
        )
    else:
        stfft = signal.ShortTimeFFT(
            win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
            hop=cfg.feature.hop_length,
            fs=cfg.feature.sample_rate,
            mfft=cfg.feature.n_fft,
            phase_shift=None,
        )
        audio = stfft.istft(abs_feats.T)
    wav_file = get_wavname(cfg, os.path.basename(logabs_path))
    sf.write(wav_file, audio, cfg.feature.sample_rate)


def compute_eval_score(cfg, logabs_list, device):
    """Compute objective scores; PESQ, STOI and LSC.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (tuple): tuple of DNN params (nn.Module).
        logamp_list (list): list of path to the log-amplitude spectrum.
        device: device info.

    Returns:
        score_list (dict): dictionary of objective score lists.
    """
    score_dict = {"pesq": [], "stoi": [], "lsc": []}
    stats_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.stats_dir)
    scaler = joblib.load(os.path.join(stats_dir, "stats.pkl"))
    for logabs_path in prg(logabs_list):
        reconst_waveform(cfg, logabs_path, scaler, device)
        score_dict["pesq"].append(compute_pesq(cfg, os.path.basename(logabs_path)))
        score_dict["stoi"].append(compute_stoi(cfg, os.path.basename(logabs_path)))
        score_dict["lsc"].append(compute_lsc(cfg, os.path.basename(logabs_path)))
    return score_dict


@torch.no_grad()
def main(cfg: DictConfig):
    """Perform model training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = os.path.join(
        cfg.RPU.root_dir, cfg.RPU.feat_dir, cfg.RPU.evalset_dir, cfg.feature.window
    )
    logabs_list = glob.glob(feat_dir + "/*-feats_logabs.npy")
    logabs_list.sort()

    if cfg.demo.gla is False:
        demo_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.demo_dir, "zero", "0")
    else:
        demo_dir = os.path.join(
            cfg.RPU.root_dir, cfg.RPU.demo_dir, "zero", f"{cfg.feature.gla_iter}"
        )
    os.makedirs(demo_dir, exist_ok=True)

    score_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.score_dir)
    os.makedirs(score_dir, exist_ok=True)
    score_dict = compute_eval_score(cfg, logabs_list, device)
    for score_type, score_list in score_dict.items():
        if cfg.demo.gla is False:
            out_filename = os.path.join(score_dir, f"{score_type}_score_zero.txt")
        else:
            out_filename = f"{score_type}_score_{cfg.feature.gla_iter}_zero.txt"
        out_filename = os.path.join(score_dir, out_filename)
        with open(out_filename, mode="w", encoding="utf-8") as file_handler:
            for score in score_list:
                file_handler.write(f"{score}\n")
        score_array = np.array(score_list)
        print(
            f"{score_type}: "
            f"mean={np.mean(score_array):.6f}, "
            f"median={np.median(score_array):.6f}, "
            f"std={np.std(score_array):.6f}, "
            f"max={np.max(score_array):.6f}, "
            f"min={np.min(score_array):.6f}"
        )


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
