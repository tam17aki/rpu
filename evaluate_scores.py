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

from model import get_model


def load_checkpoint(cfg: DictConfig, device):
    """Load checkpoint."""
    model_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.model_dir)
    model_ifreq = get_model(cfg, device)
    model_file = os.path.join(model_dir, cfg.training.model_file + ".ifreq")
    checkpoint = torch.load(model_file)
    model_ifreq.load_state_dict(checkpoint)

    model_grd = get_model(cfg, device)
    model_file = os.path.join(model_dir, cfg.training.model_file + ".grd")
    checkpoint = torch.load(model_file)
    model_grd.load_state_dict(checkpoint)
    return model_ifreq, model_grd


def get_wavname(cfg, basename):
    """Return filename of wavefile to be evaluate."""
    wav_name, _ = os.path.splitext(basename)
    wav_name = wav_name.split("-")[0]
    if cfg.demo.gla is False:
        demo_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.demo_dir, "RPU", "0")
    else:
        demo_dir = os.path.join(
            cfg.RPU.root_dir, cfg.RPU.demo_dir, "RPU", f"{cfg.feature.gla_iter}"
        )
    wav_file = os.path.join(demo_dir, wav_name + ".wav")
    return wav_file


def compute_pesq(cfg, basename):
    """Compute PESQ and wideband PESQ."""
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
    """Compute STOI and extended STOI (optional)."""
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
    """Compute log-spectral convergence (LSC)."""
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
    """Compensate uniform linear phases."""
    k = np.arange(0, win_len // 2 + 1)
    angle_freq = (2 * np.pi / win_len) * k * (win_len - 1) / 2
    angle_freq = np.tile(np.expand_dims(angle_freq, 0), [n_frame, 1])
    phase = phase + np.angle(np.exp(1j * (angle_freq)))
    return phase


def wrap_phase(phase):
    """Wrap phase."""
    return (phase + np.pi) % (2 * np.pi) - np.pi


def compute_rpu(ifreq, grd, n_frame, n_feats):
    """Reconstruct phase by Recurrent Phase Unwrapping (RPU).

    This function performs phase reconstruction via RPU.

    Y. Masuyama, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada,
    Phase reconstruction based on recurrent phase unwrapping with deep neural
    networks, IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2020.
    """
    grd_new = np.zeros_like(grd)
    phase = np.zeros((n_frame, n_feats))
    fd_mat = (
        -np.triu(np.ones((n_feats - 1, n_feats)), 1)
        + np.triu(np.ones((n_feats - 1, n_feats)), 2)
        + np.eye(n_feats - 1, n_feats)
    )
    coef = fd_mat.T @ fd_mat + np.eye(n_feats)
    for tau in range(1, n_frame):
        phase_temp = wrap_phase(phase[tau - 1, :]) + ifreq[tau - 1, :]  # Eq. (10)
        dwp = fd_mat @ phase_temp
        grd_new[tau, :] = dwp + wrap_phase(grd[tau, :] - dwp)  # Eq. (11)
        rhs = phase_temp + fd_mat.T @ grd_new[tau, :]
        phase[tau, :] = np.linalg.solve(coef, rhs)  # Eq. (9)
    return phase


def get_ifreq_grd(model_tuple, logamp):
    """Estimate instantaneous frequency and group delay from log-amplitude."""
    model_ifreq, model_grd = model_tuple  # DNNs
    ifreq = model_ifreq(logamp)
    grd = model_grd(logamp)
    ifreq = ifreq.cpu().detach().numpy().copy().squeeze()
    grd = grd.cpu().detach().numpy().copy().squeeze()
    ifreq = ifreq[:-1, :]
    grd = grd[:, :-1]
    return ifreq, grd


def reconst_waveform(cfg, model_tuple, logabs_path, scaler, device):
    """Reconstruct audio waveform only from the amplitude spectrum."""
    logabs_feats = np.load(logabs_path)
    abs_feats = np.exp(scaler.inverse_transform(logabs_feats))
    logabs_feats = np.pad(
        logabs_feats, ((cfg.model.win_range, cfg.model.win_range), (0, 0)), "constant"
    )
    logabs_feats = torch.from_numpy(logabs_feats).float().unsqueeze(0).to(device)
    logabs_feats = logabs_feats.unfold(1, 2 * cfg.model.win_range + 1, 1)
    _, n_frame, _, _ = logabs_feats.shape
    logabs_feats = logabs_feats.reshape(1, n_frame, -1)
    ifreq, grd = get_ifreq_grd(model_tuple, logabs_feats)
    phase = compute_rpu(ifreq, grd, n_frame, n_feats=abs_feats.shape[-1])
    if cfg.demo.gla is True:
        phase = compensate_phase(phase, cfg.feature.win_length, phase.shape[0])
        audio = pra.phase.griffin_lim(
            abs_feats,
            hop=cfg.feature.hop_length,
            analysis_window=signal.get_window(
                cfg.feature.window, cfg.feature.win_length
            ),
            n_iter=cfg.feature.gla_iter,
            ini=np.exp(1j * phase),
        )
    else:
        reconst_spec = abs_feats * np.exp(1j * phase)
        stfft = signal.ShortTimeFFT(
            win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
            hop=cfg.feature.hop_length,
            fs=cfg.feature.sample_rate,
            mfft=cfg.feature.n_fft,
        )
        audio = stfft.istft(reconst_spec.T)

    wav_file = get_wavname(cfg, os.path.basename(logabs_path))
    sf.write(wav_file, audio, cfg.feature.sample_rate)


def compute_eval_score(cfg, model_tuple, logabs_list, device):
    """Compute objective scores; PESQ, STOI and LSC."""
    score_list = {"pesq": [], "stoi": [], "lsc": []}
    stats_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.stats_dir)
    scaler = joblib.load(os.path.join(stats_dir, "stats.pkl"))
    for logabs_path in prg(logabs_list):
        reconst_waveform(cfg, model_tuple, logabs_path, scaler, device)
        score_list["pesq"].append(compute_pesq(cfg, os.path.basename(logabs_path)))
        score_list["stoi"].append(compute_stoi(cfg, os.path.basename(logabs_path)))
        score_list["lsc"].append(compute_lsc(cfg, os.path.basename(logabs_path)))
    return score_list


@torch.no_grad()
def main(cfg: DictConfig):
    """Perform model training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ifreq, model_grd = load_checkpoint(cfg, device)
    feat_dir = os.path.join(
        cfg.RPU.root_dir, cfg.RPU.feat_dir, cfg.RPU.evalset_dir, cfg.feature.window
    )
    logabs_list = glob.glob(feat_dir + "/*-feats_logabs.npy")
    logabs_list.sort()

    if cfg.demo.gla is False:
        demo_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.demo_dir, "RPU", "0")
    else:
        demo_dir = os.path.join(
            cfg.RPU.root_dir, cfg.RPU.demo_dir, "RPU", f"{cfg.feature.gla_iter}"
        )
    os.makedirs(demo_dir, exist_ok=True)

    score_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.score_dir)
    os.makedirs(score_dir, exist_ok=True)
    score_dict = compute_eval_score(cfg, (model_ifreq, model_grd), logabs_list, device)
    for score_type, score_list in score_dict.items():
        if cfg.demo.gla is False:
            out_filename = os.path.join(score_dir, f"{score_type}_score_0_RPU.txt")
        else:
            out_filename = f"{score_type}_score_{cfg.feature.gla_iter}_RPU.txt"
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
