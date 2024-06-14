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
from concurrent.futures import ProcessPoolExecutor

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
from scipy.linalg import solve_banded
from scipy.sparse import csr_array
from torch.multiprocessing import set_start_method

from model import get_model


def load_checkpoint(cfg: DictConfig, device):
    """Load checkpoint.

    Args:
        cfg (DictConfig): configuration.
        device: device info.

    Returns:
        model_ifreq (torch.nn.Module): DNNs to estimate instantaneous frequency.
        model_grd   (torch.nn.Module): DNNs to estimate group delay.
    """
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


def get_wavdir(cfg):
    """Return dirname of wavefile to be evaluated.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        wav_dir (str): dirname of wavefile.
    """
    if cfg.demo.gla is False:
        if cfg.demo.weighted_rpu is False:
            wav_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.demo_dir, "RPU", "0")
        else:
            wav_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.demo_dir, "wRPU", "0")
    else:
        if cfg.demo.weighted_rpu is False:
            wav_dir = os.path.join(
                cfg.RPU.root_dir, cfg.RPU.demo_dir, "RPU", f"{cfg.feature.gla_iter}"
            )
        else:
            wav_dir = os.path.join(
                cfg.RPU.root_dir, cfg.RPU.demo_dir, "wRPU", f"{cfg.feature.gla_iter}"
            )
    return wav_dir


def get_wavname(cfg, basename):
    """Return filename of wavefile to be evaluated.

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        wav_file (str): filename of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_name = wav_name.split("-")[0]
    wav_dir = get_wavdir(cfg)
    wav_file = os.path.join(wav_dir, wav_name + ".wav")
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
        basename (str): basename of wavefile for evaluation.

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


def wrap_phase(phase):
    """Compute wrapped phase.

    Args:
        phase (ndarray): phase spectrum.

    Returns:
        wrapped phase (ndarray).
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def get_band_coef(matrix):
    """Return band tridiagonal elements of coef matrix.

    Args:
        matrix (ndarray): band tridiagonal matrix.

    Returns:
        band_elem (ndarray): band tridiagonal elements (upper, diag, and lower).
    """
    upper = np.diag(matrix, 1)
    upper = np.concatenate((np.array([0]), upper))
    lower = np.diag(matrix, -1)
    lower = np.concatenate((lower, np.array([0])))
    band_elem = np.concatenate(
        (upper.reshape(1, -1), np.diag(matrix).reshape(1, -1), lower.reshape(1, -1))
    )
    return band_elem


def compute_rpu(ifreq, grd, amplitude, weighted_rpu=False, weight_power=5):
    """Reconstruct phase by Recurrent Phase Unwrapping (RPU).

    This function performs phase reconstruction via RPU.

    Y. Masuyama, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada,
    Phase reconstruction based on recurrent phase unwrapping with deep neural
    networks, IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2020.

    For weighted RPU, see:

    N. B. Thien, Y. Wakabayashi, K. Iwai and T. Nishiura,
    Inter-Frequency Phase Difference for Phase Reconstruction Using Deep Neural
    Networks and Maximum Likelihood, in IEEE/ACM Transactions on Audio,
    Speech, and Language Processing, vol. 31, pp. 1667-1680, 2023.

    Args:
        ifreq (ndarray): instantaneous frequency. [T-1, K]
        grd   (ndarray): group delay. [T, K-1]
        amplitude (ndarray): amplitude spectrum. [T, K]
        weighted_rpu (bool): flag to apply weighted RPU.
        weight_power (int): power to weight.

    Returns:
        phase (ndarray): reconstructed phase. [T, K]
    """
    n_frame, n_feats = amplitude.shape
    phase = np.zeros_like(amplitude)
    fd_mat = (  # frequency-directional differential operator (matrix)
        -np.triu(np.ones((n_feats - 1, n_feats)), 1)
        + np.triu(np.ones((n_feats - 1, n_feats)), 2)
        + np.eye(n_feats - 1, n_feats)
    )
    fd_mat = csr_array(fd_mat)
    var = {"ph_temp": None, "dwp": None, "fdd_coef": None, "coef": None, "rhs": None}
    for k in range(1, n_feats):
        phase[0, k] = phase[0, k - 1] - grd[0, k - 1]
    if weighted_rpu is False:
        var["coef"] = fd_mat.T @ fd_mat + np.eye(n_feats)
        var["coef"] = get_band_coef(var["coef"])
        for tau in range(1, n_frame):
            var["ph_temp"] = wrap_phase(phase[tau - 1, :]) + ifreq[tau - 1, :]
            var["dwp"] = fd_mat @ var["ph_temp"]
            grd_new = var["dwp"] + wrap_phase(grd[tau, :] - var["dwp"])
            var["rhs"] = var["ph_temp"] + fd_mat.T @ grd_new
            phase[tau, :] = solve_banded((1, 1), var["coef"], var["rhs"])
    else:
        for tau in range(1, n_frame):
            w_ifreq = amplitude[tau - 1, :] ** weight_power
            w_grd = amplitude[tau, :-1] ** weight_power
            var["fdd_coef"] = fd_mat.T * w_grd
            var["coef"] = np.diag(w_ifreq) + var["fdd_coef"] @ fd_mat
            var["coef"] = get_band_coef(var["coef"])
            var["ph_temp"] = wrap_phase(phase[tau - 1, :]) + ifreq[tau - 1, :]
            var["dwp"] = fd_mat @ var["ph_temp"]
            grd_new = var["dwp"] + wrap_phase(grd[tau, :] - var["dwp"])
            var["rhs"] = w_ifreq * var["ph_temp"] + var["fdd_coef"] @ grd_new
            phase[tau, :] = solve_banded((1, 1), var["coef"], var["rhs"])
    return phase


@torch.no_grad()
def get_ifreq_grd(model_tuple, logamp):
    """Estimate instantaneous frequency and group delay from log-amplitude.

    Args:
        model_tuple (tuple): tuple of DNN params (nn.Module).
        logamp (ndarray): log amplitude spectrum. [T, K]

    Returns:
        ifreq (ndarray): estimated instantaneous frequency. [T-1, K]
        grd (ndarray): group delay. [T, K-1]
    """
    model_ifreq, model_grd = model_tuple  # DNNs
    ifreq = model_ifreq(logamp)
    grd = model_grd(logamp)
    ifreq = ifreq.cpu().detach().numpy().copy().squeeze()
    grd = grd.cpu().detach().numpy().copy().squeeze()
    ifreq = ifreq[:-1, :]
    grd = grd[:, :-1]
    return ifreq, grd


def _reconst_waveform(cfg, model_tuple, logamp_path, scaler, device):
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
    logamp = np.load(logamp_path)
    amplitude = np.exp(scaler.inverse_transform(logamp))
    logamp = np.pad(
        logamp, ((cfg.model.win_range, cfg.model.win_range), (0, 0)), "constant"
    )
    logamp = torch.from_numpy(logamp).float().unsqueeze(0).to(device)
    logamp = logamp.unfold(1, 2 * cfg.model.win_range + 1, 1)
    _, n_frame, _, _ = logamp.shape
    logamp = logamp.reshape(1, n_frame, -1)
    ifreq, grd = get_ifreq_grd(model_tuple, logamp)
    phase = compute_rpu(
        ifreq, grd, amplitude, cfg.demo.weighted_rpu, cfg.demo.weight_power
    )
    if cfg.demo.gla is True:
        phase = compensate_phase(phase, cfg.feature.win_length, phase.shape[0])
        audio = pra.phase.griffin_lim(
            amplitude,
            hop=cfg.feature.hop_length,
            analysis_window=signal.get_window(
                cfg.feature.window, cfg.feature.win_length
            ),
            n_iter=cfg.feature.gla_iter,
            ini=np.exp(1j * phase),
        )
    else:
        reconst_spec = amplitude * np.exp(1j * phase)
        stfft = signal.ShortTimeFFT(
            win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
            hop=cfg.feature.hop_length,
            fs=cfg.feature.sample_rate,
            mfft=cfg.feature.n_fft,
        )
        audio = stfft.istft(reconst_spec.T)

    wav_file = get_wavname(cfg, os.path.basename(logamp_path))
    sf.write(wav_file, audio, cfg.feature.sample_rate)


def reconst_waveform(cfg, model_tuple, logmag_list, device):
    """Reconstruct audio waveforms in parallel.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (Tuple): tuple of DNN params (nn.Module).
        logmag_list (list): list of path to the log-magnitude spectrum.
        device: device info.

    Returns:
        score_list (dict): dictionary of objective score lists.
    """
    print("Reconstruct waveform.")
    set_start_method("spawn")
    stats_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.stats_dir)
    scaler = joblib.load(os.path.join(stats_dir, "stats.pkl"))
    with ProcessPoolExecutor(cfg.preprocess.n_jobs) as executor:
        futures = [
            executor.submit(
                _reconst_waveform, cfg, model_tuple, logmag_path, scaler, device
            )
            for logmag_path in logmag_list
        ]
        for future in prg(futures):
            future.result()  # return None


def compute_eval_score(cfg, model_tuple, logamp_list, device):
    """Compute objective scores; PESQ, STOI and LSC.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (tuple): tuple of DNN params (nn.Module).
        logamp_list (list): list of path to the log-amplitude spectrum.
        device: device info.

    Returns:
        score_list (dict): dictionary of objective score lists.
    """
    print("Compute objective scores.")
    score_list = {"pesq": [], "stoi": [], "lsc": []}
    for logamp_path in prg(logamp_list):
        score_list["pesq"].append(compute_pesq(cfg, os.path.basename(logamp_path)))
        score_list["stoi"].append(compute_stoi(cfg, os.path.basename(logamp_path)))
        score_list["lsc"].append(compute_lsc(cfg, os.path.basename(logamp_path)))
    return score_list


def main(cfg: DictConfig):
    """Perform evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ifreq, model_grd = load_checkpoint(cfg, device)
    model_ifreq.eval()
    model_grd.eval()
    feat_dir = os.path.join(
        cfg.RPU.root_dir, cfg.RPU.feat_dir, cfg.RPU.evalset_dir, cfg.feature.window
    )
    logamp_list = glob.glob(feat_dir + "/*-feats_logabs.npy")
    logamp_list.sort()
    demo_dir = get_wavdir(cfg)
    os.makedirs(demo_dir, exist_ok=True)
    score_dir = os.path.join(cfg.RPU.root_dir, cfg.RPU.score_dir)
    os.makedirs(score_dir, exist_ok=True)
    reconst_waveform(cfg, (model_ifreq, model_grd), logamp_list, device)
    score_dict = compute_eval_score(cfg, (model_ifreq, model_grd), logamp_list, device)
    for score_type, score_list in score_dict.items():
        if cfg.demo.gla is False:
            if cfg.demo.weighted_rpu is False:
                out_filename = os.path.join(score_dir, f"{score_type}_score_0_RPU.txt")
            else:
                out_filename = os.path.join(score_dir, f"{score_type}_score_0_wRPU.txt")
        else:
            if cfg.demo.weighted_rpu is False:
                out_filename = f"{score_type}_score_{cfg.feature.gla_iter}_RPU.txt"
            else:
                out_filename = f"{score_type}_score_{cfg.feature.gla_iter}_wRPU.txt"
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
