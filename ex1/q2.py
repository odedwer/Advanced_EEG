from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


# %% a
def DTFT(func, time, freqs) -> np.array:
    r"""
    Calculates DTFT of function, using the formula:
    .. math::
        \tilde{x}[f] = \sum_{n=-\infty}^{\infty} x[n]e^{-i*2*\pi*f*n}
    :param func: the function to calculate DTFT of
    :param time: time to evaluate f at
    :param freqs: freqs to evaluate DTFT at
    :return: the DTFT of the function with the given freqs
    """
    x = func(time) if callable(func) else func  # function results
    U = np.exp((-1j) * 2 * np.pi * np.outer(freqs, time))  # DTFT matrix
    return U @ x


def gen_sig_dtft(func, fs, min_t=-2, max_t=2, dt=None, min_freq=-20, max_freq=20, freq_step=1e-1):
    time = np.arange(min_t, max_t, (1 / fs) if dt is None else dt)
    freqs = np.arange(min_freq, max_freq + freq_step, freq_step)
    y = func(time) if dt is None else func(time, fs)
    dtft = DTFT(y, time, freqs)
    return y, time, dtft, freqs


def plot_dtft_with_sig(y, time, dtft, freqs, fig=None, func_ax=None, dtft_ax=None) -> Tuple[
    plt.Figure, plt.Axes, plt.Axes]:
    if not fig:
        fig = plt.figure()
        fig.suptitle(f"Sampled at {int(round(1 / (time[1] - time[0])))}Hz")
        func_ax, dtft_ax = fig.subplots(1, 2)
        func_ax.set_title("function")
        func_ax.set_xlabel("time (sec)")
        func_ax.set_ylabel("value")
        dtft_ax.set_title("DTFT")
        dtft_ax.set_xlabel("Frequency (Hz)")
        func_ax.set_ylabel("Amplitude")
    func_ax.plot(time, y)
    func_ax.scatter(time, y, edgecolors='r', facecolors='none', s=10)
    dtft_ax.plot(freqs, np.abs(dtft))
    dtft_ax.scatter(freqs, np.abs(dtft), edgecolors='r', facecolors='none', s=10)
    return fig, func_ax, dtft_ax


def get_dirac_comb(time, fs):
    """
    returns the dirac comb representing the resampling operation of fs to fs/decim
    :param step: the decimation factor of the original sampling frequency
    :return: the dirac comb corresponding to resampling a vector with
    size timepoints, from fs sampling rate to fs/decim sampling rate`
    """
    comb = np.zeros_like(time)
    comb[::1000 // fs] = 1
    return comb


# %% b+c
triangle_func = lambda t: (2 - np.abs(t)) * (np.abs(t) < 2)
figs, func_axes, dtft_axes = [], [], []
fs_list = [100, 50, 10, 5, 2, 1]
for fs in fs_list:
    fig, func_ax, dtft_ax = plot_dtft_with_sig(*gen_sig_dtft(triangle_func, fs))
    figs.append(fig)
    func_axes.append(func_ax)
    dtft_axes.append(dtft_ax)
# %% d
for i, fs in enumerate(fs_list):
    plot_dtft_with_sig(*gen_sig_dtft(get_dirac_comb, fs, min_t=-10, max_t=10, dt=1e-3), figs[i], func_axes[i],
                       dtft_axes[i])
    func_axes[i].legend(["triangle", "dirac comb"])
    dtft_axes[i].legend(["triangle", "dirac comb"])

# %% e
triangle, triangle_time, triangle_dtft, triangle_freqs = gen_sig_dtft(triangle_func, 100)
dirac, dirac_time, dirac_dtft, dirac_freqs = gen_sig_dtft(get_dirac_comb, 100, min_t=-10, max_t=10, dt=1e-3)
dtft_of_triangle = np.convolve(dirac_dtft, triangle_dtft, mode="same")

plt.plot(dirac_freqs, np.abs(dtft_of_triangle) / (dirac.size / 10), label="convolved")
plt.plot(dirac_freqs, np.abs(triangle_dtft), label="calculated")
plt.legend()
