import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs
import scipy.fft as scf


# %%
def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(- 2 * np.pi * 1J / N)
    W = np.power(omega, i * j) / np.sqrt(N)
    return W


def DTFT(f, t, freq):
    """

    :param f: the function to calculate DTFT of
    :param t: time to evaluate f at
    :param freq: freqs to evaluate DTFT at
    :return: the DTFT of the function with the given freqs
    """
    x = f(t)
    U = np.exp((-1j) * 2 * np.pi * np.outer(freq, np.arange(x.size)))
    return U @ x


time = np.arange(0, 5, 1e-3)
sin = np.sin(2 * np.pi * time)
plt.plot(time, sin)
freqs = np.arange(-20,20+1e-1,1e-1)
plt.plot(freqs, np.abs(DTFT(lambda t: np.sin(2 * np.pi * t), time, freqs)))

# %%
