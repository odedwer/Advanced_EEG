import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scf
import scipy.io as scio
import fooof as f
import scipy.signal as scs
import mne
import mne.time_frequency as tf
import seaborn as sns

def load_data(filename):
    """
    Loads a single matrix from a given .mat file
    :param filename: The full path to the .mat file to load
    :return: np.array with the saved matrix in the .mat file
    """
    return scio.loadmat(filename)


def get_data_from_attention(mat_file):
    data: np.ndarray = mat_file['data']
    info = mne.create_info([ch.item() for ch in data['clab'].item().squeeze()], data['fs'].item().squeeze().item(),
                           [("eog" if "eog" in ch.item().lower() else "eeg") for ch in data['clab'].item().squeeze()])
    raw = mne.io.RawArray(data['X'].item().T, info)
    raw._data *= 1e-6
    mnt = mat_file['mnt']
    mrk = mat_file['mrk']
    return raw, data, mnt, mrk


def plot_psd(psd, freqs, fig=None, axs=None):
    if fig is None:
        fig = plt.figure()
        axs = fig.subplots(1, 3)
        axs[0].set_title("linear PSD")
        axs[1].set_yscale('log')
        axs[1].set_title("log-linear PSD")
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].set_title("log-log PSD")
    axs[0].plot(freqs, np.squeeze(psd))
    axs[1].plot(freqs, np.squeeze(psd))
    axs[2].plot(freqs, np.squeeze(psd))
    return fig, axs


def get_fooof_peaks(psd, freqs, exclution_elecs):
    peak_freqs = []
    peak_amps = []
    for i in range(psd.shape[0]):
        if i in exclution_elecs:
            continue
        fooof: f.FOOOF = f.FOOOF(peak_width_limits=[2.5, 12], max_n_peaks=1)
        fooof.add_data(freqs, psd[i, :], freq_range=[1, 35])
        fooof.fit()
        for peak_specs in fooof.peak_params_:
            peak_freqs.append(peak_specs[0])
            peak_amps.append(peak_specs[1])
        if fooof.peak_params_.size == 0:
            peak_amps.append(0)
            peak_freqs.append(-1)
    return np.array(peak_amps), np.array(peak_freqs)


# %%
mat_file = load_data("Advanced_EEG/ex1/covertShiftsOfAttention_VPiac.mat")
raw, data, mnt, mrk = get_data_from_attention(mat_file)

# %% 1
psd, freqs = tf.psd_welch(raw, picks='O1', n_per_seg=int(2 * raw.info['sfreq']),
                          n_overlap=raw.info['sfreq'], n_fft=int(2 * raw.info['sfreq']))
psd = np.squeeze(psd)  # required to remove redundant axis
plot_psd(psd, freqs)
# a. log-linear coordinates are the most informative,
# making it easier to differentiate between the frequencies in the x axis
# b. the center frequency is ~10.5Hz. The beta peak is small compared to the aperiodic slope & the alpha peak,
# causing us to speculate that it is a harmonic of the alpha peak and not actually

# %% 2
misc = mrk["misc"].item()
run_start_indices = misc["pos"].item().squeeze()[np.where(misc["y"].item()[2, :])[0]] - 1
run_end_indices = misc["pos"].item().squeeze()[np.where(misc["y"].item()[3, :])[0]] - 1

run_raws = []
fig, axs = None, None
run_psd, run_freqs, run_cf, run_pw = [], [], [], []
for i in range(len(run_start_indices)):
    run_raws.append(mne.io.RawArray(raw.get_data(start=run_start_indices[i],
                                                 stop=run_end_indices[i]), raw.info.copy()))
    psd, freqs = tf.psd_welch(run_raws[-1], picks='O1', n_per_seg=int(2 * raw.info['sfreq']),
                              n_overlap=raw.info['sfreq'], n_fft=int(2 * raw.info['sfreq']))
    psd = np.squeeze(psd)  # required to remove redundant axis
    run_psd.append(psd)
    run_freqs.append(freqs)
    fig, axs = plot_psd(psd, freqs, fig, axs)
    fooof: f.FOOOF = f.FOOOF(peak_width_limits=[1.5, 12], max_n_peaks=1)
    fooof.add_data(freqs, psd, freq_range=[1, 35])
    fooof.fit()
    run_cf.append(fooof.peak_params_[0, 0])  # center frequency
    run_pw.append(fooof.peak_params_[0, 1])  # power
for ax in axs:
    ax.legend([f"run {i}" for i in range(len(run_raws))])
# plot peak pw and center freq
fig2 = plt.figure()
axes2 = fig2.subplots(1, 2)
axes2[0].plot(np.arange(1, 7), run_cf)
axes2[0].set_title("Center frequency per run")
axes2[1].plot(np.arange(1, 7), run_pw)
axes2[1].set_title("peak alpha power per run")

# a. We can see that the alpha peak in the first run is lower than the other 5,
# b. plotting the center frequncy and it's power reveals a relatinship that matches our inspection:
# a U-shape (and inverted U for power) function as a function of run.
# c. suppression might have been worse until the subject learned how to perform the task

# %% 3
mat_file = load_data("Advanced_EEG/ex1/covertShiftsOfAttention_VPiae.mat")
raw, data, mnt, mrk = get_data_from_attention(mat_file)
psd, freqs = tf.psd_welch(raw, picks='eeg', n_per_seg=int(2 * raw.info['sfreq']),
                          n_overlap=raw.info['sfreq'], n_fft=int(2 * raw.info['sfreq']))
raw.plot_psd(n_overlap=raw.info['sfreq'], n_fft=int(2 * raw.info['sfreq']), fmax=40)

# We identify several parital electrodes with elevated broadband power in beta-gamma frequencies, non-oscillatory.
# In addition, frontal electrodes as F7-F8 are not showing alpha band activity.
exclution_elecs = np.where((np.asarray(raw.ch_names) == 'F7') | (np.asarray(raw.ch_names) == 'F8'))[0]
peak_amps, peak_freqs = get_fooof_peaks(psd, freqs, exclution_elecs)
plt.hist(peak_freqs[peak_freqs >= 0])

# create position array fro topoplot
pos = np.hstack([mnt["x"].item(), mnt["y"].item()])
pos_idx = np.zeros(pos.shape[0], dtype=bool)
eeg_idx = mne.channel_indices_by_type(raw.info)["eeg"]

pos_idx[eeg_idx] = True
pos_idx[exclution_elecs] = False
pos = pos[pos_idx, :]  # remove non eeg elecs

# theta topoplot
theta_amps = peak_amps.copy()
theta_amps[~((peak_freqs < 8) & (peak_freqs >= 5))] = 0
mne.viz.plot_topomap(theta_amps, pos, cmap="Greens")
# alpha topoplot
plt.figure()
alpha_amps = peak_amps.copy()
alpha_amps[~((peak_freqs < 11) & (peak_freqs >= 8))] = 0
mne.viz.plot_topomap(alpha_amps, pos, cmap='Blues')

# %% 4
filenames = ["Piac","Piae"]
for i in range(len(filenames)):
    mat_file = load_data(f"Advanced_EEG/ex1/covertShiftsOfAttention_V{filenames[i]}.mat")
    raw, data, mnt, mrk = get_data_from_attention(mat_file)
    psd, freqs = tf.psd_welch(raw, picks='eeg', n_per_seg=int(2 * raw.info['sfreq']),
                              n_overlap=raw.info['sfreq'], n_fft=int(2 * raw.info['sfreq']))

    # We identify several parital electrodes with elevated broadband power in beta-gamma frequencies, non-oscillatory.
    # In addition, frontal electrodes as F7-F8 are not showing alpha band activity.
    exclution_elecs = np.where((np.asarray(raw.ch_names) == 'F7') | (np.asarray(raw.ch_names) == 'F8'))[0]
    peak_amps, peak_freqs = get_fooof_peaks(psd, freqs, [])
    sns.kdeplot(peak_freqs[(peak_freqs>8)&(peak_freqs<12)],
                fill=True,label=filenames[i])
    plt.legend()

#We can see large individual differences in the alpha peak frequencies
# and between-electrodes variability across subjects. This hints that the use of pre-defined bands
# might introduce artifacts due to inclusion and exclusion of relevant bands in different subjects.