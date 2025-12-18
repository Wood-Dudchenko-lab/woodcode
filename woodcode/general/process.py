import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def trim_int_to_tsd(int, tsd):

    tsd_restricted = tsd.restrict(int)
    new_int = nap.IntervalSet(start=tsd_restricted.index[0], end=tsd_restricted.index[-1])

    return new_int


def get_waveform_features(nwbfile, plot_result=False):
    """
    Extracts waveform features from an NWB file, selecting the waveform with the highest amplitude
    and computing trough-to-peak duration for each cell without upsampling.

    Additionally, assigns NaN to trough-to-peak duration if the average of the first 1/3
    of samples is lower than the average of the second 2/3.

    Parameters:
    - nwbfile: NWB file object containing spike waveforms.
    - plot_result (bool, optional): Whether to plot the result. Default is False.

    Returns:
    - waveform_features (pd.DataFrame): DataFrame containing waveform features, including trough-to-peak durations.
    """

    # Get waveform and sampling rate info from the NWB file
    waveforms = nwbfile.nwb.units['waveform_mean'].data[:]

    sampling_rate = nwbfile.nwb.units['sampling_rate'].data[0]

    # Unpack shape dynamically
    n_cells, n_samples, n_channels = waveforms.shape

    # Compute sum of squared values for each channel
    sum_squared = np.sum(waveforms ** 2, axis=1)  # (n_cells, n_channels)

    # Find the index of the channel with the maximum sum for each cell
    max_channel_indices = np.argmax(sum_squared, axis=1)  # (n_cells,)

    # Extract the max waveform efficiently
    max_waveforms = np.take_along_axis(waveforms, max_channel_indices[:, None, None], axis=2).squeeze(axis=2)

    # Generate time vector in milliseconds
    t = np.linspace(0, (n_samples - 1) / sampling_rate * 1000, n_samples)

    # Find trough (minimum value index) in a vectorized manner
    trough_indices = np.argmin(max_waveforms, axis=1)  # (n_cells,)

    # Find peak after the trough
    peak_indices = np.array([
        trough_idx + np.argmax(max_waveforms[cell_idx, trough_idx:])
        for cell_idx, trough_idx in enumerate(trough_indices)
    ])

    # Compute trough-to-peak durations in milliseconds
    trough_to_peak_durations = t[peak_indices] - t[trough_indices]

    # **New Feature: Assign NaN if first 1/4 mean < second 1/4 mean**
    baseline_idx = n_samples // 4
    trough_idx = n_samples // 2
    first_quarter_mean = np.mean(max_waveforms[:, :baseline_idx], axis=1)
    second_quarter_mean = np.mean(max_waveforms[:, baseline_idx:trough_idx], axis=1)

    mask = first_quarter_mean < second_quarter_mean
    num_positive_waveforms = np.sum(mask)  # Count how many waveforms meet the condition
    trough_to_peak_durations[mask] = np.nan  # Assign NaN where condition is met

    # Display how many positive waveforms were detected
    print(f"Detected {num_positive_waveforms} positive waveforms (assigned NaN).")

    # Create a DataFrame for the results
    waveform_features = pd.DataFrame({
        'cell_index': np.arange(n_cells),
        'max_channel_index': max_channel_indices,
        'trough_to_peak': trough_to_peak_durations
    })

    if plot_result:
        fig, axs = plt.subplots(2, 1, figsize=(4, 6), constrained_layout=True)

        # Use a color palette for aesthetics
        colors = sns.color_palette("husl", n_cells)

        # Plot all mean waveforms individually, normalized
        for i in range(n_cells):
            min_val = np.min(max_waveforms[i])
            normalized_waveform = max_waveforms[i] / abs(min_val)  # Normalize so min is 1

            axs[0].plot(t, normalized_waveform, color=colors[i], alpha=0.8, lw=1)

        # Formatting for waveform plot
        axs[0].set_xlabel("Time (ms)", fontsize=8, fontname="Arial")
        axs[0].set_ylabel("Normalized Amplitude", fontsize=8, fontname="Arial")
        axs[0].set_title("Normalized Mean Waveforms", fontsize=8, fontweight="bold", pad=10)
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        axs[0].tick_params(direction="out", length=4, width=1, labelsize=10)

        # Plot histogram of trough-to-peak durations with KDE overlay
        sns.histplot(trough_to_peak_durations, bins=20, color="gray", alpha=0.7, edgecolor="black",
                     ax=axs[1])

        # Formatting for histogram
        axs[1].set_xlabel("Trough-to-Peak Duration (ms)", fontsize=8, fontname="Arial")
        axs[1].set_ylabel("Count", fontsize=8, fontname="Arial")
        axs[1].set_title("Trough-to-Peak Duration Distribution", fontsize=8, fontweight="bold", pad=10)
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].tick_params(direction="out", length=4, width=1, labelsize=10)

        # Show the final polished figure
        plt.show()

    return waveform_features


