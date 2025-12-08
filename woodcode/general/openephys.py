
import pynapple as nap
from woodcode.nwb.io import read_xml
from pathlib import Path
import numpy as np
import pandas as pd
from dateutil import parser

def get_ttl_times(dat_path, channel: int):

    if not isinstance(channel, int):
        raise TypeError(f"channel must be an int, got {type(channel).__name__}")

    dat_path = Path(dat_path) # convert to Path

    if not dat_path.exists():
        raise FileNotFoundError(f"DAT file not found: {dat_path}")


    # try to find an xml file first
    xml_path = dat_path.with_suffix(".xml")
    xml_data = read_xml(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # get info from xml file
    sampling_rate = xml_data['dat_sampling_rate']
    n_channels = xml_data['n_channels']

    # lazy load DAT file
    signal = nap.load_eeg(filepath=str(dat_path), channel=None, n_channels=n_channels,
                            frequency=float(sampling_rate), precision='int16',
                            bytes_size=2)
    signal = signal[:, channel]  # get only probe channels

    total_samples = len(signal)  # total number of samples

    # detect TTL rise and fall times

    # Normalize to 0..1
    signal = signal - np.min(signal)
    signal = signal / np.max(signal)

    # Convert to binary (0/1) using midpoint threshold
    signal = signal > 0.5  # True = high, False = low

    # Detect edges
    diff = np.diff(signal.astype(int))
    rising_edges = np.where(diff == 1)[0] + 1
    falling_edges = np.where(diff == -1)[0] + 1

    return rising_edges, falling_edges, total_samples


def get_tracking_bonsai(bonsai_path):
    # Read Bonsai CSV
    df = pd.read_csv(bonsai_path, header=0, dtype=str)  # read all as string to be safe

    # Assume the first column is timestamp
    ts_series = df.iloc[:, 0]  # select the timestamp column as a Series

    # Parse timestamps
    dt_series = ts_series.apply(parser.isoparse)

    # Compute elapsed time in seconds
    bonsai_timestamps = (dt_series - dt_series.iloc[0]).dt.total_seconds()

    # Get tracking data (all columns after the timestamp)
    tracking_data = df.iloc[:, 1:].to_numpy().astype(float)  # convert numeric columns
    tracking_data = np.column_stack([bonsai_timestamps, tracking_data])

    return tracking_data

