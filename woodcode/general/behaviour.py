import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt


def compute_velocity(pos: nap.TsdFrame, window_size: int = None, sampling_rate: float = None) -> nap.Tsd:
    """
    Computes the velocity of an animal based on position data.

    Parameters:
    -----------
    pos : nap.TsdFrame
        A TsdFrame with time as the index and two columns (x, y) for position data.
    window_size : int, optional
        Moving average window size in bins. If None, no smoothing is applied.
    sampling_rate : float, optional
        Sampling rate in Hz (samples per second). If None, the median interval between samples is used.

    Returns:
    --------
    vel : nap.Tsd
        A Tsd containing velocity values indexed by time.
    """

    if not isinstance(pos, nap.TsdFrame):
        raise TypeError("Input must be a pynapple TsdFrame.")

    if pos.shape[1] != 2:
        raise ValueError("Input TsdFrame must have exactly two columns (x, y).")

    # Extract timestamps and position data
    timestamps = pos.index.to_numpy()  # Convert to NumPy array
    pos_values = pos.values  # Convert to NumPy array

    # Determine tracking interval
    if sampling_rate is None:
        tracking_interval = np.median(np.diff(timestamps))  # Compute from timestamps
    else:
        tracking_interval = 1.0 / sampling_rate  # Define explicitly

    # Compute displacement in x and y
    displacement = np.diff(pos_values, axis=0)

    # Compute 2D displacement (Euclidean distance)
    displacement = np.hypot(displacement[:, 0], displacement[:, 1])

    # Compute velocity
    vel = displacement / tracking_interval

    # Convert to pandas Series
    vel_series = pd.Series(vel, index=timestamps[1:])

    # Apply moving average smoothing if window_size is specified
    if window_size is not None:
        vel_series = vel_series.rolling(window=window_size, center=True).mean()

    # Convert back to nap.Tsd (ensuring timestamps are NumPy arrays)
    vel_tsd = nap.Tsd(t=np.array(vel_series.index), d=vel_series.values)

    return vel_tsd


def get_lap_intervals(track_pos_tsd, track_end=0.15, plot_result=False):
    """
    Similar logic to your step-based code, but if there's no 'next crossing'
    for the final lap, we treat [last crossing : end] as a partial lap.
    """

    pos_series = track_pos_tsd.as_series().dropna()
    if len(pos_series) < 2:
        return {"direction_1": nap.IntervalSet([],[]), "direction_2": nap.IntervalSet([],[])}

    timestamps = pos_series.index.values.astype(float)
    pos = pos_series.values

    pmin, pmax = pos.min(), pos.max()
    if pmin == pmax:
        return {"direction_1": nap.IntervalSet([],[]), "direction_2": nap.IntervalSet([],[])}

    # Normalize
    norm_pos = (pos - pmin) / (pmax - pmin)
    N = len(norm_pos)
    left_boundary = track_end
    right_boundary = 1.0 - track_end

    # Where the rat crosses left or right boundaries
    left_crosses = np.where(norm_pos < left_boundary)[0]
    right_crosses = np.where(norm_pos > right_boundary)[0]

    if len(left_crosses) == 0 or len(right_crosses) == 0:
        return {"direction_1": nap.IntervalSet(), "direction_2": nap.IntervalSet()}

    # Helper: find next left crossing after index
    def next_left_cross(after_ix):
        cands = left_crosses[left_crosses > after_ix]
        if len(cands) == 0:
            return None
        return cands[0]

    # Helper: find next right crossing after index
    def next_right_cross(after_ix):
        cands = right_crosses[right_crosses > after_ix]
        if len(cands) == 0:
            return None
        return cands[0]

    def get_local_min_in_segment(s_ix, e_ix):
        if s_ix > e_ix:
            s_ix, e_ix = e_ix, s_ix
        segment_idx = np.arange(s_ix, e_ix + 1, dtype=int)
        if len(segment_idx) == 0:
            return None, None
        sub_vals = norm_pos[segment_idx]
        i_min = np.argmin(sub_vals)
        best_ix = segment_idx[i_min]
        return timestamps[best_ix], best_ix

    def get_local_max_in_segment(s_ix, e_ix):
        if s_ix > e_ix:
            s_ix, e_ix = e_ix, s_ix
        segment_idx = np.arange(s_ix, e_ix + 1, dtype=int)
        if len(segment_idx) == 0:
            return None, None
        sub_vals = norm_pos[segment_idx]
        i_max = np.argmax(sub_vals)
        best_ix = segment_idx[i_max]
        return timestamps[best_ix], best_ix

    # Determine which boundary crossing is first
    first_left = left_crosses[0]
    first_right = right_crosses[0]

    if first_left < first_right:
        # Earliest crossing is left => going UP
        state = 'UP'
        last_left_ix = first_left
        last_right_ix = None
    else:
        # Earliest crossing is right => going DOWN
        state = 'DOWN'
        last_right_ix = first_right
        last_left_ix = None

    up_laps = []
    down_laps = []

    while True:
        if state == 'DOWN':
            # We have last_right_ix. Next we want the left crossing after it
            if last_right_ix is None:
                break
            this_left = next_left_cross(last_right_ix)
            if this_left is None:
                # => No next left crossing. We'll do a FINAL partial lap from
                # [last_right_ix : end of data].
                # start is local max in [last_right_ix : N-1]
                start_t, start_ix = get_local_max_in_segment(last_right_ix, N - 1)
                # end is local min in [start_ix : N-1]
                # (or you might do [last_right_ix : N-1] again, depending on your definition)
                # Because the rat doesn't cross left boundary again,
                # let's just define the local min in [start_ix, N-1]
                if start_ix is not None:
                    end_t, end_ix = get_local_min_in_segment(start_ix, N - 1)
                    if (end_t is not None) and (start_t < end_t):
                        down_laps.append((start_t, end_t))
                break

            # start is local max in [last_right_ix, this_left]
            start_t, start_ix = get_local_max_in_segment(last_right_ix, this_left)
            if start_t is None:
                break

            # next crossing on the right side after this_left
            next_right = next_right_cross(this_left)
            if next_right is None:
                # partial final lap in [this_left : end of data]
                # start is the local max we found.
                # end is local min in [this_left : N-1]
                end_t, end_ix = get_local_min_in_segment(this_left, N - 1)
                if (end_t is not None) and (start_t < end_t):
                    down_laps.append((start_t, end_t))
                break

            # end is local min in [this_left, next_right]
            end_t, end_ix = get_local_min_in_segment(this_left, next_right)
            if end_t is None:
                break

            if start_t < end_t:
                down_laps.append((start_t, end_t))

            # Move on
            # The end of this lap => start of next (UP) lap
            last_left_ix = this_left  # so next UP has a 'last_left_ix'
            last_right_ix = next_right
            state = 'UP'

        else:
            # state == 'UP'
            if last_left_ix is None:
                break

            # find the right crossing after last_left_ix
            this_right = next_right_cross(last_left_ix)
            if this_right is None:
                # partial final lap in [last_left_ix : end of data]
                # start is local min
                start_t, start_ix = get_local_min_in_segment(last_left_ix, N - 1)
                if start_ix is not None:
                    end_t, end_ix = get_local_max_in_segment(start_ix, N - 1)
                    if (end_t is not None) and (start_t < end_t):
                        up_laps.append((start_t, end_t))
                break

            # start is local min in [last_left_ix, this_right]
            start_t, start_ix = get_local_min_in_segment(last_left_ix, this_right)
            if start_t is None:
                break

            # next left crossing after this_right
            next_left = next_left_cross(this_right)
            if next_left is None:
                # partial final lap in [this_right : end of data]
                end_t, end_ix = get_local_max_in_segment(this_right, N - 1)
                if (end_t is not None) and (start_t < end_t):
                    up_laps.append((start_t, end_t))
                break

            # end is local max in [this_right, next_left]
            end_t, end_ix = get_local_max_in_segment(this_right, next_left)
            if end_t is None:
                break

            if start_t < end_t:
                up_laps.append((start_t, end_t))

            # Move on
            last_right_ix = this_right
            last_left_ix = next_left
            state = 'DOWN'

    # Build IntervalSets
    up_iv = nap.IntervalSet(
        start=np.array([t[0] for t in up_laps]),
        end=np.array([t[1] for t in up_laps])
    )
    down_iv = nap.IntervalSet(
        start=np.array([t[0] for t in down_laps]),
        end=np.array([t[1] for t in down_laps])
    )

    intervals = {
        "direction_1": up_iv,
        "direction_2": down_iv,
    }

    # Optional: Plot
    if plot_result:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timestamps, pos, 'k-', lw=1)
        # shade intervals
        for (s, e) in intervals["direction_1"].values:
            ax.axvspan(s, e, color='red', alpha=0.15)
        for (s, e) in intervals["direction_2"].values:
            ax.axvspan(s, e, color='blue', alpha=0.15)
        # boundaries
        ax.set_title("Lap Intervals")
        ax.set_xlabel("Time")
        ax.set_ylabel("Track position")
        plt.show()

    return intervals






