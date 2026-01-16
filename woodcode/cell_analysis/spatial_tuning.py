import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau
import pynapple as nap


import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def smooth_1d_tuning_curves(
    tuning_curves,
    smooth_factor: float = None,
    circular: bool = False,
    nan_handling: str = "interpolate"):
    """
    Applies Gaussian smoothing to 1D tuning curves with options for circular continuity and NaN handling.

    Parameters
    ----------
    tuning_curves : pd.DataFrame, pd.Series, or np.ndarray
        1D or 2D input where rows represent tuning curve bins and columns represent different units.
    smooth_factor : float, optional
        The standard deviation for Gaussian smoothing. If None, no smoothing is applied.
    circular : bool, optional
        Whether to apply circular smoothing by replicating the tuning curves.
    nan_handling : str, optional
        How to handle NaNs before smoothing:
        - 'interpolate': Linearly interpolate missing values **before circular extension**.
        - 'zero': Replace NaNs with zeros before smoothing.
        - 'ignore': Leave NaNs untouched (may propagate).

    Returns
    -------
    Same type as input
        Smoothed tuning curves in the same format as the input.
    """

    # Keep track of input type
    input_type = type(tuning_curves)
    input_index = None
    input_columns = None

    if isinstance(tuning_curves, pd.DataFrame):
        input_index = tuning_curves.index
        input_columns = tuning_curves.columns
        arr = tuning_curves.to_numpy()
    elif isinstance(tuning_curves, pd.Series):
        input_index = tuning_curves.index
        arr = tuning_curves.to_numpy().reshape(-1, 1)
    elif isinstance(tuning_curves, np.ndarray):
        arr = tuning_curves.copy()
    else:
        raise TypeError("tuning_curves must be a Pandas DataFrame, Series, or NumPy ndarray.")

    if smooth_factor is None:
        smoothed_data = arr.copy()
    else:
        # Ensure 2D array (rows = bins, cols = units)
        arr = np.atleast_2d(arr)

        # Handle NaNs
        if nan_handling == "interpolate":
            nan_mask = np.isnan(arr)
            if np.any(nan_mask):
                x = np.arange(arr.shape[0])
                for i in range(arr.shape[1]):
                    col = arr[:, i]
                    if np.any(np.isnan(col)):
                        valid = ~nan_mask[:, i]
                        arr[:, i] = np.interp(x, x[valid], col[valid])
        elif nan_handling == "zero":
            np.nan_to_num(arr, copy=False)

        # Apply smoothing
        if circular:
            n_bins = arr.shape[0]
            extended = np.concatenate((arr, arr, arr), axis=0)
            smoothed_data = gaussian_filter1d(extended, sigma=smooth_factor, axis=0, mode="wrap")
            smoothed_data = smoothed_data[n_bins:n_bins*2]
        else:
            smoothed_data = gaussian_filter1d(arr, sigma=smooth_factor, axis=0, mode="nearest")

    # Restore original format
    if input_type is pd.DataFrame:
        return pd.DataFrame(smoothed_data, index=input_index, columns=input_columns)
    elif input_type is pd.Series:
        return pd.Series(smoothed_data.ravel(), index=input_index)
    else:  # ndarray
        return smoothed_data


def compute_1d_tuning_correlation(data1, data2, method: str = "pearson", circular: bool = False) -> np.ndarray:
    """
    Compute column-wise correlation between corresponding columns of two inputs.
    Optionally, perform circular shifting of data2 before computing correlations.

    Parameters:
    data1 (pd.DataFrame, pd.Series, or np.ndarray): First input (features x bins).
    data2 (pd.DataFrame, pd.Series, or np.ndarray): Second input (features x bins).
    method (str): Correlation method, one of 'pearson', 'spearman', or 'kendall'. Default is 'pearson'.
    circular (bool): Whether to compute correlations with circular shifts of data2. Default is False.

    Returns:
    np.ndarray: If circular=False, a (1, n_features) array with correlations.
                If circular=True, an (n_bins, n_features) array where each row corresponds to a shift.
    """

    # Convert Pandas inputs to NumPy arrays
    if isinstance(data1, (pd.DataFrame, pd.Series)):
        data1 = data1.to_numpy()
    if isinstance(data2, (pd.DataFrame, pd.Series)):
        data2 = data2.to_numpy()

    # Ensure inputs are 2D arrays
    data1 = np.atleast_2d(data1)
    data2 = np.atleast_2d(data2)

    if data1.shape != data2.shape:
        raise ValueError("Both inputs must have the same shape.")

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Method must be one of 'pearson', 'spearman', or 'kendall'.")

    n_bins, n_cols = data1.shape

    if method == "spearman":
        # Rank transform for Spearman correlation
        data1 = np.apply_along_axis(lambda x: x.argsort().argsort(), axis=0, arr=data1)
        data2 = np.apply_along_axis(lambda x: x.argsort().argsort(), axis=0, arr=data2)

    if not circular:
        # Standard correlation
        if method in {"pearson", "spearman"}:
            corrs = np.array([np.corrcoef(data1[:, i], data2[:, i])[0, 1] for i in range(n_cols)])
        else:  # Kendall
            corrs = np.array([kendalltau(data1[:, i], data2[:, i])[0] for i in range(n_cols)])

        return corrs#.reshape(1, -1)  # Shape: (1, n_features)

    else:
        # Circular mode: Compute correlations for each shift
        shift_correlations = np.zeros((n_bins, n_cols))

        for shift in range(n_bins):
            rolled_data2 = np.roll(data2, shift=shift, axis=0)  # Fast circular shift
            if method in {"pearson", "spearman"}:
                corrs = np.array([np.corrcoef(data1[:, i], rolled_data2[:, i])[0, 1] for i in range(n_cols)])
            else:  # Kendall
                corrs = np.array([kendalltau(data1[:, i], rolled_data2[:, i])[0] for i in range(n_cols)])

            shift_correlations[shift, :] = corrs  # Store correlations for each shift

        return shift_correlations  # Shape: (n_bins, n_features)


def compute_occupancy_1d(feature: nap.Tsd, nb_bins: int,
                              minmax: tuple,
                              ep: nap.IntervalSet = None) -> pd.Series:
    """
    Compute histogram of occupancy of the animal's heading angle.
    Mimics pynapple.compute_1d_tuning_curves argument style.
    Parameters
    ----------
    feature : nap.Tsd
      Time series with heading angle samples (radians).
    nb_bins : int, optional
      Number of bins. Default is 20.
    minmax : tuple, optional
      (min, max) range for heading angles. Default is (0, 2π).
    ep : nap.IntervalSet, optional
      Epochs to restrict analysis. If None, use the entire feature.
    Returns
    -------
    pd.Series
      pandas Series of length `nb_bins` with occupancy counts per bin.
    """
    if ep is not None:
        feature = feature.restrict(ep)
    values = np.mod(feature.values, 2 * np.pi)
    bin_edges = np.linspace(minmax[0], minmax[1], nb_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return pd.Series(counts, index=bin_centers)


def compute_vector_length(tcs):
    """
    Compute Rayleigh mean vector length for tuning curves.

    Parameters
    ----------
    tcs : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Tuning curves.
        - If DataFrame or Series, index should represent bin angles (radians).
        - If ndarray, bins are assumed uniformly spaced between 0 and 2π.

    Returns
    -------
    rayleigh_vector_length : pandas.Series
        Rayleigh vector length for each cell.
        - If input is Series or 1D array, returns length-1 Series.
    """
    # If pandas object, extract bins from index
    if isinstance(tcs, (pd.Series, pd.DataFrame)):
        bins_arr = np.asarray(tcs.index, dtype=float)
        tcs_arr = np.asarray(tcs, dtype=float)
    else:
        # Assume uniform bins across [0, 2π)
        n_bins = tcs.shape[0] if tcs.ndim > 1 else len(tcs)
        bins_arr = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
        tcs_arr = np.asarray(tcs, dtype=float)

    # Ensure 2D (nBins, nCells)
    if tcs_arr.ndim == 1:
        tcs_arr = tcs_arr[:, None]

    # Complex unit vectors for each bin
    unit_vectors = np.cos(bins_arr) + 1j * np.sin(bins_arr)

    # Weighted complex sum across bins for each cell
    weighted_sum = tcs_arr.T @ unit_vectors

    # Rayleigh vector length
    vector_length = np.abs(weighted_sum) / tcs_arr.sum(axis=0)

    # Build pandas index
    if isinstance(tcs, pd.DataFrame):
        index = tcs.columns
    elif isinstance(tcs, pd.Series):
        index = pd.Index([tcs.name or "cell0"], name="cell")
    else:
        n_cells = tcs_arr.shape[1]
        index = pd.RangeIndex(n_cells, name="cell")

    return pd.Series(vector_length, index=index, name="rayleigh_vector_length")


def compute_spatial_info_1d(tcs, occ=None):
    """
    Compute 1D spatial / head-direction information (bits per spike) using the Skaggs measure.

    This implements the classic information-per-spike metric:

        I = Σ_i p_i * (λ_i / λ̄) * log2(λ_i / λ̄)

    where:
        - i indexes bins (spatial bins, head-direction bins, etc.)
        - p_i is occupancy probability of bin i
        - λ_i is the firing rate in bin i
        - λ̄ = Σ_i p_i * λ_i is the occupancy-weighted mean firing rate

    Parameters
    ----------
    tcs : array-like or pandas object
        Tuning curve(s) expressed as non-negative firing rates per bin (e.g., Hz).

        Accepted forms:
          - Single tuning curve:
              * np.ndarray of shape (n_bins,)
              * pd.Series of length n_bins
            Returns a scalar float.

          - Multiple tuning curves:
              * np.ndarray of shape (n_bins, n_cells)
              * pd.DataFrame of shape (n_bins, n_cells)
            Returns a pd.Series (length n_cells).

        Notes for 2D rate maps:
          - If you have a 2D place-field rate map (e.g., shape (nx, ny) per cell),
            this function expects a 1D vector of bins. You can flatten the map
            (and occupancy) before calling, e.g. `rate_map.ravel()` (and
            `occ_map.ravel()`), or reshape to (n_bins, n_cells) for multiple cells.
            Information will then be computed over the flattened bins.

    occ : array-like, pandas object, optional
        Occupancy distribution across bins (time spent / probability per bin).
        Must have length n_bins. If provided, it will be normalised to sum to 1.
        If None, uniform occupancy is assumed.

    Returns
    -------
    spatial_info : float or pandas.Series
        Skaggs information per spike (bits/spike).
          - float for single input tuning curve
          - pd.Series for multi-cell input

    Raises
    ------
    ValueError
        If `tcs` is not 1D or 2D, if `occ` length does not match n_bins,
        or if `occ` sums to <= 0.

    Notes
    -----
    - Bins with λ_i = 0 contribute 0 to the sum (handled safely by masking ratio <= 0).
    - This is the naive (uncorrected) estimator and can be upward biased when spike
      counts are low or sampling is sparse. Consider shuffle controls or
      cross-validation for inference.
    """

    # ---- Determine input type / dimensionality ----
    if isinstance(tcs, pd.Series):
        # Single tuning curve (1 cell)
        tcs_arr = tcs.to_numpy(dtype=float)[:, None]
        is_single = True
        out_index = None

    elif isinstance(tcs, pd.DataFrame):
        # Multiple tuning curves (many cells)
        tcs_arr = tcs.to_numpy(dtype=float)
        is_single = False
        out_index = tcs.columns

    else:
        # NumPy / array-like
        tcs_arr = np.asarray(tcs, dtype=float)

        if tcs_arr.ndim == 1:
            # Single tuning curve
            tcs_arr = tcs_arr[:, None]
            is_single = True
            out_index = None

        elif tcs_arr.ndim == 2:
            # Multiple tuning curves
            is_single = False
            out_index = pd.RangeIndex(tcs_arr.shape[1], name="cell")

        else:
            raise ValueError("tcs must be 1D or 2D: (n_bins,) or (n_bins, n_cells)")

    # ---- Basic shape check ----
    n_bins, n_cells = tcs_arr.shape

    # ---- Occupancy ----
    if occ is None:
        occ_arr = np.full(n_bins, 1.0 / n_bins)
    else:
        # accept Series, array, column vector etc.
        occ_arr = np.asarray(occ, dtype=float).ravel()
        if occ_arr.shape[0] != n_bins:
            raise ValueError(f"occ must have length {n_bins}, got {occ_arr.shape[0]}")
        s = occ_arr.sum()
        if s <= 0:
            raise ValueError("occ must sum to > 0")
        occ_arr = occ_arr / s

    # ---- Skaggs info per spike ----
    f = occ_arr @ tcs_arr                    # mean rate per cell, shape (n_cells,)
    f_safe = np.where(f > 0, f, np.nan)      # avoid divide-by-zero
    ratio = tcs_arr / f_safe                 # λ_i / λ̄

    mask = ratio > 0                         # only where log is defined
    contrib = np.zeros_like(ratio)
    contrib[mask] = (occ_arr[:, None] * ratio * np.log2(ratio))[mask]

    info = np.nansum(contrib, axis=0)        # shape (n_cells,)

    # ---- Return type ----
    if is_single:
        return float(info[0])

    return pd.Series(info, index=out_index, name="spatial_info")




