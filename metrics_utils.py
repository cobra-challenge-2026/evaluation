import logging
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import normalized_mutual_info_score


def setup_logger(name: str = __name__) -> logging.Logger:
    """Return the module logger."""
    return logging.getLogger(name)


def calculate_mae(reference: np.ndarray,
                 prediction: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    """Compute mean absolute error, optionally inside a mask."""
    if mask is not None:
        ref_masked = reference[mask > 0]
        pred_masked = prediction[mask > 0]
    else:
        ref_masked = reference.flatten()
        pred_masked = prediction.flatten()

    if len(ref_masked) == 0:
        return float('nan')

    mae = np.mean(np.abs(ref_masked - pred_masked))
    return float(mae)


def calculate_psnr(reference: np.ndarray,
                  prediction: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  data_range: float = 4095.0,
                  logger: Optional[logging.Logger] = None) -> float:
    """Compute PSNR for the arrays, with an optional mask."""
    if logger is None:
        logger = setup_logger()


    try:
        if mask is not None:
            ref_masked = reference[mask > 0]
            pred_masked = prediction[mask > 0]
        else:
            ref_masked = reference.flatten()
            pred_masked = prediction.flatten()

        if len(ref_masked) == 0:
            return float('nan')

        # When prediction == reference, MSE is exactly 0 and skimage's
        # peak_signal_noise_ratio divides by it, emitting a RuntimeWarning
        # and returning inf. Suppress that specific warning and cap the
        # result at a large sentinel value instead of propagating inf.
        with np.errstate(divide="ignore"):
            psnr = peak_signal_noise_ratio(ref_masked, pred_masked, data_range=data_range)
        if np.isinf(psnr):
            psnr = 99999
        return float(psnr)
    except Exception as e:
        logger.warning(f"PSNR calculation failed: {e}")
        return float('nan')


def calculate_nmi(source: np.ndarray,
                 prediction: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 n_bins: int = 256,
                 logger: Optional[logging.Logger] = None) -> float:
    """Compute normalized mutual information between source and prediction."""
    if logger is None:
        logger = setup_logger()

    try:
        # Apply mask if provided
        if mask is not None:
            src_masked = source[mask > 0]
            pred_masked = prediction[mask > 0]
        else:
            src_masked = source.flatten()
            pred_masked = prediction.flatten()

        if len(src_masked) == 0:
            return float('nan')

        # Discretize continuous CT values into bins for histogram-based MI calculation
        # CT range is typically -1024 to 3071 (4096 values)
        src_binned = np.digitize(src_masked, bins=np.linspace(src_masked.min(), src_masked.max(), n_bins))
        pred_binned = np.digitize(pred_masked, bins=np.linspace(pred_masked.min(), pred_masked.max(), n_bins))

        # Calculate normalized mutual information
        nmi = normalized_mutual_info_score(src_binned, pred_binned, average_method='arithmetic')

        return float(nmi)
    except Exception as e:
        logger.warning(f"NMI calculation failed: {e}")
        return float('nan')


def calculate_cc(source: np.ndarray,
                prediction: np.ndarray,
                mask: Optional[np.ndarray] = None,
                logger: Optional[logging.Logger] = None) -> float:
    """Compute Pearson correlation between source and prediction."""
    if logger is None:
        logger = setup_logger()

    try:
        # Apply mask if provided
        if mask is not None:
            src_masked = source[mask > 0]
            pred_masked = prediction[mask > 0]
        else:
            src_masked = source.flatten()
            pred_masked = prediction.flatten()

        if len(src_masked) == 0:
            logger.warning(f"CC calculation failed: {e}")
            return float('nan')

        # Calculate Pearson correlation coefficient
        # np.corrcoef returns a 2x2 correlation matrix, we want the off-diagonal element
        correlation_matrix = np.corrcoef(src_masked, pred_masked)
        cc = correlation_matrix[0, 1]

        return float(cc)
    except Exception as e:
        logger.warning(f"CC calculation failed: {e}")
        return float('nan')
