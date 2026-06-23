"""
src/conformal/crc.py

Conformal Risk Control (CRC) cho clinical metrics.

Theory (Angelopoulos et al., ICLR 2024):
    Thay vi control coverage (nhu Split CP),
    CRC control mot ham loss tong quat L(lambda):

        E[L(lambda)] <= alpha

    Bang cach tim lambda nho nhat thoa man dieu kien tren
    tren calibration set.

    Voi clinical metrics:
        lambda = do rong interval
        L(lambda) = 1{y_true NOT in [y_pred - lambda, y_pred + lambda]}
                  = miscoverage indicator

    => CRC voi loss nay tuong duong Split CP nhung framework tong quat hon.
"""

import numpy as np
from typing import Callable


def risk_fn_miscoverage(true: np.ndarray,
                        pred: np.ndarray,
                        lam: float,
                        mode: str = 'abs') -> float:
    """
    Loss function: ty le miscoverage (khong nam trong interval).

    L(lambda) = P(|true - pred| > lambda)

    Args:
        true : true metrics, shape (n,)
        pred : predicted metrics, shape (n,)
        lam  : interval half-width (lambda)
        mode : 'abs' hoac 'rel'

    Returns:
        risk: ty le miscoverage (0-1)
    """
    from src.metrics.clinical_metrics import nonconformity_score
    scores = nonconformity_score(pred, true, mode=mode)
    return float(np.mean(scores > lam))


def find_lambda(cal_true: np.ndarray,
                cal_pred: np.ndarray,
                alpha: float,
                mode: str = 'abs',
                n_grid: int = 1000) -> dict:
    """
    Tim lambda nho nhat sao cho risk(lambda) <= alpha.

    Cach lam:
        1. Tao grid cac gia tri lambda tu 0 den max_score
        2. Voi moi lambda, tinh risk tren calibration set
        3. Chon lambda nho nhat thoa man risk <= alpha

    Nay chinh la CRC procedure (Angelopoulos et al. 2024).

    Args:
        cal_true : true metrics, shape (n,)
        cal_pred : predicted metrics, shape (n,)
        alpha    : risk level (vi du 0.1)
        mode     : 'abs' hoac 'rel'
        n_grid   : so diem trong grid tim kiem

    Returns:
        dict voi:
            'lambda'   : gia tri lambda tim duoc
            'risk'     : risk tai lambda nay
            'alpha'    : alpha duoc su dung
    """
    from src.metrics.clinical_metrics import nonconformity_score

    scores  = nonconformity_score(cal_pred, cal_true, mode=mode)
    max_lam = float(np.max(scores)) * 1.1

    # Grid cac gia tri lambda
    lambdas = np.linspace(0, max_lam, n_grid)

    # CRC finite-sample correction (Angelopoulos et al. 2024):
    # R_hat(lambda) = (1/(n+1)) * (sum_{i=1}^{n} L_i(lambda) + 1)
    n = len(scores)
    corrected_alpha = alpha * n / (n + 1)  # BH-style correction

    best_lam = max_lam  # default: conservative
    for lam in lambdas:
        risk = float(np.mean(scores > lam))
        if risk <= corrected_alpha:
            best_lam = lam
            break  # Tim thay lambda nho nhat thoa man

    final_risk = float(np.mean(scores > best_lam))

    return {
        'lambda' : best_lam,
        'risk'   : final_risk,
        'alpha'  : alpha,
        'n_cal'  : n,
        'mode'   : mode,
    }


def predict_interval_crc(test_pred: np.ndarray,
                          lam: float,
                          mode: str = 'abs'):
    """
    Tao prediction intervals dung lambda tu CRC.
    Interface tuong tu split_conformal.predict_interval().
    """
    from src.conformal.split_conformal import predict_interval
    return predict_interval(test_pred, lam, mode=mode)
