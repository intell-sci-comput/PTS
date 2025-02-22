# pylint: disable=unused-variable

import numpy as np
import scipy

from utils.train import acc_tau


def remove_nans(y_test, y_hat_test):
    y_test_r = []
    y_hat_test_r = []
    for i in zip(y_test, y_hat_test):
        if not np.isnan(i[0]) and not np.isnan(i[1]):
            y_test_r.append(i[0])
            y_hat_test_r.append(i[1])
    y_test_r = np.array(y_test_r)
    y_hat_test_r = np.array(y_hat_test_r)
    return y_test_r, y_hat_test_r


def test_metrics():
    y_test = np.array([1, 2, 3])
    y_hat_test = np.array([1, 2, 3])
    r2 = scipy.stats.pearsonr(y_test, y_hat_test)[0]
    acc_iid = acc_tau(y_test, y_hat_test)
    y_test = np.array([1, np.nan, 3, 4])
    y_hat_test = np.array([1, 2, np.nan, 4])
    y_test, y_hat_test = remove_nans(y_test, y_hat_test)
    if y_test.size > 1:
        r2 = scipy.stats.pearsonr(y_test, y_hat_test)[0]
    else:
        r2 = 0
    acc_iid = acc_tau(y_test, y_hat_test)
    y_test = np.array([np.nan])
    y_hat_test = np.array([np.nan])
    y_test, y_hat_test = remove_nans(y_test, y_hat_test)
    if y_test.size > 1:
        r2 = scipy.stats.pearsonr(y_test, y_hat_test)[0]
    else:
        r2 = 0  # noqa: F841
    acc_iid = acc_tau(y_test, y_hat_test)  # noqa: F841


if __name__ == "__main__":
    test_metrics()
