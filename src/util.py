import numpy as np
from scipy import special


def Q_function(x: np.ndarray) -> np.ndarray:
    """Computes Q function

    :param x: input_array
    :return: result of Q function applied to input array
    """
    return 0.5 * (1 - special.erf(x / np.sqrt(2)))


def erfc(x: np.ndarray) -> np.ndarray:
    """Complementary of error function

    :param x: input array
    :return: error function of the input array
    """
    return 2 * Q_function(np.sqrt(2) * x)


def add_awgn(signal: np.ndarray, snr: float, k: int = 1) -> np.ndarray:
    """Adds Additive White Gaussian Noise (AWGN) to a signal

    :param signal: input signal
    :param snr: signal to noise ratio (SNR) to add fixed noise to the signal
    :param k: tuning value, defaults to 1
    :return: noisy signal
    """
    power = sum([abs(x) ** 2 for x in signal]) / signal.size
    snr_linear = 10 ** (snr / 10)

    noise = np.sqrt(power / (2 * k * snr_linear)) * np.random.normal(size=signal.size)
    return signal + noise
