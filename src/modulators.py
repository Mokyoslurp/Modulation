import numpy as np
import math as m

from abc import ABC, abstractmethod

from .util import Q_function, add_awgn, erfc
from .signals import BitStream, Carrier


class AbstractModulator(ABC):
    """Base modulator to modulate and demodulate signal"""

    def __init__(self):
        self.name: str = None
        self.bits_per_symbol: int = None
        # Low pass filter frequency
        self.lpf_frequency: float = None

    @abstractmethod
    def modulate(
        self, bit_stream: BitStream, carrier: Carrier, time_vector: np.ndarray
    ) -> np.ndarray:
        """Modulates a bit stream into a time signal

        :param bit_stream: input bit stream
        :param carrier: carrier to modulate the signal on
        :param time_vector: array of time values to index the signal
        :return: the modulated signal
        """

    @abstractmethod
    def demodulate(
        self, signal: np.ndarray, carrier: Carrier, time_vector: np.ndarray
    ) -> list[np.ndarray]:
        """Demodulates a time signal into demodulated bit signals needing bit identification to rebuild the bi stream

        :param signal: input time signal
        :param carrier: carrier used to modulate the signal
        :param time_vector: array of time values to index the signal
        :return: the output demodulated signals
        """

    @abstractmethod
    def bit_identification(
        self, demodulated_signals: list[np.ndarray], bit_clock: float, time_vector: np.ndarray
    ) -> BitStream:
        """Identifies bits in demodulated signals

        :param demodulated_signals: previously demodulated signals
        :param time_vector: reference time array
        :param bit_clock: frequency of the original bit stream
        :return: output bit stream
        """

    @abstractmethod
    def theoretical_BER(self, EbN0s: list[float]) -> np.ndarray:
        """Calculates the theoretical values of bit error rate (BER) for this modulation scheme

        :param EbN0s: bit energy on noise ratios to compute BER
        :return: the theoretical BERs
        """

    @abstractmethod
    def fast_modulation_demodulation(self, bit_stream: BitStream, snr: float) -> BitStream:
        """Completes a modulation-demodulation without using the carrier for faster BER computation

        :param bit_stream: input bit stream to use
        :param snr: signal to noise ratio used
        :return: output bit stream after demodulation
        """

    def _coherent_demodulation(
        self, signal: np.ndarray, carrier_1: np.ndarray, carrier_2: np.ndarray
    ):
        samples = round(signal.size / self.lpf_frequency)
        box = 2 * np.ones(samples) / samples

        demodulated_signal_1 = np.convolve(signal * carrier_1, box, "same")
        demodulated_signal_2 = np.convolve(signal * carrier_2, box, "same")

        return [demodulated_signal_1, demodulated_signal_2]

    def get_demodulated_signals_average(
        self, demodulated_signals: list[np.ndarray], bit_clock: float, time_vector: np.ndarray
    ):
        symbol_index = time_vector * bit_clock / self.bits_per_symbol

        n_symbols = m.floor(symbol_index[-1])
        n_bits = self.bits_per_symbol * n_symbols

        bit_length = round(time_vector.size / symbol_index[-1])

        average_1 = [0] * n_bits
        average_2 = [0] * n_bits

        for i in range(n_symbols):
            average_1[i] = (
                np.sum(demodulated_signals[0][i * bit_length : (i + 1) * bit_length]) / bit_length
            )
            average_2[i] = (
                np.sum(demodulated_signals[1][i * bit_length : (i + 1) * bit_length]) / bit_length
            )

        return np.array(average_1), np.array(average_2)


class BPSKModulator(AbstractModulator):
    def __init__(self, low_pass_filter_frequency: int = 100):
        """
        :param low_pass_filter_frequency: frequency of the low pass filter used after
            matched filter, defaults to 100
        """
        super().__init__()
        self.name = "BSPK"
        self.bits_per_symbol = 1
        self.lpf_frequency = low_pass_filter_frequency

    def modulate(self, bit_stream, carrier, time_vector):
        bit_signal = bit_stream.encode_nrz().to_signal(time_vector)
        cos_carrier = carrier.cos(time_vector)

        modulated_signal = bit_signal * cos_carrier

        return modulated_signal

    def demodulate(self, signal, carrier, time_vector):
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        return self._coherent_demodulation(signal, cos_carrier, sin_carrier)

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        averages = self.get_demodulated_signals_average(demodulated_signals, bit_clock, time_vector)

        bits = np.zeros((averages[0].shape))
        bits[averages[0] > 0] = 1

        return BitStream(bit_clock, bits)

    def theoretical_BER(self, EbN0s):
        return Q_function(np.sqrt(2 * (10 ** (np.array(EbN0s) / 10))))

    def fast_modulation_demodulation(self, bit_stream, snr):
        bit_signal = bit_stream.encode_nrz()
        noisy_signal = add_awgn(np.array(bit_signal.bits), snr)

        output_bits = [0] * len(bit_signal)
        for i, bit in enumerate(noisy_signal):
            if bit > 0:
                output_bits[i] = 1
            else:
                output_bits[i] = 0

        return BitStream(bit_stream.frequency, output_bits)


class QPSKModulator(AbstractModulator):
    def __init__(self, low_pass_filter_frequency: int = 100):
        """
        :param low_pass_filter_frequency: frequency of the low pass filter used after
            matched filter, defaults to 100
        """
        super().__init__()
        self.name = "QPSK"
        self.bits_per_symbol = 2
        self.lpf_frequency = low_pass_filter_frequency

    def modulate(self, bit_stream, carrier, time_vector):
        bit_signal = bit_stream.encode_nrz().to_signal(time_vector, 2)
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        modulated_signal = bit_signal[0] * cos_carrier + bit_signal[1] * sin_carrier

        return modulated_signal

    def demodulate(self, signal, carrier, time_vector):
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        return self._coherent_demodulation(signal, cos_carrier, sin_carrier)

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        averages = self.get_demodulated_signals_average(demodulated_signals, bit_clock, time_vector)

        bits_1 = np.zeros((averages[0].shape))
        bits_2 = np.zeros((averages[0].shape))

        bits_1[averages[0] > 0] = 1
        bits_2[averages[1] > 0] = 1

        bits = np.ravel([bits_1, bits_2], order="F")

        return BitStream(bit_clock, bits)

    def theoretical_BER(self, EbN0s):
        return Q_function(np.sqrt(2 * (10 ** (np.array(EbN0s) / 10))))

    def fast_modulation_demodulation(self, bit_stream, snr):
        bit_signals = bit_stream.encode_nrz().to_parallel_streams(2)

        noisy_signal_1 = add_awgn(np.array(bit_signals[0].bits), snr)
        noisy_signal_2 = add_awgn(np.array(bit_signals[1].bits), snr)

        output_bits = [0] * len(bit_stream)
        for i, bit in enumerate(noisy_signal_1):
            if bit > 0:
                output_bits[2 * i] = 1
            else:
                output_bits[2 * i] = 0
        for i, bit in enumerate(noisy_signal_2):
            if bit > 0:
                output_bits[2 * i + 1] = 1
            else:
                output_bits[2 * i + 1] = 0

        return BitStream(bit_stream.frequency, output_bits)


class RotatedEightPSKModulator(AbstractModulator):
    def __init__(self, low_pass_filter_frequency: int = 100):
        """
        :param low_pass_filter_frequency: frequency of the low pass filter used after
            matched filter, defaults to 100
        """
        super().__init__()
        self.name = "8PSK"
        self.bits_per_symbol = 3
        self.lpf_frequency = low_pass_filter_frequency

    def modulate(self, bit_stream, carrier, time_vector):
        bit_signal = bit_stream.encode_nrz().to_signal(time_vector, 3)
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        I_signal = (-0.5 * bit_signal[1]) * (
            (bit_signal[2] + 1) * np.sin(np.pi / 8) - (bit_signal[2] - 1) * np.cos(np.pi / 8)
        )
        Q_signal = (-0.5 * bit_signal[0]) * (
            (bit_signal[2] + 1) * np.cos(np.pi / 8) - (bit_signal[2] - 1) * np.sin(np.pi / 8)
        )

        modulated_signal = I_signal * cos_carrier + Q_signal * sin_carrier
        return modulated_signal

    def demodulate(self, signal, carrier, time_vector):
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        return self._coherent_demodulation(signal, cos_carrier, sin_carrier)

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        averages = self.get_demodulated_signals_average(demodulated_signals, bit_clock, time_vector)

        bits_1 = np.zeros((averages[0].shape))
        bits_2 = np.zeros((averages[0].shape))
        bits_3 = np.zeros((averages[0].shape))

        bits_1[averages[1] < 0] = 1
        bits_2[averages[0] < 0] = 1
        bits_3[np.abs(averages[0]) - np.abs(averages[1]) < 0] = 1

        bits = np.ravel([bits_1, bits_2, bits_3], order="F")

        return BitStream(bit_clock, bits)

    def theoretical_BER(self, EbN0s):
        return 0.5 * Q_function(np.sqrt(2 * (10 ** (np.array(EbN0s) / 10)) * np.sin(np.pi / 8)))

    def fast_modulation_demodulation(self, bit_stream, snr): ...


class SixteenQAMModulator(AbstractModulator):
    def __init__(self, low_pass_filter_frequency: int = 100, threshold: float = 2):
        """
        :param low_pass_filter_frequency: frequency of the low pass filter used after
            matched filter, defaults to 100
        :param threshold: threshold to use to differentiate the 2 values of amplitude
        """
        super().__init__()
        self.name = "16QAM"
        self.bits_per_symbol = 4
        self.lpf_frequency = low_pass_filter_frequency
        self.threshold = threshold

    def modulate(self, bit_stream, carrier, time_vector):
        bit_signal = bit_stream.encode_nrz().to_signal(time_vector, 4)
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        modulated_signal = (
            bit_signal[0] * (2 - bit_signal[1]) * cos_carrier
            - bit_signal[2] * (2 - bit_signal[3]) * sin_carrier
        )

        return modulated_signal

    def demodulate(self, signal, carrier, time_vector):
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        return self._coherent_demodulation(signal, cos_carrier, sin_carrier)

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        averages = self.get_demodulated_signals_average(demodulated_signals, bit_clock, time_vector)

        bits_1 = np.zeros((averages[0].shape))
        bits_2 = np.zeros((averages[0].shape))
        bits_3 = np.zeros((averages[0].shape))
        bits_4 = np.zeros((averages[0].shape))

        bits_1[averages[0] > 0] = 1
        bits_2[np.abs(averages[0]) < self.threshold] = 1
        bits_3[averages[1] < 0] = 1
        bits_4[np.abs(averages[1]) < self.threshold] = 1

        bits = np.ravel([bits_1, bits_2, bits_3, bits_4], order="F")

        return BitStream(bit_clock, bits)

    def theoretical_BER(self, EbN0s):
        return (3 / (2 * 4)) * erfc(np.sqrt((10 ** (np.array(EbN0s) / 10)) * (4 / 10)))

    def fast_modulation_demodulation(self, bit_stream, snr):
        bit_signals = bit_stream.encode_nrz().to_parallel_streams(4)

        signal_1 = np.array(bit_signals[0].bits) * (2 - np.array(bit_signals[1].bits))
        signal_2 = -np.array(bit_signals[2].bits) * (2 - np.array(bit_signals[3].bits))

        noisy_signal_1 = add_awgn(signal_1, snr, 2)
        noisy_signal_2 = add_awgn(signal_2, snr, 2)

        output_bits = [0] * len(bit_stream)
        for i, bit in enumerate(noisy_signal_1):
            if bit > 0:
                output_bits[4 * i] = 1
            else:
                output_bits[4 * i] = 0

            if abs(bit) > self.threshold:
                output_bits[4 * i + 1] = 0
            else:
                output_bits[4 * i + 1] = 1

        for i, bit in enumerate(noisy_signal_2):
            if bit > 0:
                output_bits[4 * i + 2] = 0
            else:
                output_bits[4 * i + 2] = 1

            if abs(bit) > self.threshold:
                output_bits[4 * i + 3] = 0
            else:
                output_bits[4 * i + 3] = 1

        return BitStream(bit_stream.frequency, output_bits)


class ThirtytwoQAMModulator(AbstractModulator):
    def __init__(
        self, low_pass_filter_frequency: int = 100, threshold_1: float = 2, threshold_2: float = 4
    ):
        """
        :param low_pass_filter_frequency: frequency of the low pass filter used after
            matched filter, defaults to 100
        :param threshold: threshold to use to differentiate the 2 values of amplitude
        """
        super().__init__()
        self.name = "32QAM"
        self.bits_per_symbol = 5
        self.lpf_frequency = low_pass_filter_frequency
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2

    def modulate(self, bit_stream, carrier, time_vector):
        bit_signal = bit_stream.encode_nrz().to_signal(time_vector, 5)
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        external_symbols = 0.25 * (bit_signal[0] + 1) * (bit_signal[3] - 1) * (bit_signal[4] - 1)
        I_signal = bit_signal[1] * (
            2
            + bit_signal[4]
            + 0.5 * (bit_signal[0] + 1) * (bit_signal[4] + 1)
            + 1 * external_symbols
        )
        Q_signal = bit_signal[2] * (
            2
            + bit_signal[3]
            - 0.25 * (bit_signal[0] + 1) * (bit_signal[3] + 1) * (bit_signal[4] - 1)
            + 2 * external_symbols
        )

        modulated_signal = I_signal * cos_carrier + Q_signal * sin_carrier
        return modulated_signal

    def demodulate(self, signal, carrier, time_vector):
        cos_carrier = carrier.cos(time_vector)
        sin_carrier = carrier.sin(time_vector)

        return self._coherent_demodulation(signal, cos_carrier, sin_carrier)

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        averages = self.get_demodulated_signals_average(demodulated_signals, bit_clock, time_vector)

        bits_1 = np.zeros((averages[0].shape))
        bits_2 = np.zeros((averages[0].shape))
        bits_3 = np.zeros((averages[0].shape))
        bits_4 = np.zeros((averages[0].shape))
        bits_5 = np.zeros((averages[0].shape))

        bits_1[
            np.any(
                [np.abs(averages[0]) > self.threshold_2, np.abs(averages[1]) > self.threshold_2],
                axis=0,
            )
        ] = 1
        bits_2[averages[0] > 0] = 1
        bits_3[averages[1] > 0] = 1
        bits_4[
            np.all(
                [
                    np.abs(averages[1]) > self.threshold_1,
                    np.any(
                        [
                            np.abs(averages[1]) < self.threshold_2,
                            np.abs(averages[0]) < self.threshold_1,
                        ],
                        axis=0,
                    ),
                ],
                axis=0,
            )
        ] = 1
        bits_5[
            np.all(
                [np.abs(averages[0]) > self.threshold_1, np.abs(averages[1]) < self.threshold_2],
                axis=0,
            )
        ] = 1

        bits = np.ravel([bits_1, bits_2, bits_3, bits_4, bits_5], order="F")

        return BitStream(bit_clock, bits)

    def theoretical_BER(self, EbN0s):
        return ((4 / 5) * (1 - (1 / np.sqrt(32)))) * Q_function(
            np.sqrt((10 ** (np.array(EbN0s) / 10)) * (3 * 5 / 31))
        )

    def fast_modulation_demodulation(self, bit_stream, snr): ...


class BFSKModulator(AbstractModulator):
    def __init__(
        self,
        low_pass_filter_frequency: int = 100,
        frequency_shift: float = 100,
    ):
        """
        :param low_pass_filter_frequency: frequency of the low pass filter used after
            matched filter, defaults to 100
        :param frequency_shift: frequency shift to use for the modulation. The carrier of
            frequency f will be modulated into f - shift and f + shift
        """
        super().__init__()
        self.name = "BFSK"
        self.bits_per_symbol = 1
        self.frequency_shift = frequency_shift
        self.lpf_frequency = low_pass_filter_frequency

    def modulate(self, bit_stream, carrier, time_vector):
        bit_signal = bit_stream.encode_nrz().to_signal(time_vector)

        modulated_signal = carrier.frequency_shift(
            time_vector, bit_signal, shift=self.frequency_shift
        )

        return modulated_signal

    def demodulate(self, signal, carrier, time_vector):
        carrier_1 = carrier.frequency_shift(time_vector, shift=self.frequency_shift)
        carrier_2 = carrier.frequency_shift(time_vector, shift=-self.frequency_shift)

        return self._coherent_demodulation(signal, carrier_1, carrier_2)

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        averages = self.get_demodulated_signals_average(demodulated_signals, bit_clock, time_vector)

        bits = np.zeros((averages[0].shape))
        bits[averages[0] - averages[1] > 0] = 1

        return BitStream(bit_clock, bits)

    def theoretical_BER(self, EbN0s):
        return 0.5 * erfc(np.sqrt((10 ** (np.array(EbN0s) / 10)) / 2))

    def fast_modulation_demodulation(self, bit_stream, snr):
        bit_signal = bit_stream.encode_nrz()

        noisy_signal_1 = add_awgn(np.array(bit_signal.bits), snr)
        noisy_signal_2 = add_awgn(-np.array(bit_signal.bits), snr)

        noisy_signal = (noisy_signal_1 - noisy_signal_2) / 2

        output_bits = [0] * len(bit_stream)
        for i, bit in enumerate(noisy_signal):
            if bit > 0:
                output_bits[i] = 1
            else:
                output_bits[i] = 0

        return BitStream(bit_stream.frequency, output_bits)
