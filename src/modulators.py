import numpy as np
import math as m

from abc import ABC, abstractmethod

from .util import Q_function, add_awgn, erfc
from .signals import BitStream, Carrier


class AbstractModulator(ABC):
    """Base modulator to modulate and demodulate signal"""

    def __init__(self):
        self.name: str = None

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


class BPSKModulator(AbstractModulator):
    def __init__(self, low_pass_filter_frequency: int = 100):
        """
        :param low_pass_filter_frequency: frequency of the low pass filter used after
            matched filter, defaults to 100
        """
        super().__init__()
        self.name = "BSPK"
        self.lpf_frequency = low_pass_filter_frequency

    def modulate(self, bit_stream, carrier, time_vector):
        bit_signal = bit_stream.encode_nrz().to_signal(time_vector)
        cos_carrier = carrier.cos(time_vector)

        modulated_signal = bit_signal * cos_carrier

        return modulated_signal

    def demodulate(self, signal, carrier, time_vector):
        cos_carrier = carrier.cos(time_vector)

        samples = round(signal.size / self.lpf_frequency)
        box = 2 * np.ones(samples) / samples

        demodulated_signal = np.convolve(signal * cos_carrier, box, "same")

        return [demodulated_signal]

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        bit_index = time_vector * bit_clock
        bit_length = round(time_vector.size / bit_index[-1])
        bits = [0] * m.floor(bit_index[-1])

        for i in range(m.floor(bit_index[-1])):
            value_sum = (
                np.sum(demodulated_signals[0][i * bit_length : (i + 1) * bit_length]) / bit_length
            )
            if value_sum > 0:
                bits[i] = 1
            else:
                bits[i] = 0

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

        samples = round(signal.size / self.lpf_frequency)
        box = 2 * np.ones(samples) / samples

        demodulated_signal_1 = np.convolve(signal * cos_carrier, box, "same")
        demodulated_signal_2 = np.convolve(signal * sin_carrier, box, "same")

        return [demodulated_signal_1, demodulated_signal_2]

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        bit_index = time_vector * bit_clock / 2
        bit_length = round(time_vector.size / bit_index[-1])

        bits = [0] * 2 * m.floor(bit_index[-1])

        for i in range(m.floor(bit_index[-1])):
            value_sum_1 = (
                np.sum(demodulated_signals[0][i * bit_length : (i + 1) * bit_length]) / bit_length
            )
            value_sum_2 = (
                np.sum(demodulated_signals[1][i * bit_length : (i + 1) * bit_length]) / bit_length
            )

            if value_sum_1 > 0:
                if value_sum_2 > 0:
                    bits[2 * i] = 1
                    bits[2 * i + 1] = 1
                else:
                    bits[2 * i] = 1
                    bits[2 * i + 1] = 0
            else:
                if value_sum_2 > 0:
                    bits[2 * i] = 0
                    bits[2 * i + 1] = 1
                else:
                    bits[2 * i] = 0
                    bits[2 * i + 1] = 0

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

        samples = round(signal.size / self.lpf_frequency)
        box = 2 * np.ones(samples) / samples

        demodulated_signal_1 = np.convolve(signal * cos_carrier, box, "same")
        demodulated_signal_2 = np.convolve(signal * sin_carrier, box, "same")

        return [demodulated_signal_1, demodulated_signal_2]

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        bit_index = time_vector * bit_clock / 3
        bit_length = round(time_vector.size / bit_index[-1])

        bits = [0] * 3 * m.floor(bit_index[-1])

        for i in range(m.floor(bit_index[-1])):
            value_sum_1 = (
                np.sum(demodulated_signals[0][i * bit_length : (i + 1) * bit_length]) / bit_length
            )
            value_sum_2 = (
                np.sum(demodulated_signals[1][i * bit_length : (i + 1) * bit_length]) / bit_length
            )

            if value_sum_2 > 0:
                bits[3 * i] = 0
            else:
                bits[3 * i] = 1

            if value_sum_1 > 0:
                bits[3 * i + 1] = 0
            else:
                bits[3 * i + 1] = 1

            if abs(value_sum_1) - abs(value_sum_2) < 0:
                bits[3 * i + 2] = 1
            else:
                bits[3 * i + 2] = 0

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

        samples = round(signal.size / self.lpf_frequency)
        box = 2 * np.ones(samples) / samples

        demodulated_signal_1 = np.convolve(signal * cos_carrier, box, "same")
        demodulated_signal_2 = np.convolve(signal * sin_carrier, box, "same")

        return [demodulated_signal_1, demodulated_signal_2]

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        bit_index = time_vector * bit_clock / 4
        bit_length = round(time_vector.size / bit_index[-1])

        bits = [0] * 4 * m.floor(bit_index[-1])

        for i in range(m.floor(bit_index[-1])):
            value_sum_1 = (
                np.sum(demodulated_signals[0][i * bit_length : (i + 1) * bit_length]) / bit_length
            )
            value_sum_2 = (
                np.sum(demodulated_signals[1][i * bit_length : (i + 1) * bit_length]) / bit_length
            )

            if value_sum_1 > 0:
                bits[4 * i] = 1
            else:
                bits[4 * i] = 0

            if abs(value_sum_1) > self.threshold:
                bits[4 * i + 1] = 0
            else:
                bits[4 * i + 1] = 1

            if value_sum_2 > 0:
                bits[4 * i + 2] = 0
            else:
                bits[4 * i + 2] = 1

            if abs(value_sum_2) > self.threshold:
                bits[4 * i + 3] = 0
            else:
                bits[4 * i + 3] = 1

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

        samples = round(signal.size / self.lpf_frequency)
        box = 2 * np.ones(samples) / samples

        demodulated_signal_1 = np.convolve(signal * cos_carrier, box, "same")
        demodulated_signal_2 = np.convolve(signal * sin_carrier, box, "same")

        return [demodulated_signal_1, demodulated_signal_2]

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        bit_index = time_vector * bit_clock / 5
        bit_length = round(time_vector.size / bit_index[-1])

        bits = [0] * 5 * m.floor(bit_index[-1])

        for i in range(m.floor(bit_index[-1])):
            value_sum_1 = (
                np.sum(demodulated_signals[0][i * bit_length : (i + 1) * bit_length]) / bit_length
            )
            value_sum_2 = (
                np.sum(demodulated_signals[1][i * bit_length : (i + 1) * bit_length]) / bit_length
            )

            if abs(value_sum_1) > self.threshold_2 or abs(value_sum_2) > self.threshold_2:
                bits[5 * i] = 1
            else:
                bits[5 * i] = 0

            if value_sum_1 > 0:
                bits[5 * i + 1] = 1
            else:
                bits[5 * i + 1] = 0

            if value_sum_2 > 0:
                bits[5 * i + 2] = 1
            else:
                bits[5 * i + 2] = 0

            if abs(value_sum_2) > self.threshold_1 and (
                abs(value_sum_2) < self.threshold_2 or abs(value_sum_1) < self.threshold_1
            ):
                bits[5 * i + 3] = 1
            else:
                bits[5 * i + 3] = 0

            if abs(value_sum_1) > self.threshold_1 and abs(value_sum_2) < self.threshold_2:
                bits[5 * i + 4] = 1
            else:
                bits[5 * i + 4] = 0

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

        samples = round(signal.size / self.lpf_frequency)
        box = 2 * np.ones(samples) / samples

        demodulated_signal_1 = np.convolve(signal * carrier_1, box, "same")
        demodulated_signal_2 = np.convolve(signal * carrier_2, box, "same")

        return [demodulated_signal_1, demodulated_signal_2]

    def bit_identification(self, demodulated_signals, bit_clock, time_vector):
        bit_index = time_vector * bit_clock
        bit_length = round(time_vector.size / bit_index[-1])
        bits = [0] * m.floor(bit_index[-1])

        for i in range(m.floor(bit_index[-1])):
            value_sum = (
                np.sum(demodulated_signals[0][i * bit_length : (i + 1) * bit_length])
                - np.sum(demodulated_signals[1][i * bit_length : (i + 1) * bit_length])
            ) / (bit_length)

            if value_sum > 0:
                bits[i] = 1
            else:
                bits[i] = 0

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
