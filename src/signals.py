from typing import Self, Union

import numpy as np
import math as m


class BitStream:
    """Represents a bit stream generated at a given frequency"""

    def __init__(self, frequency: float, bits: list[float]):
        """
        :param frequency: clock of the stream
        :param bits: list of bits in the stream
        """
        self.frequency = frequency
        self.bits = np.array(bits)

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):
        self.i += 1
        if self.i >= len(self.bits):
            raise StopIteration
        return self.bits[self.i]

    def __len__(self):
        return len(self.bits)

    def __getitem__(self, index):
        return self.bits[index]

    def encode_nrz(self) -> Self:
        """Encodes the bistream with NRZ (non return to zero)

        :return: encoded bitstream
        """
        bits = [1 if bit == 1 else -1 for bit in self]
        new_stream = BitStream(self.frequency, bits)

        return new_stream

    def to_parallel_streams(self, n: int = 1) -> list[Self]:
        """Transforms the serial bit stream into a list of parallel bit streams

        :param n: number of streams, defaults to 1
        :return: parallel bit streams
        """
        streams = [0] * n
        for i in range(n):
            streams[i] = BitStream(
                self.frequency, [bit for j, bit in enumerate(self) if j % n == i]
            )

        return streams

    def to_signal(self, time_vector: np.ndarray, n: int = 1) -> Union[np.ndarray, list[np.ndarray]]:
        """Generates arrays of values representing the bit stream at its clock frequency, indexed on a time array

        :param time_vector: time array for indexation
        :param n: number of arrays wanted, if more than 1 will split the bit stream into n parallel streams, defaults to 1
        :return: arrays of values, if more than 1 will return a list of arrays
        """
        streams = self.to_parallel_streams(n)
        bit_index = time_vector * self.frequency / n

        bits = [
            np.array(
                [streams[i][m.floor(j)] if m.floor(j) < len(streams[i]) else 0 for j in bit_index]
            )
            for i in range(n)
        ]

        if n == 1:
            return bits[0]

        return bits


class Carrier:
    """Represents a carrier signal of a given frequency and amplitude"""

    def __init__(self, frequency: float, amplitude: float = 1):
        """
        :param frequency: frequency of the carrier
        :param amplitude: amplitude of the carrier, defaults to 1
        """
        self.frequency = frequency
        self.amplitude = amplitude

    def cos(self, time_vector: np.ndarray) -> np.ndarray:
        """Generates an array of values indexed on a time vector, with a cos shaped carrier

        :param time_vector: time array to index values
        :return: the array of values
        """
        return self.amplitude * np.cos(2 * np.pi * self.frequency * time_vector)

    def sin(self, time_vector: np.ndarray) -> np.ndarray:
        """Generates an array of values indexed on a time vector, with a cos shaped carrier

        :param time_vector: time array to index values
        :return: the array of values
        """
        return self.amplitude * np.sin(2 * np.pi * self.frequency * time_vector)

    def frequency_shift(self, time_vector: np.ndarray, bits: np.ndarray = 1, shift: float = 100):
        """Generates an array of values indexed on a time vector, with a cos shaped carrier,
        shifted in frequency relative to a bit array

        :param time_vector: time array to index values
        :param bits: the bit array, if not provided, the frequency shift is constant
        :param shift: the frequency shift in Hertz to apply to a bit with value 1
        :return: the array of values
        """
        return self.amplitude * np.cos(2 * np.pi * (self.frequency + bits * shift) * time_vector)
