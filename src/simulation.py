import numpy as np
import matplotlib.pyplot as plt

from .signals import BitStream, Carrier
from .modulators import AbstractModulator, add_awgn


class SimulationParameters:
    """Parameters of a modulation simulation to pass to simulations for consistency"""

    def __init__(
        self,
        sampling_frequency: float,
        t_max: float,
        carrier: Carrier,
        bit_stream: BitStream,
    ):
        """
        :param sampling_frequency: the sampling frequency of the simulation
        :param t_max: end time of the simulation
        :param carrier: carrier to consider for all modulations
        :param bit_stream: input bit stream to consider for all modulations
        """
        self.sampling_frequency = sampling_frequency
        self.time_vector = np.linspace(0, t_max, round(t_max * sampling_frequency))

        self.input_bit_stream = bit_stream
        self.carrier = carrier


class ModulationSimulation:
    """Simulation of a complete modulation and demodulation to analyse the signals"""

    def __init__(
        self,
        parameters: SimulationParameters,
        modulator: AbstractModulator,
        snr: float = 5.0,
    ):
        """
        :param parameters: generic simulation parameters to use
        :param modulator: modulator to use
        :param snr: signal to noise ratio to apply after modulation, defaults to 5.0
        """
        # References to the parameters to make them easier to use
        self.time_vector = parameters.time_vector
        self.input_bit_stream = parameters.input_bit_stream
        self.carrier = parameters.carrier

        self.modulator = modulator
        self.snr = snr

        self.modulated_signal: np.ndarray = None
        self.demodulated_signals: np.ndarray = None
        self.output_bit_stream: BitStream = None

    def clean(self):
        """Restores signals to None so that the simulation can be run again after modifications"""
        self.modulated_signal = None
        self.demodulated_signals = None
        self.output_bit_stream = None

    def modulate(self):
        """Modulates the bit stream and adds noise to it
        Only runs if it hasn't yet
        """
        if self.modulated_signal is None:
            modulated_signal = self.modulator.modulate(
                self.input_bit_stream, self.carrier, self.time_vector
            )

            self.modulated_signal = add_awgn(modulated_signal, self.snr)

    def demodulate(self):
        """Demodulates the modulated signal into one or many demodulated signals
        Only runs if it hasn't yet
        """
        if self.demodulated_signals is None:
            self.demodulated_signals = self.modulator.demodulate(
                self.modulated_signal, self.carrier, self.time_vector
            )

    def bit_identification(self):
        """Reconstitutes the bit stream using the previously demodulated signals
        Only runs if it hasn't yet
        """
        if self.output_bit_stream is None:
            self.output_bit_stream = self.modulator.bit_identification(
                self.demodulated_signals, self.input_bit_stream.frequency, self.time_vector
            )

    def get_bit_error_rate(self):
        """Calculates the bit error rate (BER) from a complete modulation-demodulation

        :return: _description_
        """
        if self.output_bit_stream is not None:
            difference = np.sum(
                abs(
                    self.input_bit_stream.bits[: len(self.output_bit_stream)]
                    - self.output_bit_stream.bits
                )
            )
            return difference / len(self.output_bit_stream)

    def run(self, with_plots=True):
        """Run the full simulation and plots the results if asked

        :param with_plots: if True plots the results after doing the modulation-demodulation
            , defaults to True
        """
        self.modulate()
        self.demodulate()
        self.bit_identification()

        if with_plots:
            self.plot_all()

    def plot_bit_stream(self):
        """Plots the input and output bit streams for comparison"""
        if self.output_bit_stream is not None:
            plt.figure(0)
            plt.plot(
                self.time_vector,
                self.input_bit_stream.to_signal(self.time_vector),
                label="Input bits",
            )
            plt.plot(
                self.time_vector,
                self.output_bit_stream.to_signal(self.time_vector),
                label="Output bits",
            )

            plt.xlabel("Bit")
            plt.ylabel("Amplitude")
            plt.legend()

    def plot_modulation(self):
        """Plots the carrier and the signal modulated on the carrier"""
        if self.modulated_signal is not None:
            plt.figure(1)
            plt.plot(self.time_vector, self.carrier.cos(self.time_vector), label="Carrier")
            plt.plot(self.time_vector, self.modulated_signal, label="Modulated signal")

            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()

    def plot_demodulation(self):
        """Plots the demodulated signal(s) together with the input bit stream for comparison"""
        if self.demodulated_signals is not None:
            plt.figure(2)
            plt.plot(
                self.time_vector,
                self.input_bit_stream.to_signal(self.time_vector),
                label="Input bits",
            )
            for i, signal in enumerate(self.demodulated_signals):
                plt.plot(self.time_vector, signal, label=f"Demodulated signal {i}")

            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()

    def plot_all(self):
        """Plots and shows all implemented plots"""
        self.plot_bit_stream()
        self.plot_modulation()
        self.plot_demodulation()

        plt.show()


class BERSimulation:
    """A simulation to measure bit error rate (BER) of the desired modulation schemes
    and compare them and their theoretical values
    """

    def __init__(
        self,
        parameters: SimulationParameters,
        modulators: list[AbstractModulator],
        min: float = -5,
        max: float = 10,
        n: int = 10,
    ):
        """
        :param parameters: simulation parameters to use
        :param modulators: a list of modulators to compare
        :param min: minimum value for bit energy to noise ratio to test, defaults to -5
        :param max: maximum value for bit energy to noise ratio to test, defaults to 10
        :param n: number of values for bit energy to noise ratio to test, defaults to 10
        """
        self.parameters = parameters
        self.modulators = modulators

        # Bit energy Eb to noise ratio N0
        self.EbN0s = [min + (i / (n - 1)) * (max - min) for i in range(n)]

        # Bit error rates for all modulators
        self.BERs: dict[AbstractModulator, list[float]] = {}

    def run(self, full_modulation=False):
        """Runs the simulation. Can run either the full modulation scheme or the simplified
            one without carrier


        :param full_modulation: if True, uses the full modulation scheme, defaults to False
        """
        for i, modulator in enumerate(self.modulators):
            # Modulation simulation initialization with a 0 signal to noise ratio (SNR)
            self.BERs[modulator] = []
            modulation_simulation = ModulationSimulation(
                self.parameters,
                modulator,
                0,
            )

            for EbN0 in self.EbN0s:
                if full_modulation:
                    # If full modulation scheme is used, the SNR has to be adjusted taking
                    # into account sampling and bit clock to get same noise effect
                    snr = EbN0 - 10 * np.log10(
                        self.parameters.sampling_frequency
                        / self.parameters.input_bit_stream.frequency
                    )

                    modulation_simulation.clean()
                    modulation_simulation.snr = snr

                    modulation_simulation.run(with_plots=False)
                    ber = modulation_simulation.get_bit_error_rate()

                else:
                    snr = EbN0

                    output_bit_stream = modulator.fast_modulation_demodulation(
                        self.parameters.input_bit_stream, snr
                    )

                    ber = sum(
                        input_bit != output_bit
                        for input_bit, output_bit in zip(
                            self.parameters.input_bit_stream.bits, output_bit_stream.bits
                        )
                    ) / len(output_bit_stream)

                print(f"{modulator.name}: SNR = {snr:.2f}, BER = {ber}")
                self.BERs[modulator].append(ber)

    def plot(self):
        """Plots the simulation results, both theoretical BERs and simulated ones"""
        fig = plt.figure(4)
        ax = fig.add_subplot(1, 1, 1)
        for modulator in self.modulators:
            ax.plot(
                self.EbN0s,
                modulator.theoretical_BER(self.EbN0s),
                label=f"Theoretical {modulator.name}",
            )
            ax.plot(self.EbN0s, self.BERs[modulator], label=modulator.name)

        ax.set_yscale("log")
        ax.legend()
        plt.show()
