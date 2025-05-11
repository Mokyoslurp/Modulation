import random

from src import (
    SimulationParameters,
    ModulationSimulation,
    BitStream,
    Carrier,
    BPSKModulator,
    QPSKModulator,
    SixteenQAMModulator,
    BFSKModulator,
)

random.seed(100)

bits = [random.randint(0, 1) for _ in range(100000)]

parameters = SimulationParameters(
    sampling_frequency=100000,
    t_max=1,
    carrier=Carrier(10000),
    bit_stream=BitStream(100, bits),
)


simulation = ModulationSimulation(
    parameters,
    BFSKModulator(100),
    # snr=-20,
    snr=0,
)
simulation.run()
