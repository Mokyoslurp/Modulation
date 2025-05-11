import random

from src import (
    SimulationParameters,
    BERSimulation,
    BitStream,
    Carrier,
    BPSKModulator,
    QPSKModulator,
    SixteenQAMModulator,
    BFSKModulator,
)

random.seed(100)

# bits = [random.randint(0, 1) for _ in range(1000000)]
bits = [random.randint(0, 1) for _ in range(100000)]

parameters = SimulationParameters(
    sampling_frequency=100000,
    t_max=10,
    carrier=Carrier(10000),
    bit_stream=BitStream(100, bits),
)

BER_sim = BERSimulation(
    parameters,
    modulators=[
        BPSKModulator(10000),
        QPSKModulator(10000),
        SixteenQAMModulator(10000),
        BFSKModulator(10000),
    ],
    min=-5,
    max=10,
    n=5,
    # n=10,
)

BER_sim.run(full_modulation=True)
BER_sim.plot()
