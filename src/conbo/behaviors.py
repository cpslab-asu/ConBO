from dataclasses import dataclass
import enum
import time
from typing import Any

from attr import frozen


class Behavior(enum.IntEnum):
    """Behavior when minimizing or falsifying components.

    Attributes:
    ----------
        MINIMIZATION: Minimize robustness value until the budget is exhausted.
        FALSIFICATION_ANY: Stop searching when the first falsifying component (rob < 0) is encountered.
        FALSIFICATION_ITERATIVE: Continue falsification, eliminating falsified components,
                                 until the budget is exhausted or all components are eliminated.
        FALSIFICATION_AT_ONCE: Continue falsification until all components are falsified
                               at a single sample or the budget is exhausted.
    """

    MINIMIZATION = enum.auto()
    FALSIFICATION_ANY = enum.auto()
    FALSIFICATION_ITERATIVE = enum.auto()
    FALSIFICATION_AT_ONCE = enum.auto()

class AlgorithmPreference(enum.IntEnum):
    """
    Enumeration of available algorithm preferences for optimization.

    This enum defines different strategies used in the optimization process.
    Each algorithm has distinct requirements and behaviors when interacting 
    with the `Configuration` class.

    Attributes:
    ----------
    CONBOPS : int
        "Conjunctive Bayesian Optimization - Pure Sampling":
        - Uses all components without requiring additional input parameters.
        - Applied when full exploration is needed.

    CONBOLS : int
        "Conjunctive Bayesian Optimization - Limited Sampling":
        - Requires `top_k` to specify the number of components considered.
        - If `k = #R`, behaves like CONBOPS.
        - If `k < #R`, starts with CONBOLS and may switch to CONBOPS.
        - Switches to MINBO if `#AR = 1`.

    MINBO : int
        "Minimal Budget Optimization":
        - Focuses on minimizing the budget while searching for solutions.
        - Does not require `top_k` as an input.
        - Works with both minimization and falsification strategies.

    """
    CONBOPS = enum.auto()
    CONBOLS = enum.auto()
    MINBO = enum.auto()

class Sampling(enum.IntEnum):
    """Behavior when falsifying case for system is encountered.

    Attributes:
     ----------
         FALSIFICATION: Stop searching when the first falsifying case is encountered
         MINIMIZATION: Continue searching after encountering a falsifying case until iteration
                       budget is exhausted
    """

    LHS = enum.auto()
    UNIF_SAMPLING = enum.auto()



@dataclass(frozen=False)
class SampleStats:
    iteration_timestamps: Any = None
    topk_time: Any = None
    sample_generation_time: Any = None
    optimal_pair_set: Any = None

class StatStorer:
    def __init__(self):
        self.initial_timestamp = time.perf_counter()
        self.history = []
    
    def __call__(self, iteration_timestamps=None, top_k_time=None, sample_generation_time=None, optimal_pair=None):
        self.history.append(SampleStats(iteration_timestamps, top_k_time, sample_generation_time, optimal_pair))
