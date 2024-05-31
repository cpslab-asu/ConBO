import numpy as np
import numpy.typing as npt
from typing import List, Type
from .function import Evaluator


def compute_robustness(samples_in: npt.NDArray, requirement, test_function: Type[Evaluator]) -> npt.NDArray:
    """Compute the fitness (robustness) of the given sample.

    Args:
        samples_in: Samples points for which the fitness is to be computed.
        test_function: Test Function insitialized with Fn
    Returns:
        Fitness (robustness) of the given sample(s)
    """

    if samples_in.shape[0] == 1:
        samples_out = test_function(samples_in[0], requirement)
        samples_out = np.array([samples_out])
    else:
        samples_out = []
        # active_specs = []
        for sample in samples_in:
            samp_out = test_function(sample, requirement)
            samples_out.append(samp_out)

        samples_out = np.array(samples_out)
        

    return samples_out