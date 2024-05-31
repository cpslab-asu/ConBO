import numpy as np
import unittest

from numpy.random import default_rng
from numpy.typing import NDArray

from lsemibo.coreAlgorithm.specification import Requirement

import logging
import math
from typing import Any

import numpy as np


from staliro.models import State, ode
from staliro.optimizers import UniformRandom
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro



class TestSpecification(unittest.TestCase):
    def test1_unclassified_regions(self):
        
        
        rng = default_rng(12345)
        times = np.arange(0,10,0.01)
        vals = rng.random((5,1000))

        phi_1 = "G[0,10] (a<=0.2)"
        phi_2 = "F[2,10] (b<=0.2)"
        phi_3 = "G[5,10] (a<=0.2)"
        phi_4 = "F[4,10] (b<=0.2)"
        phi_5 = "G[5,10] (a<=0.2)"
        phi_6 = "F[8,10] (b<=0.2)"

        pred_map = {"a":([0,1], 0), 
                        "b":([0,1], 1),
                    }

        tf_dim = 2
        
        phi_list = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6]

        reqs = Requirement(tf_dim, phi_list, pred_map)

        print(reqs.falsified_components, reqs.num_falsified_components)
        print(reqs.unfalsified_components, reqs.num_unfalsified_components)
        
        print(reqs.evaluate(vals, times))

        vals = rng.random((5,1000))
        print(reqs.falsified_components, reqs.num_falsified_components)
        print(reqs.unfalsified_components, reqs.num_unfalsified_components)
        print(reqs.evaluate(vals, times))
        print(reqs.falsified_components, reqs.num_falsified_components)
        print(reqs.unfalsified_components, reqs.num_unfalsified_components)
        # print(reqs._generate_dataset())
        print(reqs._get_individual_monitoring_times())
    
    def test2_specification(self):
        @ode()
        def nonlinear_model(time: float, state: State, _: Any) -> State:
            x1_dot = state[0] - state[1] + 0.1 * time
            x2_dot = state[1] * math.cos(2 * math.pi * state[0]) + 0.1 * time

            return np.array([x1_dot, x2_dot])

        initial_conditions = [(-1, 1), (-1, 1)]
        phi_1 = r"always !(a >= -1.4 and a <= -1.2  and b >= -1.1 and b <= -0.9)"
        phi_2 = r"always !(a >= -1.3 and a <= -1.1  and b >= -1.2 and b <= -0.8)"
        phi_3 = r"always !(a >= -1.2 and a <= -1.0  and b >= -1.3 and b <= -0.7)"
        req = [phi_1, phi_2, phi_3]
        pred_map = {"a": ([0,1], 0), "b": ([0,1], 1)}
        specification = Requirement(2, req, pred_map)
        options = Options(runs=1, iterations=100, interval=(0, 2), static_parameters=initial_conditions)
        optimizer = UniformRandom()

        result = staliro(nonlinear_model, specification, optimizer, options)

