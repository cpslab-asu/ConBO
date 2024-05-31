import math
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.options import Options, SignalOptions
from staliro.core.interval import Interval
import numpy as np
from numpy.typing import NDArray

from lsemibo.coreAlgorithm import LSemiBO
from lsemibo.gprInterface import InternalGPR
from lsemibo.classifierInterface import InternalClassifier

NLFDataT = NDArray[np.float_]
NLFResultT = ModelData[NLFDataT, None]


class NLFModel(Model[NLFResultT, None]):
    def simulate(
        self, static: StaticInput, signals: Signals, intrvl: Interval
    ) -> NLFResultT:

        timestamps_array = np.array(1.0).flatten()
        X = static[0]
        Y = static[1]
        d1 = X**3
        d2 = math.sin(X/2) + math.sin(Y/2) + 2
        d3 = math.sin((X-3)/2) + math.sin((Y-3)/2) + 4
        d4 = (math.sin((X - 6)/2)/2) + (math.sin((Y-6)/2)/2) + 2
        # print(f"True val = {d2}, {d3}, {d4}")
        data_array = np.hstack((d1, d2,d3, d4)).reshape((-1,1))
        
        return ModelData(data_array, timestamps_array)


model = NLFModel()

initial_conditions = [
    np.array([-5,5]),
    np.array([-5,5]),
]

options = Options(runs=1, iterations=1, interval=(0, 1),  static_parameters=initial_conditions ,signals=[])

phi_1 = "w>=0"
phi_2 = "x>=0.1"
phi_3 = "y>=2.1"
phi_4 = "z>=1.1"


fn_list_1 = [phi_2, phi_3, phi_4]
pred_map_1 = {"w": ([0], 0), "x": ([0,1], 1), "y":([0, 1], 2), "z":([0, 1], 3)}


is_budget = 5
max_budget = 20
cs_budget = 1000
spec_list = fn_list_1
predicate_mapping = pred_map_1
region_support = np.array([[-5., 5.], [-5., 5.]])
tf_dim = 2
R = 20
M = 500
top_k = 5
Benchmark_name = "NLF"
folder_name = "Test"
seed = 12345
total_runs = 10
method = "falsification_elimination"
# for runs in range(total_runs):
    # benchmark_name, folder_name, run_number, is_budget, max_budget, cs_budget, top_k, classified_sample_bias, model, component_list, predicate_mapping, tf_dim, options, R, M, is_type = "lhs_sampling", cs_type = "lhs_sampling", pi_type = "lhs_sampling", starting_seed = 12345):
lsemibo = LSemiBO(method, Benchmark_name, folder_name, 1, is_budget, max_budget, cs_budget, top_k, 0.8, model, spec_list, predicate_mapping, tf_dim, options, R, M, is_type = "lhs_sampling", cs_type = "lhs_sampling", starting_seed = 12345)
out_data = lsemibo.sample(InternalGPR(), InternalClassifier())



    # print(x_train)
    # print(y_train)
    # print(time_taken)
