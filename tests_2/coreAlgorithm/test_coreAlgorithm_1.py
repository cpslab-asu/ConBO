import math
from lsemibo.sampling.uniformSampling import uniform_sampling
from staliro.specifications import RTAMTDense
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.models import StaticInput, SignalTimes, SignalValues, ModelData, blackbox

from staliro.options import Options, SignalOptions
from staliro.core.interval import Interval
import numpy as np
from numpy.typing import NDArray
import numpy as np
from numpy.typing import NDArray
from lsemibo.sampling.lhsSampling import lhs_sampling
from lsemibo.utils import Evaluator, SingleRequirement, compute_robustness
from lsemibo.coreAlgorithm import LSemiBO
from lsemibo.gprInterface import InternalGPR
from lsemibo.classifierInterface import InternalClassifier

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True

CCDataT = NDArray[np.float_]
CCResultT = ModelData[CCDataT, None]
eng = matlab.engine.start_matlab()
MODEL_NAME = "cars"
mo = eng.simget(MODEL_NAME)
model_opts = eng.simset(mo, "SaveFormat", "Array")


@blackbox
def cc_simulate(
    static: StaticInput, times: SignalTimes, signals: SignalValues
) -> CCResultT:

    # n_times = (max(times) // self.sampling_step) + 2
    # signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
    # signal_values = np.array(
    #     [[signal.at_time(t) for t in signal_times] for signal in signals]
    # )
    # print(signal_times.shape)
    # print(signal_values.shape)
    # with matlab.engine.start_matlab() as eng:
    # print(times)
    # print("*****************")
    # print(signals)
    # print("*****************")
    # print(times.shape)
    # print(signals.shape)
    # print("************************")
    # print(efeof)

    sim_t = matlab.double([0, max(times)])
    model_input = matlab.double(np.row_stack((times, signals)).T.tolist())

    timestamps, _, data = eng.sim(MODEL_NAME, sim_t, model_opts, model_input, nargout=3)

    timestamps_array = np.array(timestamps).flatten()
    data_array = np.array(data)

    y54 = (data_array[:, 4] - data_array[:, 3]).reshape((-1, 1))
    y43 = (data_array[:, 3] - data_array[:, 2]).reshape((-1, 1))
    y32 = (data_array[:, 2] - data_array[:, 1]).reshape((-1, 1))
    y21 = (data_array[:, 1] - data_array[:, 0]).reshape((-1, 1))
    diff_array = np.hstack((y21, y32, y43, y54))
    # print(diff_array.shape)
    return ModelData(diff_array.T, timestamps_array)



#####################################################################################################################
# Define Signals and Specification

signals = [
    SignalOptions(control_points=[(0,1)]*10, signal_times=np.linspace(0.0, 100.0, 10)),
    SignalOptions(control_points=[(0,1)]*10, signal_times=np.linspace(0.0, 100.0, 10)),
]
options = Options(runs=1, iterations=1, interval=(0, 100),  signals=signals)

phi_1 = "(G[0, 50] (y21 >= 7.5))"
phi_2 = "(G[0, 50] (y32 >= 7.5))"
phi_3 = "(G[0, 50] (y43 >= 7.5))"
phi_4 = "(G[0, 50] (y54 >= 7.5))"
phi_5 = f"{phi_1} and {phi_2} and {phi_3} and {phi_4}"

fn_list = [phi_1, phi_2, phi_3, phi_4]
pred_map = {"y21":([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 0), 
            "y32":([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 1),
            "y43":([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 2), 
            "y54":([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 3), }
# io_mapping = [
#                 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
#                 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
#                 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
#                 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
#                 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
#              ]

is_budget = 50
max_budget = 300
cs_budget = 10000
spec_list = [fn_list]
predicate_mapping = pred_map
region_support = np.array([[0., 1.]]*20)
tf_dim = 20
R = 20
M = 500

points = []
samples = []
robs = []
for i in range(10):
    seed = 12345 + i
    emibo = LSemiBO(is_budget, max_budget, cs_budget, 2, 0.8, cc_simulate, spec_list, predicate_mapping, region_support, tf_dim, options, R, M, is_type = "lhs_sampling", cs_type = "lhs_sampling", seed = seed)
    point, sample, rob = emibo.sample(InternalGPR(), InternalClassifier())
    points.append(point)
    samples.append(sample)
    robs.append(rob)
    print("****************************")
    print("****************************")
    print("****************************")
    print("****************************")
    print(points)
    print("****************************")
    print("****************************")
    print("****************************")
    print("****************************")

print(points)
print(f"Mean points: {np.mean(points)}")
print(f"Median points: {np.median(points)}")

import pickle
with open("cc_test.py", "wb") as f:
    pickle.dump((points, samples, robs), f)

print(points)
print(f"Mean points: {np.mean(points)}")
print(f"Median points: {np.median(points)}")
# sample = uniform_sampling(10, region_support, 20, np.random.default_rng(12345))
# rob, ac_spec = compute_robustness(sample, emibo.tf_wrapper)

# print(rob)
# print(ac_spec)
# tf_wrapper = emibo.sample(InternalGPR(), InternalClassifier())

# for s in tf_wrapper.spec_list_local:
#     print(vars(s))

# # print(np.array(tf_wrapper.point_history, dtype = object))
# f = Fn(model, fn_list, pred_map, options)
# reg_sup = np.array([[-5., 5.], [-5., 5.]])
# samples_in = lhs_sampling(1, reg_sup, 2, np.random.default_rng(1223))
# rob, act = compute_robustness(samples_in, f)

# print(rob)
# print(act)

# print(np.array(f.point_history, dtype = object))

# for s in f.spec_list_local:
#     print(vars(s))