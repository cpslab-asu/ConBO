import math
import pickle
from staliro import Sample, SignalInput, TestOptions, staliro
from staliro.models import Model, Result


from staliro import TestOptions, SignalInput
import numpy as np
from numpy.typing import NDArray

from conbo.staliroIntegration import AlgorithmPreference, Behavior, ConjunctiveBO, Sampling
from conbo.gpr import InternalGPR
from conbo.classifier import InternalClassifier
import staliro.optimizers as optimizers

import staliro

NLFDataT = NDArray[np.float_]
NLFResultT = Result[NLFDataT, None]


""" class NLFModel(Model[NLFResultT, None]):
    def simulate(
        self, static: ModelInputs, intrvl: Interval
    ) -> NLFResultT:
        print(static)
        timestamps_array = np.array(1.0).flatten()
        X = static.static[0]
        Y = static.static[1]
        d1 = X**3
        d2 = math.sin(X/2) + math.sin(Y/2) + 2
        d3 = math.sin((X-3)/2) + math.sin((Y-3)/2) + 4
        d4 = (math.sin((X - 6)/2)/2) + (math.sin((Y-6)/2)/2) + 2
        # print(f"True val = {d2}, {d3}, {d4}")
        data_array = np.hstack((d2,d3, d4)).reshape((-1,1))
        timestamps = timestamps_array
        data_list = data_array
        trace = Trace(timestamps, data_list)
        return BasicResult(trace)


model = NLFModel()
 """

""" model = NLFModel()

initial_conditions = [
    np.array([-5,5]),
    np.array([-5,5]),
]


options = Options(runs=1, iterations=5, interval=(0, 1),  static_parameters=initial_conditions ,signals=[])


phi_2 = "x>=0"
phi_3 = "y>=2"
phi_4 = "z>=1"

fn_list_1 = [phi_2, phi_3, phi_4]
pred_map_1 = {"x": ([0,1], 0), "y":([0, 1],1), "z":([0, 1], 2)} """

class NLFModel(Model[list[float], None]):
    def simulate(
        self, sample:Sample
    ) -> NLFResultT:
        print(sample.static)
        timestamps_array = np.array(1.0).flatten()
        X = sample.static["dim1"]
        Y = sample.static["dim2"]
        
        d0 = 0.8077039507222558 * math.sin(0.31749166137974444 * X + 3.414235551061579) + 1.7652959525445269 * math.sin(0.38688344598162405 * Y + 4.129915202504272) + 8.208314357501266-  2.0041014709029863 - 0.025375643039574403
        d1 = 0.9136927264715098 * math.sin(0.3380695707915824 * X + 2.99209733441971) + 0.9919108116224034 * math.sin(0.2737700940482345 * Y + 3.016138653872876) + 4.876355287140777-  2.0041014709029863 - 0.025375643039574403
        d2 = 0.5449256058606937 * math.sin(0.44275068365253756 * X + 5.757693083193321) + 0.778184646708369 * math.sin(0.363377557436778 * Y + 4.020695129363184) + 6.522903561787862-  2.0041014709029863 - 0.025375643039574403
        d3 = 0.9422991134289538 * math.sin(0.25636114680961314 * X + 1.9157893377362838) + 1.8404192204778744 * math.sin(0.424102626544859 * Y + 0.830011929919672) + 8.000792780533539-  2.0041014709029863 - 0.025375643039574403
        d4 = 1.1167307430089302 * math.sin(0.8434046930577452 * X + 2.7214154818652054) + 1.882234487202133 * math.sin(0.8458913723930879 * Y + 1.0796787773296896) + 8.584307539751084-  2.0041014709029863 - 0.025375643039574403
        
       
        # print(f"True val = {d2}, {d3}, {d4}")
        data_array = np.hstack((d0,d1,d2, d3, d4)).reshape((-1,1))
        data_list = data_array.T
        # trace = Trace(timestamps, data_list)
        return NLFResultT(times=timestamps_array, states=data_list, extra=None)




model = NLFModel()

initial_conditions = {
    "dim1": np.array([-5,5]),
    "dim2": np.array([-5,5]),
}

options = TestOptions(tspan=(0,1), iterations=1, runs = 1, static_inputs=initial_conditions)

phi_2 = "a>=0"
phi_3 = "b>=0"
phi_4 = "c>=0"
phi_5 = "d>=0"
phi_6 = "e>=0"
phi_7 = "f>=0"
phi_8 = "g>=0"
phi_9 = "h>=0"
phi_10 = "i>=0"
phi_11 = "j>=0"
phi_12 = "k>=0"
phi_13 = "l>=0"
phi_14 = "m>=0"
phi_15 = "n>=0"
phi_16 = "o>=0"
phi_17 = "p>=0"
phi_18 = "q>=0"
phi_19 = "r>=0"
phi_20 = "s1>=0"
phi_21 = "t>=0"
phi_22 = "u>=0"
phi_23 = "v>=0"
phi_24 = "x>=0"
phi_25 = "y>=0"
phi_26 = "z>=0"
phi_27 = "P>= 0"
phi_28 = "Q>=0"
phi_29 = "R>=0"
phi_30 = "S1>=0"
phi_31 = "T>=0"

# fn_list_1 = [phi_2, phi_3, phi_4,phi_5, phi_6 ,phi_7 ,phi_8, phi_9 ,phi_10,phi_11,phi_12,phi_13, phi_14, phi_15, phi_16, phi_17, phi_18, phi_19,phi_20, phi_21, phi_22, phi_23, phi_24, phi_25, phi_26, phi_27, phi_28, phi_29,phi_30, phi_31]
fn_list_1 = [phi_2, phi_3, phi_4,phi_5, phi_6]
 # Change this
pred_map_1 = {"a": ([0,1], 0),
              "b": ([0,1], 1),
              "c": ([0,1], 2),
              "d": ([0,1], 3),
              "e": ([0,1], 4)}


is_budget = 20
max_budget = 30
cs_budget = 1000
spec_list = [fn_list_1]
predicate_mapping = pred_map_1
region_support = np.array([[-5., 5.], [-5., 5.]])
tf_dim = 2
R = 20
M = 500


top_k = 1
Benchmark_name = "NLF_trial"
#UNCOMMENT THE PICKLE LINe	
seed = 123457

total_runs = 1
from conbo.specification import FalsificationIterativeBehaviorRequirement
specification = FalsificationIterativeBehaviorRequirement(tf_dim, fn_list_1, pred_map_1)
# optimizer = optimizers.UniformRandom()

# print(f"Rob. Sample for = {runs}")

for i in range(total_runs):

    optimizer = ConjunctiveBO( 
        behavior=Behavior.FALSIFICATION_ITERATIVE,
        algorithm = AlgorithmPreference.CONBOLS,
        is_budget = is_budget,
        max_budget= max_budget,
        cs_budget = cs_budget,
        top_k = top_k,
        classified_sample_bias = 1,
        tf_dim = tf_dim,
        R = R,  
        M = M,
        gpr_model = InternalGPR(),
        classifier_model = InternalClassifier(),
        is_type = Sampling.LHS,
        cs_type= Sampling.LHS,
        pi_type= Sampling.LHS,
        seed= seed+i)
    runs = staliro.test(model, specification, optimizer, options)
    # result = staliro(model, specification, optimizer, options)
#     with open(f'NLF_{is_budget}_{max_budget}_seed_{seed+i}.pkl', 'wb') as file:
#         pickle.dump(result, file)

# with open(f'NLF_{is_budget}_{max_budget}_seed_{seed+i}.pkl', 'rb') as f:
#     data = pickle.load(f)

# # print([x for x in data.runs[0].model_timing.durations])

# print(data.runs[0].result.start_timestamp)
# print(data.runs[0].result.iteration_timestamps)
# x = np.array([data.runs[0].result.start_timestamp] + data.runs[0].result.iteration_timestamps)

# print(np.diff(x))


# #print(result)
# # for runs in range(total_runs):
    
# #     lsemibo = LSemiBO(Benchmark_name, runs, is_budget, max_budget, cs_budget, top_k, 0.8, model, spec_list, predicate_mapping, tf_dim, options, R, M, is_type = "lhs_sampling", cs_type = "lhs_sampling", seed = 12345)
# #     x_train, y_train, time_taken = lsemibo.sample(InternalGPR(), InternalClassifier())

# #     print(x_train)
# #     print(y_train)
# #     print(time_taken)