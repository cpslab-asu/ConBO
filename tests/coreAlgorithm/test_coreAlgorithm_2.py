import math
from tokenize import Single
from staliro.specifications import RTAMTDense
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
        Z = static[2]
        # d1 = X**3
        d2 = math.sin(X/2) + math.sin(Y/2) + math.sin(Z/2)  + 3
        d3 = math.sin((X-3)/2) + math.sin((Y-3)/2) + math.sin((Z-3)/2) + 6
        d4 = (math.sin((X - 6)/2)/2) + (math.sin((Y-6)/2)/2) + (math.sin((Z - 6)/2)/2) + 3
        # print(f"True val = {d2}, {d3}, {d4}")
        data_array = np.hstack((d2,d3, d4)).reshape((-1,1))
        
        return ModelData(data_array, timestamps_array)


model = NLFModel()

initial_conditions = [
    np.array([1.95, 2.05]),
    np.array([1.95, 2.05]),
    np.array([1.95, 2.05]),
]

options = Options(runs=1, iterations=1, interval=(0, 1),  static_parameters=initial_conditions ,signals=[])


phi_1 = "x>=0"
phi_2 = "y>=3"
phi_3 = "z>=1.5"

fn_list_1 = [phi_1, phi_2, phi_3]
pred_map_1 = {"x": ([0,1,2], 0), "y":([0,1,2], 1), "z":([0,1,2], 2)}


is_budget = 20
max_budget = 60
cs_budget = 1000
spec_list = [fn_list_1]
predicate_mapping = pred_map_1
region_support = np.array([[-5., 5.],[-5., 5.], [-5., 5.]])
tf_dim = 3
R = 20
M = 500

from lsemibo.gprInterface import GaussianProcessRegressorStructure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF, WhiteKernel
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import StandardScaler
from warnings import catch_warnings
# from warnings import simplefilter
import warnings


def optimizer_lbfgs_b(obj_func, initial_theta):
    with catch_warnings():
        warnings.simplefilter("ignore")
        params = fmin_l_bfgs_b(
            obj_func, initial_theta, bounds=None, maxiter=30000, maxfun=1e10
        )
    return params[0], params[1]


class ExternalGPR(GaussianProcessRegressorStructure):
    def __init__(self, random_state = 12345):
        self.gpr_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state = random_state
        )
        self.scale = StandardScaler()

    def fit_gpr(self, X, Y):
        """Method to fit gpr Model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        
        """
        X_scaled = self.scale.fit_transform(X)
        
        with catch_warnings():
            warnings.simplefilter("ignore")
            self.gpr_model.fit(X_scaled, Y)

    def predict_gpr(self, X):
        """Method to predict mean and std_dev from gpr model

        Args:
            x_train: Samples from Training set.
            

        Returns:
            mean
            std_dev
        """
        x_scaled = self.scale.transform(X)
        with catch_warnings():
            warnings.simplefilter("ignore")
            yPred, predSigma = self.gpr_model.predict(x_scaled, return_std=True)
        return yPred, predSigma


emibo = LSemiBO(is_budget, max_budget, cs_budget, 2, 0.8, model, spec_list, predicate_mapping, region_support, tf_dim, options, R, M, is_type = "lhs_sampling", cs_type = "lhs_sampling", seed = 12345)
tf_wrapper = emibo.sample(ExternalGPR(), InternalClassifier())

point_history = np.array(tf_wrapper.point_history, dtype = object)

import matplotlib.pyplot as plt
f1 = np.array([[-3.14, -3.14]])
f2 = np.array([[-0.14, -0.14]])
f3 = np.array([[2.86, 2.86]])
plt.plot(f1[:,0], f1[:,1], "r*", label = "Function 1")
plt.plot(f2[:,0], f2[:,1], "r*", label = "Function 2")
plt.plot(f3[:,0], f3[:,1], "r*", label = "Function 3")

points = np.array([point for point in point_history[:,1]])


plt.plot(points[:,0], points[:,1], "b.", label = "Sampled Points")
plt.show()

# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()
# import pickle
# with open("nlf_test.pkl", "wb") as f:
#     pickle.dump(req, f)

# tf_wrapper = emibo.sample(InternalGPR(), InternalClassifier())

# for s in tf_wrapper.spec_list_local:
#     print(vars(s))

# print(np.array(tf_wrapper.point_history, dtype = object))


# f = Fn(model, fn_list, pred_map, options, io_mapping)
# reg_sup = np.array([[-5., 5.], [-5., 5.]])
# samples_in = lhs_sampling(10, reg_sup, 2, np.random.default_rng(1223))
# rob, act = compute_robustness(samples_in, f)

# print(rob)
# print(act)

# print(np.array(f.point_history, dtype = object))

# for s in f.spec_list_local:
#     print(vars(s))