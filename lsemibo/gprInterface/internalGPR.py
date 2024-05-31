from .gprInterface import GaussianProcessRegressorStructure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF, WhiteKernel
from scipy.optimize import fmin_l_bfgs_b
from warnings import catch_warnings
from sklearn.preprocessing import StandardScaler
import warnings


def optimizer_lbfgs_b(obj_func, initial_theta):
    with catch_warnings():
        warnings.simplefilter("ignore")
        params = fmin_l_bfgs_b(
            obj_func, initial_theta, bounds=None, maxiter=30, maxfun=1e10
        )
    return params[0], params[1]


class InternalGPR(GaussianProcessRegressorStructure):
    def __init__(self, random_state = 12345):
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.05)
        self.gpr_model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=30, random_state = random_state
        )
        self.xscale = StandardScaler()
        # self.yscale = StandardScaler()

    def fit_gpr(self, X, Y):
        """Method to fit gpr Model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        
        """
        X_scaled = self.xscale.fit_transform(X)
        # Y_scaled = self.yscale.fit_transform(Y)
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
        x_scaled = self.xscale.transform(X)
        with catch_warnings():
            warnings.simplefilter("ignore")
            yPred, predSigma = self.gpr_model.predict(x_scaled, return_std=True)
        return yPred, predSigma