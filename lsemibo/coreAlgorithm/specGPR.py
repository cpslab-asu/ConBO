from ..gprInterface import GPR
from ..sampling import uniform_sampling, lhs_sampling
from scipy.stats import norm, pearsonr
from scipy import stats
from sklearn import preprocessing
from copy import deepcopy
import numpy as np

class SpecEI:
    def __init__(self, identifier, x_train, y_train_subset1, y_train_subset2, best_point, mapping_indices_1, mapping_indices_2, gpr_model1, gpr_model2, region_support, R, M, tf_dim, rng, sampling_type = "lhs_sampling"):
        self.id = identifier
        self.x_train = x_train
        self.y_train_subset1 = y_train_subset1
        self.y_train_subset2 = y_train_subset2
        self.mapping_indices_1 = mapping_indices_1
        self.mapping_indices_2 = mapping_indices_2
        self.sampling_type = sampling_type
        self.region_support = region_support
        self.tf_dim = tf_dim
        self.rng = rng
        self.R = R
        self.M = M
        self.model1 = gpr_model1
        self.model2 = gpr_model2
        #self.model1 = deepcopy(GPR(gpr_model))
        #self.model2 = deepcopy(GPR(gpr_model))
        #self.model1.fit(self.x_train, self.y_train_subset1)
        #self.model2.fit(self.x_train, self.y_train_subset2)
        self.best_point = best_point
        # self.prob = [self._cal_prob() for _ in range(self.R)]
        self.prob = self._cal_prob()
        
    def _cal_prob(self):
        
        cdf_all_sum_1 = 0
        cdf_all_sum_2 = 0
        
        for _ in range(self.R):
            if self.sampling_type == "lhs_sampling":
                samples = lhs_sampling(self.M, self.region_support, self.tf_dim, self.rng)
            elif self.sampling_type == "uniform_sampling":
                samples = uniform_sampling(self.M, self.region_support, self.tf_dim, self.rng)
            else:
                raise ValueError(f"{self.sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
            samples_subset_1 = samples[:, self.mapping_indices_1]
            samples_subset_2 = samples[:, self.mapping_indices_2]
            
            
            y_pred_1, pred_sigma_1 = self.model1.predict(samples_subset_1)
            y_pred_2, pred_sigma_2 = self.model2.predict(samples_subset_2)

            cdf_all_sum_1 += np.sum(stats.norm.cdf(self.best_point, y_pred_1, pred_sigma_1))
            cdf_all_sum_2 += np.sum(stats.norm.cdf(self.best_point, y_pred_2, pred_sigma_2))
            
        return (cdf_all_sum_1 + cdf_all_sum_2)/2
    


    def _surrogate(self, x_train):
        """_surrogate Model function

        Args:
            model: Gaussian process model
            X: Input points

        Returns:
            [type]: predicted values of points using gaussian process model
        """

        return self.model1.predict(x_train),self.model2.predict(x_train)

    def pcc(self, var_1, var_2):
        intermed_var_1 = var_1 - np.mean(var_1)
        intermed_var_2 = var_2 - np.mean(var_2)
        num = np.sum(intermed_var_1 * intermed_var_2)
        den = np.sqrt(np.sum(intermed_var_1**2) * np.sum(intermed_var_2**2))    
        return num/den

    def minGP_pred(self, sample, c1, c2, sample_type="single"):
        
        
        if len(sample.shape) == 1:
            sample = sample.reshape((-1,1)).T

        
        if sample_type == "multiple":

            mu1, std1 = self._surrogate(sample)[0]
            mu2, std2 = self._surrogate(sample)[1]
            

            for mu1_iter, mu2_iter, std1_iter, std2_iter in zip(mu1, mu2, std1, std2):
                data1 = np.random.normal(mu1_iter, std1_iter, 1000)
                data2 = np.random.normal(mu2_iter, std2_iter, 1000)
                rho = self.pcc(data1, data2)
            
        
                

                theta = (((std1 ** 2) + (std2 ** 2) - (2 * rho * std1 * std2)) ** 0.5)
                
                t1 = mu1 * norm.cdf((mu2 - mu1) / theta)
                t2 = mu2 * norm.cdf((mu1 - mu2) / theta)
                t3 = (theta * norm.pdf((mu2 - mu1) / theta))
                t4 = (std1 ** 2 + mu1 ** 2) * norm.cdf((mu2 - mu1) / theta)
                t5 = (std2 ** 2 + mu2 ** 2) * norm.cdf((mu1 - mu2) / theta)
                t6 = (mu1 + mu2) * theta * norm.pdf((mu2 - mu1) / theta)

                
                pred_mean = t1 + t2 - t3
                pred_var = t4 + t5 - t6 - (pred_mean ** 2)
                pred_std = pred_var ** 0.5
                
                
        elif sample_type == "single":
            mu1, std1 = self._surrogate(sample.reshape(1, -1))[0]
            mu2, std2 = self._surrogate(sample.reshape(1, -1))[1]
            
            data1 = np.random.normal(mu1, std1, 1000)
            data2 = np.random.normal(mu2, std2, 1000)
            rho, _ = pearsonr(data1, data2)
            #print(rho)
            theta = (std1 ** 2 + std2 ** 2 - 2 * rho * std1 * std2) ** 0.5
            t1 = mu1 * norm.cdf((mu2 - mu1) / theta)
            t2 = mu2 * norm.cdf((mu1 - mu2) / theta)
            t3 = (theta * norm.pdf((mu2 - mu1) / theta))
            t4 = (std1 ** 2 + mu1 ** 2) * norm.cdf((mu2 - mu1) / theta)
            t5 = (std2 ** 2 + mu2 ** 2) * norm.cdf((mu1 - mu2) / theta)
            t6 = (mu1 + mu2) * theta * norm.pdf((mu2 - mu1) / theta)

            pred_mean = t1 + t2 - t3
            pred_var = t4 + t5 - t6 - (pred_mean ** 2)
            pred_std = pred_var ** 0.5
           
        return pred_mean, pred_std, pred_var
    

    def _acquisition(self, sample, c1,c2, sample_type="single"):
        
        curr_best = self.best_point
        
        if len(sample.shape) == 1:
            sample = sample.reshape((-1,1)).T
        
       
        if sample_type == "multiple":
            
            
            pred_mean, pred_std, pred_var = self.minGP_pred(sample, c1,c2,sample_type="multiple")
        
            #print("sample",sample)
            ei_list = []
            for mu_iter, std_iter, var_iter in zip(pred_mean, pred_std, pred_var):
                ppred_mean = mu_iter
                ppred_std = std_iter
                pvar_iter = var_iter
                

                if pvar_iter > 0:
                    var_1 = curr_best - ppred_mean
                    var_2 = var_1 / ppred_std
            

                    ei = (var_1 * norm.cdf(var_2)) + (ppred_std * norm.pdf(var_2))

                else:
                    ei = 0.0
                
                ei_list.append(ei)
            #print("ei_list",ei_list)
            return ei_list, pred_mean

        elif sample_type == "single":
            sample = sample.reshape(1, -1)
            
            #print("sample",sample)
            pred_mean, pred_std, pred_var = self.minGP_pred(sample,c1,c2, sample_type="single")

            if pred_var > 0:
                var_1 = curr_best - pred_mean
                var_2 = var_1 / pred_std
                
              
                ei = (var_1 * norm.cdf(var_2)) + (pred_std * norm.pdf(var_2))
            else:
                ei = 0.0
        #print("ei",ei)
        return ei, pred_mean
 

class minSpecEI:
    def __init__(self, identifier, x_train, y_train, best_point, mapping_indices, gpr_model, region_support, tf_dim, rng, sampling_type = "lhs_sampling"):
        self.id = identifier
        self.x_train = x_train
        self.y_train = y_train
        self.mapping_indices = mapping_indices
        self.sampling_type = sampling_type
        self.region_support = region_support
        self.tf_dim = tf_dim
        self.model = gpr_model
        self.model = deepcopy(GPR(gpr_model))
        self.model.fit(self.x_train, self.y_train)
        self.best_point = best_point


    def _surrogate(self, x_train):
        """_surrogate Model function

        Args:
            model: Gaussian process model
            X: Input points

        Returns:
            [type]: predicted values of points using gaussian process model
        """

        return self.model.predict(x_train)

    def _acquisition(self, sample, sample_type="single"):
        
        if len(sample.shape) == 1:
            sample = sample.reshape((-1,1)).T


        sample_subset = sample[:, self.mapping_indices]
        curr_best = self.best_point
        # curr_best = np.min(self.y_train)
        
        if sample_type == "multiple":
            
            mu, std = self._surrogate(sample_subset)
            ei_list = []
            for mu_iter, std_iter in zip(mu, std):
                pred_var = std_iter
                
                if pred_var > 0:
                    var_1 = curr_best - mu_iter
                    var_2 = var_1 / pred_var

                    ei = (var_1 * norm.cdf(var_2)) + (
                        pred_var * norm.pdf(var_2)
                    )
                else:
                    ei = 0.0

                ei_list.append(ei)

            return np.array(ei_list)

        elif sample_type == "single":
            mu, std = self._surrogate(sample_subset.reshape(1, -1))
            
            pred_var = std[0]
            if pred_var > 0:
                var_1 = curr_best - mu[0]
                var_2 = var_1 / pred_var

                ei = (var_1 * norm.cdf(var_2)) + (
                    pred_var * norm.pdf(var_2)
                )
            else:
                ei = 0.0
            return ei
       