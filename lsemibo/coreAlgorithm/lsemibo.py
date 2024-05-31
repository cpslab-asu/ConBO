from __future__ import annotations

from .specification import Requirement
from ..sampling import uniform_sampling, lhs_sampling
from ..utils import compute_robustness, Evaluator, sample_spec
from ..gprInterface import GPR
from .specGPR import SpecEI
from ..classifierInterface import Classifier

from staliro.core.sample import Sample
from staliro.core import Optimizer
from pyswarms.single.global_best import GlobalBestPSO
from staliro.staliro import _signal_bounds
import numpy as np
from tqdm import tqdm
import pathlib
import pickle
import time
from scipy.stats import norm
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, Any
from attr import frozen





class LSemiBO:
    def __init__(self, test_function, method, is_budget, 
                max_budget, cs_budget, top_k, classified_sample_bias, 
                tf_dim, region_support, R, M, 
                is_type = "lhs_sampling", cs_type = "lhs_sampling", pi_type = "lhs_sampling", 
                starting_seed = 12345):

        
        self.seed = starting_seed
        self.method = method
        self.region_support = region_support
        
        self.is_type = is_type
        self.is_budget = is_budget
        self.cs_budget = cs_budget
        self.cs_type = cs_type
        self.max_budget = max_budget
        
        self.tf_dim = tf_dim
        self.R = R
        self.M = M
        
        
        self.top_k = top_k
        self.classified_sample_bias = classified_sample_bias
        
        self.pi_type = pi_type

        self.func = test_function
        
        self.rng = np.random.default_rng(self.seed)
        self.num_requirements = len(self.func.specification)
        self.rem_budget = self.max_budget - self.is_budget
        self.bo_budget = max_budget - is_budget
        
        
    def sample(self, input_gpr_model, input_classifier_model):
        # time_stats = {}
                    

        total_start_time = time.perf_counter()
        print(f"Starting Replication for seed {self.seed}")

        if self.is_type == "lhs_sampling":
            x_train = lhs_sampling(self.is_budget, self.region_support, self.tf_dim, self.rng)
        elif self.is_type == "uniform_sampling":
            x_train = uniform_sampling(self.is_budget, self.region_support, self.tf_dim, self.rng)
        else:
            raise ValueError(f"{self.is_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        # print(y_train)
        x_train_ = [Sample(tuple(l)) for l in x_train]
        y_train = self.func.eval_samples(x_train_)
        # y_train = compute_robustness(x_train, self.specification, self.tf_wrapper)
        print(y_train)
        
        topk_time = {}
        sample_generation_time = {}

        for budget in tqdm(range(self.max_budget - x_train.shape[0])):
            # print(f"***************************************************")
            # print(f"********************{budget}***********************")
            print(f"Falsified Components: {self.func.specification.num_falsified_components}\n Unfalsified components remaining: {self.func.specification.num_unfalsified_components}")
            if self.func.specification.num_unfalsified_components <= self.num_requirements:
                c1 = (self.func.specification.num_unfalsified_components == 0 and self.method == "falsification_elimination")
                c2 = (self.func.specification.num_unfalsified_components < self.num_requirements and self.method == "falsification")
                if c1 or c2:
                    # time_stats["total_time"] = time.perf_counter() - total_start_time
                    # time_stats["monitoring_time"] = self.tf_wrapper.monitoring_time
                    # time_stats["simulation_time"] = self.tf_wrapper.sim_time
                    # time_stats["individual_monitoring_time"] = self.requirement._get_individual_monitoring_times()
                    # time_stats["sample_generation_time"] = sample_generation_time
                    # time_stats["topk_time"] = topk_time



                    print("All reqs falsified")
                    print(f"Ending Replication Early for seed {self.seed}")
                    if c1:
                        print("Ending early due to elimination of all reqs")
                    elif c2:
                        print("Ending early due to falsification")
                    # output_data = {}
                    # output_data["samples"] = x_train
                    # output_data["components"] = self.requirement._get_complete_data()
                    # output_data["time_res"] = time_stats
                    # print(f"Ending Replication After Exhuasting budget for Run {self.run_number} with seed {self.seed}")
                    # with open(self.benchmark_directory.joinpath(self.benchmark_name + f"_{self.method}_point_history{self.run_number}.pkl"), "wb") as f:
                    #     pickle.dump(output_data, f)
                    
                    return x_train, \
                        self.func.specification._get_complete_data(), \
                        time.perf_counter() - total_start_time,\
                        self.func.history,\
                        self.func.specification._get_individual_monitoring_times(),\
                        sample_generation_time,\
                        topk_time
                            
                else:
                    # curr_req = self.requirements[self._select_requirement(curr_active_specs)]
                    
                    
                    pred_sample_x, topk_time_sample, sample_generation_time_sample = self.lsemibo_req(x_train, input_gpr_model, input_classifier_model)
                    topk_time[self.is_budget + budget+1] = topk_time_sample
                    sample_generation_time[self.is_budget + budget+1] = sample_generation_time_sample

                    x_train = np.vstack((x_train, np.array([pred_sample_x])))
                    pred_sample_y = self.func.eval_sample(Sample(tuple(pred_sample_x)))
                    # pred_sample_y = compute_robustness(np.array([pred_sample_x]), self.specification, self.tf_wrapper)
                    print(f"pred_sample_y= {pred_sample_y}")
                    
        
    
        # time_stats["total_time"] = time.perf_counter() - total_start_time
        # time_stats["monitoring_time"] = self.tf_wrapper.monitoring_time
        # time_stats["simulation_time"] = self.tf_wrapper.sim_time
        # time_stats["individual_monitoring_time"] = self.requirement._get_individual_monitoring_times()
        # time_stats["sample_generation_time"] = sample_generation_time
        # time_stats["topk_time"] = topk_time
        # output_data = {}
        # output_data["samples"] = x_train
        # output_data["components"] = self.requirement._get_complete_data()
        # output_data["time_res"] = time_stats
        print(f"Ending Replication After Exhuasting budget for seed {self.seed}")
        # with open(self.benchmark_directory.joinpath(self.benchmark_name + f"_{self.method}_point_history{self.run_number}.pkl"), "wb") as f:
        #     pickle.dump(output_data, f)

        return x_train, \
                self.func.specification._get_complete_data(), \
                time.perf_counter() - total_start_time,\
                self.func.history,\
                self.func.specification._get_individual_monitoring_times(),\
                sample_generation_time,\
                topk_time


    def lsemibo_req(self, x_train, gpr_model, classifier_model):
        t_start_sample_gen_time = time.perf_counter()
        idxs, y_train = self.func.specification._generate_unfaslified_dataset()
        
        idxs_dict = dict(zip(idxs, list(range(len(idxs)))))

        y_train_classes = np.argmin(y_train, 1)

        t_start_topk_time = time.perf_counter()
        sampled_specs = self.choose_top_k(x_train, y_train_classes, idxs, classifier_model)
        topk_time = time.perf_counter() - t_start_topk_time

        
        component_ei = []
        best_point = np.min(y_train)
        pi = []
        
        for iterate in sampled_specs:
            
            mapping_indices = np.where(self.func.specification.requirements[iterate].io_mapping == 1)[0]
            
            x_train_subset = x_train[:, mapping_indices]
            y_train_subset = y_train[:, idxs_dict[iterate]]
            
            req_comp = SpecEI(iterate, x_train_subset, y_train_subset, best_point, mapping_indices, gpr_model, self.region_support, self.R, self.M, self.tf_dim, self.rng, self.pi_type)
            pi.append(req_comp.prob)
            component_ei.append(req_comp)
        # print("***************************************************************")
        # print("***************************************************************")
        
        # print(pi)
        
        pi_mc = np.array(pi) / (np.sum(pi))
        #print(f"Original PI = {pi}\nNormalized Value and Sum = {pi_mc}, {np.sum(pi_mc)}")
        
        pred_sample_x = self._opt_acquisition(component_ei, pi_mc, self.region_support, self.tf_dim)
        sample_generation_time = time.perf_counter() - t_start_sample_gen_time
        
        return pred_sample_x, topk_time, sample_generation_time

    def choose_top_k(self, x_train, y_train, idxs, classifier_model):
        
        # top_k = min(self.top_k, self.requirement._get_num_unfalsified_comp())
        if self.top_k >= self.func.specification.num_unfalsified_components:
            sampled_specs = self.func.specification.unfalsified_components
            print("Entered c1")
        
        else:
            if np.unique(y_train).shape[0] == 1:
                _sampled_specs = np.unique(y_train)
                sampled_specs = []
                
                for iter in _sampled_specs:
                    sampled_specs.append(idxs[iter])
                # print("Enter c2.1")
            else:

                classifier_model = Classifier(classifier_model)
                classifier_model.fit(x_train, y_train)
                # print(y_train)
                if self.cs_type == "lhs_sampling":
                    x_test = lhs_sampling(self.cs_budget, self.region_support, self.tf_dim, self.rng)
                    # x_test_classifier_active = x_test 
                elif self.cs_type == "uniform_sampling":
                    x_test = uniform_sampling(self.cs_budget, self.region_support, self.tf_dim, self.rng)
                    # x_test_classifier_active = x_test 
                else:
                    raise ValueError(f"{self.cs_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")

                y_test_classifier = classifier_model.predict(x_test)
                _unique, _counts = np.unique(y_train, return_counts=True)
                unique, counts = np.unique(y_test_classifier, return_counts=True)
                #print("**********************")
                #print("**********************")
                #print(f"Unique in y_train is \t {_unique}")
                #print(f"Counts in y_train is \t {_counts}")
                #print(f"Unique in y_test is \t {unique}")
                #print(f"Counts in y_test is \t {counts}")
                #print("**********************")
                #print("**********************")
                spec_prob = {}
                classified_spec_prob = []
                unclassified_spec_prob = []
                for spec_number in range(len(idxs)):
                    if spec_number not in set(unique):
                        spec_prob[spec_number] = 0
                        unclassified_spec_prob.append(spec_number)
                    else:
                        spec_prob[spec_number] = counts[np.where(unique == spec_number)][0] / self.cs_budget
                        classified_spec_prob.append(spec_number)
                # print(spec_prob, unclassified_spec_prob, classified_spec_prob)
                _sampled_specs = sample_spec(spec_prob, unclassified_spec_prob, classified_spec_prob, self.top_k, self.classified_sample_bias, self.rng)
                # print(classified_spec_prob)
                # print(unclassified_spec_prob)
                # print(classified_spec_prob + unclassified_spec_prob)
                # _sampled_specs = self.rng.choice(classified_spec_prob+ unclassified_spec_prob, size=self.top_k, p = [self.classified_sample_bias, 1 - self.classified_sample_bias])
                sampled_specs = []
                
                for iter in _sampled_specs:
                    sampled_specs.append(idxs[iter])
                #print("Enter c.2")
                print(_sampled_specs)
                print(sampled_specs)
                #print("Exit")
        return sampled_specs

    
    def _emi_ei(self, x, spec_ei, pi_mc, sample_type="single"):
        if x.shape[0] != 1:
            sample_type = "multiple"
        prob = []
        comp_ei = []
        
        for specs in spec_ei:
            comp_ei.append(specs._acquisition(x, sample_type))
        
        prob = np.array(pi_mc)
        if sample_type == "single":
            comp_ei = np.array([comp_ei])
        elif sample_type == "multiple":
            comp_ei = np.array(comp_ei).T
        
        # print("*********")
        # print(prob)
        # print(comp_ei)
        # print("*********")
        ei = np.sum(np.array(comp_ei) * prob, 1)
        
        return ei


    def _opt_acquisition(self, spec_ei, pi_mc, region_support, tf_dim):
        """Get the sample points

        Args:
            X (np.array): sample points
            y (np.array): corresponding robustness values
            model ([type]): the GP models
            sbo (list): sample points to construct the robustness values
            test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
            region_support (np.array): The bounds of the region within which the sampling is to be done.
                                        Region Bounds is M x N x O where;
                                            M = number of regions;
                                            N = test_function_dimension (Dimensionality of the test function);
                                            O = Lower and Upper bound. Should be of length 2;

        Returns:
            [np.array]: the new sample points by BO
            [np.array]: sbo - new samples for resuse
        """

        lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
        upper_bound_theta = np.ndarray.flatten(region_support[:, 1])
        bounds = (lower_bound_theta, upper_bound_theta)
        
        # bnds = Bounds(lower_bound_theta, upper_bound_theta)
        fun = lambda _x: -1 * self._emi_ei(_x, spec_ei, pi_mc)
        # t = time.perf_counter()
        random_samples = uniform_sampling(2000, region_support, tf_dim, self.rng)
        min_bo_val = -1 * self._emi_ei(
            random_samples, spec_ei, pi_mc, sample_type="multiple"
        )


        
        min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])
        min_bo_val = np.min(min_bo_val)

        
        for _ in range(9):
            new_params = minimize(
                fun,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                method = "L-BFGS-B",
                x0=min_bo,
            )
            # print(new_params)

            if not new_params.success:
                continue

            if min_bo is None or fun(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = fun(min_bo)

        new_params = minimize(
            fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        
        min_bo = new_params.x
        """
        local_minimier_time = time.perf_counter() - t
        
        t = time.perf_counter()
        options = {'c1':0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p':2}
        # bnds = Bounds(lower_bound_theta, upper_bound_theta)
        fun = lambda _x: -1 * self._emi_ei(_x, spec_ei, pi_mc)
        optimizer = GlobalBestPSO(n_particles = 200, dimensions=tf_dim, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(fun, iters=50)
        pso_time = time.perf_counter() - t
        print("********************")
        print("********************")
        print("********************")

        print(f"Time Taken for restart = {local_minimier_time}")
        print(min_bo)
        print(self._emi_ei(np.array(min_bo), spec_ei, pi_mc))
        print("***************************")
        print(f"Time Taken for PSO = {pso_time}")
        print(pos)
        print(cost)
        
        print("********************")
        print("********************")
        # print("********************")
        # print(pos)
        # print(vdfavd)
        """
        return np.array(min_bo)

    
    