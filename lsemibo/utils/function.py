from staliro.staliro import  simulate_model
import numpy as np
import time
# TODO
# Can the dimensionality of the training set be reduced?



class Evaluator:
    def __init__(self, model, options):
        self.model = model
        self.options = options
        self.sample_history = []
        self.sim_time = {}
        self.monitoring_time = {}
        self.count = 0
        
    def __call__(self, sample, requirement):
        self.count = self.count + 1
        sim_start_time = time.perf_counter()
        result = simulate_model(self.model, self.options, sample)
        self.sim_time[self.count] = time.perf_counter() - sim_start_time

        start_monitoring_time = time.perf_counter()
        all_robustness = requirement.evaluate(result.states, result.times)
        robustness = np.min(list(all_robustness.values()))
        self.monitoring_time[self.count] = time.perf_counter() - start_monitoring_time
        
        self.sample_history.append([self.count, sample, robustness, all_robustness])
        return all_robustness
