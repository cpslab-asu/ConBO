from typing import Sequence, Dict, Set, List, Tuple
from numpy.typing import NDArray

import numpy as np
import time

from staliro.specifications import rtamt, Specification
from staliro.models import Trace
from staliro.cost_func import Result

class Component:
    def __init__(self, identifier:int, spec: str, pred_mapping: Dict[str, int], mapping: NDArray[np.int_]) -> None:
        self.id = identifier
        self.spec = spec
        self.pred_mapping = pred_mapping
        self.specification = rtamt.parse_dense(self.spec, self.pred_mapping)
        self.count = 0
        self.robustness_history = []
        self.falsified = False
        self.io_mapping = mapping
        self.monitoring_time = []

    def __call__(self, trace:Trace):
        self.count += 1
        start_time = time.perf_counter()
        robustness = self.specification.evaluate(trace).value
        self.monitoring_time.append(time.perf_counter() - start_time)
        self.robustness_history.append([self.count, robustness])
        if robustness <= 0.0:
            self.falsified = True
        
        return robustness

class Requirement(Specification[Sequence[float], float, None]):
    def __init__(self, tf_dim: int, component_list: List[str], predicate_mapping:Dict[str, Tuple[List, int]]) ->  None:
        self.tf_dim = tf_dim
        self.component_list = component_list
        self.predicate_mapping = predicate_mapping
        self.requirements:List[Component] = []
        self.overall_count = 0
        
        for iter, spec in enumerate(component_list):
            predicate_mapping_local = {}
            input_indices_component = []
            for var in predicate_mapping.keys():
                if var in spec:
                    index, predicate_mapping_local[var] = predicate_mapping[var]
                    if index not in input_indices_component:
                        input_indices_component += index
            mapping = np.array([1 if item in input_indices_component else 0 for item in range(tf_dim)])
            
            
            self.requirements.append(Component(iter, spec, predicate_mapping_local, mapping))   
    
    def evaluate(self, trace:Trace) -> Result[Dict[int, float], None]:
        # states = trace.states
        # times = trace.times
        # if states.shape[0] == 0 or times.shape[0] == 0 or states.shape[0] != times.shape[0]:
        #     raise ValueError("states and times have invlid shape.")
        self.overall_count += 1
        component_rob = {}
        
        num_requirements = len(self.requirements)

        for iterate in range(num_requirements):
            if not self.requirements[iterate].falsified:
                result = self.requirements[iterate](trace)
                component_rob[self.requirements[iterate].id] = result
        return Result(value = component_rob, extra=None)

    def __len__(self) -> int:
        return len(self.requirements)

    @property
    def falsified_components(self) -> Set[int]:
        return {req.id for req in self.requirements if req.falsified}

    @property
    def unfalsified_components(self) -> Set[int]:
        return {req.id for req in self.requirements if not req.falsified}

    @property
    def num_falsified_components(self) -> int:
        return len(self.falsified_components)

    @property
    def num_unfalsified_components(self) -> int:
        return len(self.unfalsified_components)
    
    @property
    def specification_reset(self) -> None:
        tf_dim = self.tf_dim
        component_list = self.component_list
        predicate_mapping = self.predicate_mapping
        self.requirements = []
        # self.falsified_reqs = []
        self.overall_count = 0
        # self.num_components = len(component_list)
        
        for iter, spec in enumerate(component_list):
            predicate_mapping_local = {}
            input_indices_component = []
            for var in predicate_mapping.keys():
                if var in spec:
                    index, predicate_mapping_local[var] = predicate_mapping[var]
                    if index not in input_indices_component:
                        input_indices_component += index
                    # input_indices.update(index)
            mapping = np.array([1 if item in input_indices_component else 0 for item in range(tf_dim)])
            
            
            self.requirements.append(Component(iter, spec, predicate_mapping_local, mapping))

    def _get_complete_data(self) -> Dict[int, NDArray[np.float_]]:
        
        y_new = {}
        for iterate in range(len(self.requirements)):
            y_new[iterate] = np.array(self.requirements[iterate].robustness_history)[:,-1]

        return y_new


    def _get_unfalsified_data(self) -> Dict[int, NDArray[np.float_]]:
        y_train_active_comp = self.unfalsified_components
        y_new = {}
        
        for iterate in y_train_active_comp:
            
            y_new[iterate] = np.array(self.requirements[iterate].robustness_history)[:,-1]

        return y_new

    def _generate_unfaslified_dataset(self) -> Tuple[List[int], NDArray[np.float_]]:
        
        robs_data = self._get_unfalsified_data()
        idxs = list(robs_data.keys())
        data = np.array(list(robs_data.values())).T

        return idxs, data

    def _get_individual_monitoring_times(self) -> Dict[int, List[np.float_]]:
        indi_monitoring_times = {}
        for req in self.requirements:
            indi_monitoring_times[req.id] = req.monitoring_time
        
        return indi_monitoring_times
        