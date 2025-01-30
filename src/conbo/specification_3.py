from typing import Sequence, Dict, Set, List, Tuple
from abc import ABC, abstractmethod, abstractproperty
from numpy.typing import NDArray

import numpy as np
import time

from staliro.specifications import rtamt, Specification
from staliro.models import Trace
from staliro.cost_func import Result

from .staliroIntegration import Behavior

class Component:
    def __init__(self, identifier:int, spec: str, pred_mapping: Dict[str, int], mapping: NDArray[np.int_]) -> None:
        self.id = identifier
        self.spec = spec
        self.pred_mapping = pred_mapping
        self.specification = rtamt.parse_dense(self.spec, self.pred_mapping)
        self.count = 0
        self.robustness_history = []
        self.active = True
        self.io_mapping = mapping
        self.monitoring_time = []

    def __call__(self, trace:Trace):
        self.count += 1
        start_time = time.perf_counter()
        robustness = self.specification.evaluate(trace).value
        self.monitoring_time.append(time.perf_counter() - start_time)
        self.robustness_history.append([self.count, robustness])
        if robustness <= 0.0:
            self.active = False
        return robustness
    


class BaseRequirement(ABC):
    def __init__(self, tf_dim: int, component_list: List[str], predicate_mapping:Dict[str, Tuple[List[int], int]]) ->  None:
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
            
            
            self.requirements.append(self.handle_component(iter, spec, predicate_mapping_local, mapping))   
          

    @abstractmethod
    def handle_component(self, iter:int, spec:str, predicate_mapping_local:Dict[str, int], mapping:NDArray[np.int_]) -> BaseComponent:
        """Manage Components using this"""
        raise NotImplementedError("handle_component() not implemented")

    def evaluate(self, trace:Trace) -> Result[Dict[int, float], None]:
        self.overall_count += 1
        component_rob = {}
        for iterate in self.active_components:
            result = self.requirements[iterate](trace)
            component_rob[self.requirements[iterate].id] = result
        return Result(value = component_rob, extra=None)

    def __len__(self) -> int:
        return len(self.requirements)

    @property
    @abstractmethod
    def active_components(self) -> Set[int]:
        # Add feature to select active components based on Bhevaior and then go to evaluate
        # if self.behavior in (Behavior.MINIMIZATION, Behavior.FALSIFICATION_AT_ONCE):
        #     active_components = set([req.id for req in self.requirements])
        # else:
        #     active_components = set([req.id for req in self.requirements if req.active])
        # return active_components
        raise NotImplementedError("active_components() not implemented")
    
    @property
    def inactive_components(self) -> Set[int]:
        return set([req.id for req in self.requirements]) - self.active_components

    @property
    def num_active_components(self) -> int:
        return len(self.active_components)

    @property
    def num_inactive_components(self) -> int:
        return len(self.inactive_components)
    
    @property
    def specification_reset(self) -> None:
        tf_dim = self.tf_dim
        component_list = self.component_list
        predicate_mapping = self.predicate_mapping
        self.requirements = []
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
            
            
            self.requirements.append(self.handle_component(iter, spec, predicate_mapping_local, mapping))

    def _get_complete_data(self) -> Dict[int, NDArray[np.float_]]:
        
        y_new = {}
        for iterate in range(len(self.requirements)):
            y_new[iterate] = np.array(self.requirements[iterate].robustness_history)[:,-1]

        return y_new


    def _get_active_data(self) -> Dict[int, NDArray[np.float_]]:
        y_train_active_comp = self.active_components
        y_new = {}
        
        for iterate in y_train_active_comp:
            
            y_new[iterate] = np.array(self.requirements[iterate].robustness_history)[:,-1]

        return y_new

    def _generate_active_dataset(self) -> Tuple[List[int], NDArray[np.float_]]:
        
        robs_data = self._get_active_data()
        idxs = list(robs_data.keys())
        data = np.array(list(robs_data.values())).T

        return idxs, data

    def _get_individual_monitoring_times(self) -> Dict[int, List[np.float_]]:
        indi_monitoring_times = {}
        for req in self.requirements:
            indi_monitoring_times[req.id] = req.monitoring_time
        
        return indi_monitoring_times
    

class MinimizationBehaviorComponent(BaseComponent):
    def __init__(self, identifier: int, spec: str, pred_mapping: Dict[str, int], mapping: NDArray[np.int_]) -> None:
        super().__init__(identifier, spec, pred_mapping, mapping)

    def __call__(self, trace: Trace):
        return super().__call__(trace)
    
class MinimizationBehaviorRequirement(BaseRequirement, Specification[Sequence[float], float, None]):
    def handle_component(self, iter: int, spec: str, predicate_mapping_local: Dict[str, int], mapping: NDArray[np.int_]) -> BaseComponent:
        return MinimizationBehaviorComponent(iter, spec, predicate_mapping_local, mapping)

    @property
    def active_components(self) -> Set[int]:
        return set([req.id for req in self.requirements if req.active])
    

class FalsificationAtOnceBehaviorComponent(BaseComponent):
    def __init__(self, identifier: int, spec: str, pred_mapping: Dict[str, int], mapping: NDArray[np.int_]) -> None:
        super().__init__(identifier, spec, pred_mapping, mapping)

    def __call__(self, trace: Trace):
        return super().__call__(trace)

class FalsificationAtOnceBehaviorRequirement(BaseRequirement, Specification[Sequence[float], float, None]):
    def handle_component(self, iter: int, spec: str, predicate_mapping_local: Dict[str, int], mapping: NDArray[np.int_]) -> BaseComponent:
        return MinimizationBehaviorComponent(iter, spec, predicate_mapping_local, mapping)

    @property
    def active_components(self) -> Set[int]:
        return set([req.id for req in self.requirements if req.active])
    

class FalsificationIterativeBehaviorComponent(BaseComponent):
    def __init__(self, identifier: int, spec: str, pred_mapping: Dict[str, int], mapping: NDArray[np.int_]) -> None:
        super().__init__(identifier, spec, pred_mapping, mapping)

    def __call__(self, trace: Trace):
        robustness = super().__call__(trace)  # Correctly call parent method
        if robustness <= 0.0:
            self.active = False
        return robustness

class FalsificationIterativeBehaviorRequirement(BaseRequirement, Specification[Sequence[float], float, None]):
    def handle_component(self, iter: int, spec: str, predicate_mapping_local: Dict[str, int], mapping: NDArray[np.int_]) -> BaseComponent:
        return MinimizationBehaviorComponent(iter, spec, predicate_mapping_local, mapping)

    @property
    def active_components(self) -> Set[int]:
        return set([req.id for req in self.requirements if req.active])
    


class FalsificationAnyBehaviorComponent(BaseComponent):
    def __init__(self, identifier: int, spec: str, pred_mapping: Dict[str, int], mapping: NDArray[np.int_]) -> None:
        super().__init__(identifier, spec, pred_mapping, mapping)

    def __call__(self, trace: Trace):
        robustness = super().__call__(trace)  # Correctly call parent method
        if robustness <= 0.0:
            self.active = False
        return robustness
    

class FalsificationAnyBehaviorRequirement(BaseRequirement, Specification[Sequence[float], float, None]):
    def handle_component(self, iter: int, spec: str, predicate_mapping_local: Dict[str, int], mapping: NDArray[np.int_]) -> BaseComponent:
        return MinimizationBehaviorComponent(iter, spec, predicate_mapping_local, mapping)

    @property
    def active_components(self) -> Set[int]:
        return set([req.id for req in self.requirements if req.active])
    



