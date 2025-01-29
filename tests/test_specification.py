import pathlib
import pickle
from typing import Tuple, Dict, Sequence, Set

import pytest
import numpy as np
import numpy.random as random
from unittest.mock import MagicMock
from numpy.typing import NDArray

from conbo.specification import Requirement, Component
from staliro.models import Trace

@pytest.fixture
def rng() -> random.Generator:
    return np.random.default_rng(42)

@pytest.fixture
def sample_data(rng) -> Trace:
    times = np.arange(0, 10, 0.01)
    vals = rng.random((1000, 5))

    return Trace(times=times, states=vals)

@pytest.fixture
def sample_requirement() -> Requirement:
    phi_list = ["G[0,10] (a<=0.2)", "F[2,10] (b<=0.2)", "G[5,10] (a<=0.2)",
                "F[4,10] (b<=0.2)", "G[5,10] (a<=0.2)", "F[8,10] (b<=0.2)"]
    pred_map = {"a": ([0, 1], 0), "b": ([0, 1], 1)}
    tf_dim = 2
    return Requirement(tf_dim, phi_list, pred_map)


@pytest.fixture
def sample_component() -> Component:
    identifier = 1
    spec = "G[0,10] (a<=0.2)"
    pred_mapping = {"a": 0}
    mapping = np.array([1, 0, 0, 0, 0])
    component = Component(identifier, spec, pred_mapping, mapping)
    component.specification = MagicMock()
    return component

##############################################################################################
# Testing for Component Class
def test_initialization_component(sample_component:Component):
    comp = sample_component
    assert comp.id == 1, "Identifier should be set correctly"
    assert comp.spec == "G[0,10] (a<=0.2)", "Specification should be set correctly"
    assert comp.count == 0, "Initial count should be 0"
    assert not comp.falsified, "Initial falsified state should be False"

def test_call_updates_count(sample_component:Component):
    comp = sample_component
    times = np.array([0, 1, 2])
    states = np.array([[0.1], [0.2], [0.3]])
    comp.specification.evaluate.return_value.value = 0.5
    
    result = comp(Trace(times=times, states=states))
    
    assert comp.count == 1, "Count should be incremented after call"
    assert isinstance(result, float), "Result should be a float"

def test_call_updates_falsified(sample_component:Component):
    comp = sample_component
    times = np.array([0, 1, 2])
    states = np.array([[0.1], [0.2], [0.3]])
    comp.specification.evaluate.return_value.value = -0.1  # Falsified condition
    
    comp(Trace(times=times, states=states))
    
    assert comp.falsified, "Component should be marked as falsified when robustness <= 0"

def test_robustness_history(sample_component:Component):
    comp = sample_component
    times = np.array([0, 1, 2])
    states = np.array([[0.1], [0.2], [0.3]])
    comp.specification.evaluate.return_value.value = 0.5
    
    comp(Trace(times=times, states=states))
    
    assert len(comp.robustness_history) == 1, "History should contain one entry after one evaluation"
    assert comp.robustness_history[0][1] == 0.5, "History should store the correct robustness value"

def test_monitoring_time_updates(sample_component:Component):
    comp = sample_component
    times = np.array([0, 1, 2])
    states = np.array([[0.1], [0.2], [0.3]])
    comp.specification.evaluate.return_value.value = 0.5
    
    comp(Trace(times=times, states=states))
    
    assert len(comp.monitoring_time) == 1, "Monitoring time should be recorded after evaluation"
    assert comp.monitoring_time[0] >= 0, "Monitoring time should be non-negative"

##############################################################################################
# Testing for Requirement Class
def test_initialization(sample_requirement:Requirement):
    reqs = sample_requirement
    assert len(reqs.requirements) == 6, "Requirement list should have 6 components"
    assert reqs.tf_dim == 2, "tf_dim should be initialized correctly"
    assert reqs.overall_count == 0, "Initial overall count should be 0"

def test_evaluate(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement
    
    result = reqs.evaluate(sample_data)
    assert isinstance(result, dict), "Output should be a dictionary"
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in result.items()), "Keys should be int, values should be float"
    assert reqs.overall_count == 1, "Overall count should be incremented"

def test_multiple_evaluations(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement
    
    reqs.evaluate(sample_data)
    reqs.evaluate(sample_data)
    assert reqs.overall_count == 2, "Overall count should increment on multiple evaluations"

def test_edge_case_single_time_step(sample_requirement:Requirement):
    reqs = sample_requirement
    times = np.array([0.0])
    vals = np.random.random((1, 5))
    result = reqs.evaluate(Trace(times=times, states=vals))
    assert isinstance(result, dict), "Output should be a dictionary"

def test_non_numeric_input(sample_requirement:Requirement):
    reqs = sample_requirement
    times = np.array([0, 1, 2])
    vals = np.array([["a", "b", "c", "d", "e"]] * 3)  # Non-numeric values
    with pytest.raises(TypeError):
        reqs.evaluate(Trace(times=times, states=vals))

def test_large_input(sample_requirement:Requirement):
    reqs = sample_requirement
    times = np.arange(0, 1000, 0.01)
    vals = np.random.random((100000, 5))
    result = reqs.evaluate(Trace(times=times, states=vals))
    assert isinstance(result, dict), "Output should be a dictionary"

def test_falsified_components(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement
    reqs.evaluate(sample_data)
    assert isinstance(reqs.falsified_components, Set), "Falsified components should be a set"

def test_unfalsified_components(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement
    reqs.evaluate(sample_data)
    assert isinstance(reqs.unfalsified_components, Set), "Unfalsified components should be a set"

def test_reset_specification(sample_requirement:Requirement):
    reqs = sample_requirement
    reqs.specification_reset
    assert reqs.overall_count == 0, "Overall count should be reset to 0"
    assert len(reqs.requirements) == 6, "Requirements should be reset to initial state"

def test_get_complete_data(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement
    reqs.evaluate(sample_data)
    complete_data = reqs._get_complete_data()
    assert isinstance(complete_data, dict), "Complete data should be a dictionary"

def test_get_unfalsified_data(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement
    reqs.evaluate(sample_data)
    unfalsified_data = reqs._get_unfalsified_data()
    assert isinstance(unfalsified_data, dict), "Unfalsified data should be a dictionary"

def test_generate_unfalsified_dataset(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement
    reqs.evaluate(sample_data)
    idxs, data = reqs._generate_unfaslified_dataset()
    assert isinstance(idxs, list), "Indices should be a list"
    assert isinstance(data, np.ndarray), "Data should be a numpy array"

def test_get_individual_monitoring_times(sample_requirement:Requirement, sample_data: Trace):
    reqs = sample_requirement 
    reqs.evaluate(sample_data)
    monitoring_times = reqs._get_individual_monitoring_times()
    assert isinstance(monitoring_times, dict), "Monitoring times should be a dictionary"

