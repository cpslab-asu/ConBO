import numpy as np
import unittest
from lsemibo.utils import sample_spec

class TestSampleSpec(unittest.TestCase):
    def test1_sample_spec(self):
        unclassified_spec = [1,2,3,4,5]
        classified_spec = [6,7,8,9]
        spec_prob = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0.1, 7:0.5, 8:0.2, 9:0.2}
        top_k = 9
        bias = 0.7
        rng = np.random.default_rng(12340)
        sampled_specs = sample_spec(spec_prob, unclassified_spec, classified_spec, top_k, bias, rng)
        # print(sampled_specs)