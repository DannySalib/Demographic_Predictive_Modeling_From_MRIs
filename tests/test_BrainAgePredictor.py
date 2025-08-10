import unittest
from Model_objs.BrainAgePredictor import BrainAgePredictor
from helper_funcs import get_participants_df

class TestBrainAgePredictor(unittest.TestCase):

    def setUp(self):
        self.model = BrainAgePredictor(participants_df=get_participants_df())

