import unittest
import numpy as np
import random
from CoopEnv import CoopEnv

class test_env(unittest.TestCase):

    def setUp(self):

        self.env = CoopEnv(n=5,tasks=3)

    def test_characteristic_function(self):

        self.setUp()

        # test 1 - normal functionality with arbitrary seed
        C = {1,2,3}
        singleton_vals = {'Player 1': 1,'Player 2': 2, 'Player 3': 3}
        seed = 0
        expected_bias = 2.533265554575144
        expected_val = 6 * expected_bias
        real_val = self.env.characteristic_function(C, singleton_vals, seed)
        self.assertEqual(real_val, expected_val)

        # test2 - special case of length 1 coalition
        C = {1}
        singleton_vals = {'Player 1': 5}
        seed = 1
        expected_value = 5
        real_value = self.env.characteristic_function(C, singleton_vals, seed)
        self.assertEqual(real_value,expected_value)

    def test_get_observations_from_CS(self):

        self.setUp()

        CS = [{'1','3'},{'2','5'},{'4'}]
        locations = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 2, 'Player 5': 1}
        n = 5
        expected_rep = [np.array([1,0,1,0,0]), np.array([0,1,0,0,1]), \
            np.array([1,0,1,0,0]), np.array([0,0,0,1,0]), np.array([0,1,0,0,1])]
        real_rep = self.env.get_observations_from_CS(CS, locations, n)

        np.testing.assert_array_equal(real_rep, expected_rep)

    def test_movement_phase(self):
        # Test the Movement Phase

        self.setUp()

        CS = [{'1','3'},{'2','5'},{'4'}]  # Initialize the state as required for your test
        locations = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 2, 'Player 5': 1}
        n = 5
        actions = [1,0,0,1,0] # Set actions for testing the Movement Phase

        # expected_CS = [{'5','3','2'},{'1','4'},set()]
        expected_state = [np.array([1,0,0,1,0]), np.array([0,1,1,0,1]), \
            np.array([0,1,1,0,1]), np.array([1,0,0,1,0]), np.array([0,1,1,0,1])]

        real_state = self.env.movement_phase(CS, locations, n, actions)

        np.testing.assert_array_equal(real_state, expected_state)

    def test_communication_phase(self):
        # Test the Communication Phase

        CS = [{'1','3'},{'2','5'},{'4'}]  # Initialize the state as required for your test
        singleton_vals = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3,'Player 4': 4, 'Player 5': 5}
        n = 5
        cnf = 0

        expected_comm_vals = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3, 'Player 4': 4, 'Player 5': 5}
        expected_char_vals = [7.648274175113995, 4.533658707664273, 4]
        expected_sum_vals = [4, 7, 4]

        real_comm_vals, real_char_vals, real_sum_vals = self.env.communication_phase(CS, singleton_vals, n, cnf)

        # Write assertions to verify the correctness of the Communication Phase results
        self.assertCountEqual(real_comm_vals, expected_comm_vals)
        self.assertCountEqual(real_char_vals, expected_char_vals)
        self.assertCountEqual(real_sum_vals, expected_sum_vals)

    def test_payoff_distribution_phase(self):
        # Test the Payoff Distribution Phase
        # test 1
        n=5
        initial_CS = [{'1','3'},{'2','5'},{'4'}]
        locs = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 2, 'Player 5': 1}
        s_vals = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3,'Player 4': 4, 'Player 5': 5}
        comm_vals= {'Player 1': 2, 'Player 2': 3, 'Player 3': 4,'Player 4': 5, 'Player 5': 6}
        sum_vals = [6, 9, 5]
        char_vals = [10, 20, 5]

        expected_rewards = np.array([3.33333,6.66667,6.66667,5,13.33333])

        real_rewards = self.env.payoff_dist_phase(comm_vals, char_vals, sum_vals, initial_CS, locs, s_vals, n)
        real_rewards = np.round(real_rewards, 5)

        # Write assertions to verify the correctness of the Payoff Distribution Phase results
        np.testing.assert_array_equal(real_rewards, expected_rewards)


        # test 2
        initial_CS = [{'1','3', '4'},{'2','5'}, set()]
        locs = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 0, 'Player 5': 1}
        s_vals = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3,'Player 4': 4, 'Player 5': 5}
        comm_vals = {'Player 1': 2, 'Player 2': 3, 'Player 3': 4,'Player 4': 5, 'Player 5': 6}
        sum_vals = [11, 9, 0]
        char_vals = [7, 10, 0]

        expected_rewards = np.array([0,3.33333,0,0,6.66667])

        real_rewards = self.env.payoff_dist_phase(comm_vals, char_vals, sum_vals, initial_CS, locs, s_vals, n)
        real_rewards = np.round(real_rewards, 5)

        np.testing.assert_array_equal(real_rewards, expected_rewards)

        # test 3
        initial_CS = [{'1','3', '4'},{'2','5'}, set()]
        locs = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 0, 'Player 5': 1}
        s_vals = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3,'Player 4': 4, 'Player 5': 5}
        comm_vals = {'Player 1': 2, 'Player 2': 3, 'Player 3': 4,'Player 4': 5, 'Player 5': 6}
        sum_vals = [11, 9, 0]
        char_vals = [7, 7, 0]

        expected_rewards = np.array([0,0,0,0,0])

        real_rewards = self.env.payoff_dist_phase(comm_vals, char_vals, sum_vals, initial_CS, locs, s_vals, n)
        real_rewards = np.round(real_rewards, 5)

        np.testing.assert_array_equal(real_rewards, expected_rewards)

    def tearDown(self):
        # Clean up any resources if needed
        pass

if __name__ == '__main__':
    unittest.main()