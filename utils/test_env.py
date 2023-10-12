import unittest
import numpy as np
import random
from CoopEnv import CoopEnv

class test_env(unittest.TestCase):

    def setUp(self):
        # Create an instance of YourClass for testing
        self.env = CoopEnv(n=5,tasks=3)  # Replace with the actual class instantiation#

    def test_characteristic_function(self):

        self.setUp()

        # test 1
        C = {1,2,3}
        singleton_vals = {'Player 1': 1,'Player 2': 2, 'Player 3': 3}
        # bias for random.uniform(0,3) with seed(0) = 2.533265554575144
        coalition_id = hash(tuple(sorted(C)))
        random.seed(coalition_id)
        expected_bias = random.uniform(0,len(C))
        expected_value = 6 * expected_bias

        real_value = self.env.characteristic_function(C, singleton_vals, coalition_id)

        # test2
        C2 = {1}
        singleton_vals2 = {'Player 1': 5}
        # bias for random.uniform(0,3) with seed(0) = 2.533265554575144
        coalition_id2 = hash(tuple(sorted(C2)))
        expected_value2 = 5

        real_value2 = self.env.characteristic_function(C2, singleton_vals2, coalition_id2)


        self.assertEqual(real_value,expected_value)
        self.assertEqual(real_value2,expected_value2)

    def test_get_observations_from_CS(self):

        self.setUp()

        initial_CS = [{'1','3'},{'2','5'},{'4'}]
        locations = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 2, 'Player 5': 1}
        n = 5
        expected_bin_rep = [np.array([1,0,1,0,0]), np.array([0,1,0,0,1]), \
            np.array([1,0,1,0,0]), np.array([0,0,0,1,0]), np.array([0,1,0,0,1])]
        real_bin_rep = self.env.get_observations_from_CS(initial_CS, locations, n)

        np.testing.assert_array_equal(real_bin_rep, expected_bin_rep)



    def test_movement_phase(self):
        # Test the Movement Phase

        self.setUp()

        initial_CS = [{'1','3'},{'2','5'},{'4'}]  # Initialize the state as required for your test
        locations = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 2, 'Player 5': 1}
        n = 5
        actions = [1,0,0,1,0] # Set actions for testing the Movement Phase

        # expected_CS = [{'5','3','2'},{'1','4'},set()]
        expected_out = [np.array([1,0,0,1,0]), np.array([0,1,1,0,1]), \
            np.array([0,1,1,0,1]), np.array([1,0,0,1,0]), np.array([0,1,1,0,1])]

        next_state = self.env.movement_phase(initial_CS, locations, n, actions)

        np.testing.assert_array_equal(next_state, expected_out)

    def test_communication_phase(self):
        # Test the Communication Phase

        initial_CS = [{'1','3'},{'2','5'},{'4'}]  # Initialize the state as required for your test
        singleton_vals = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3,'Player 4': 4, 'Player 5': 5}
        n = 5
        cnf = 0.2

        expected_comm_vals = {'Player 1': 1.1377687406100192, 'Player 2': 2.206363522352242, 'Player 3': 2.904685896997014, 'Player 4': 3.6142668004687413, 'Player 5': 5.022549442737217}
        expected_char_vals = []
        task = 0
        for C in initial_CS:
            val = self.env.characteristic_function(C, singleton_vals, task*n+hash(tuple(sorted(C))))
            expected_char_vals.append(val)
            task+= 1
        expected_sum_vals = [4.04245463761, 7.22891296509, 3.6142668004687413]
        comm_vals, char_vals, sum_vals = self.env.communication_phase(initial_CS, singleton_vals, n, cnf, a=0)

        # to avoid some rounding errors from manual calculation
        expected_sum_vals = [round(elem, 5) for elem in expected_sum_vals]
        sum_vals = [round(elem, 5) for elem in sum_vals]

        # Write assertions to verify the correctness of the Communication Phase results
        self.assertCountEqual(comm_vals, expected_comm_vals)
        self.assertCountEqual(char_vals, expected_char_vals)
        self.assertCountEqual(sum_vals, expected_sum_vals)

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
        initial_CS2 = [{'1','3', '4'},{'2','5'}, set()]
        locs2 = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 0, 'Player 5': 1}
        s_vals2 = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3,'Player 4': 4, 'Player 5': 5}
        comm_vals2 = {'Player 1': 2, 'Player 2': 3, 'Player 3': 4,'Player 4': 5, 'Player 5': 6}
        sum_vals2 = [11, 9, 0]
        char_vals2 = [7, 10, 0]

        expected_rewards2 = np.array([0,3.33333,0,0,6.66667])

        real_rewards2 = self.env.payoff_dist_phase(comm_vals2, char_vals2, sum_vals2, initial_CS2, locs2, s_vals2, n)
        real_rewards2 = np.round(real_rewards2, 5)

        np.testing.assert_array_equal(real_rewards2, expected_rewards2)

        # test 3
        initial_CS3 = [{'1','3', '4'},{'2','5'}, set()]
        locs3 = {'Player 1': 0, 'Player 2': 1, 'Player 3': 0,'Player 4': 0, 'Player 5': 1}
        s_vals3 = {'Player 1': 1, 'Player 2': 2, 'Player 3': 3,'Player 4': 4, 'Player 5': 5}
        comm_vals3 = {'Player 1': 2, 'Player 2': 3, 'Player 3': 4,'Player 4': 5, 'Player 5': 6}
        sum_vals3 = [11, 9, 0]
        char_vals3 = [7, 7, 0]

        expected_rewards3 = np.array([0,0,0,0,0])

        real_rewards3 = self.env.payoff_dist_phase(comm_vals3, char_vals3, sum_vals3, initial_CS3, locs3, s_vals3, n)
        real_rewards3 = np.round(real_rewards3, 5)

        np.testing.assert_array_equal(real_rewards3, expected_rewards3)

    def tearDown(self):
        # Clean up any resources if needed
        pass

if __name__ == '__main__':
    unittest.main()