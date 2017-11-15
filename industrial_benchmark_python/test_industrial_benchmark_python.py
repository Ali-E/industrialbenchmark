import unittest
from IDS import IDS
import numpy as np
from numpy import genfromtxt


class TestIB(unittest.TestCase):

    def test_example(self):
        # TODO create result list with fixed seed (Markov State, q Values) import it and compare ist to result of "example"
        print("Test is running")
        np.random.seed(500)
        env = IDS(p=100, inital_seed = 1005)
        at = 2 * np.random.rand(3) - 1  #[-0.72329821, -0.65278997,  0.90032803]
        print(at)
        # eine Aktion ausführen
        env.step(at)
        # Get results: all States and all o-States
        all_States = env.allStates()
        operational_States = env.operational_cost_Buffer()

        print("all States")
        print(env.allStates())
        print("operational costs")
        print(env.operational_cost_Buffer())

        #np.savetxt('all_States.csv', all_States)
        #np.savetxt('operational_States.csv', operational_States)

        muster_all_States = genfromtxt('all_States.csv', delimiter=',')
        muster_operational_States = genfromtxt('operational_States.csv', delimiter=',')


        print("all States Muster")
        print(muster_all_States)
        print("operational Costs Muster")
        print(muster_operational_States)


        # Test if result and muster are equal
        np.testing.assert_array_almost_equal(muster_all_States, all_States)
        np.testing.assert_array_almost_equal(muster_operational_States, operational_States)