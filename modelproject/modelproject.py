from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

class PrincipalAgentClass():

    def __init__(self, do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace() #Simulation variables

    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. agent
        par.pi0 = 0.5  # probability of high sales when employee is hungover
        par.pie = 0.6  # probability of high sales when employee is not hungover
        par.c = 1.0  # employee's cost of working hard/Not going out
        par.ubar = 0.0  # reservation utility level

        # b. principal
        par.wL = 1.0  # wage when sales are low
        par.wH = 2.0  # wage when sales are high

    def allocate(self):
        """ allocate arrays for simulation """

        par = self.par
        sim = self.sim

        # a. list of variables
        employee = ['e', 'sales']
        principal = ['w', 'expected_wage']
        utility = ['utility']
        allvarnames = employee + principal + utility

        # b. allocate
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan * np.ones(par.simT)

    def expected_sales(self, e):
        """ calculate the expected sales given the employee's effort level """

        par = self.par

        if e == 0:
            return par.pi0
        else:
            return par.pie

    def expected_wage(self, wH, wL, e):
        """ calculate the expected wage given wage contract and employee's effort level """

        par = self.par

        expected_sales = self.expected_sales(e)
        expected_wage = expected_sales * wH + (1 - expected_sales) * wL

        utility = self.utility(wH, wL, e, expected_sales)

        return expected_wage, utility

    def utility(self, wH, wL, e, sales):
        """ calculate the utility of the employee given wage contract, effort level, and sales """

        par = self.par

        wage = sales * wH + (1 - sales) * wL
        utility = (wage ** (1 - par.sigma)) / (1 - par.sigma) - par.c * e

        if utility < par.ubar:
            return par.ubar
        else:
            return utility

    def objective_function(self, x):
        """ calculate the objective function to be minimized """

        wH, wL = x[0], x[1]

        expected_wage_hard, _ = self.expected_wage(wH, wL, 1)
        expected_wage_facebook, _ = self.expected_wage(wH, wL, 0)

        expected_wage = max(expected_wage_hard, expected_wage_facebook)

        return expected_wage

    def solve(self):
        """ solve the model """

        par = self.par
        sim = self.sim

        for t in range(par.simT):

            # 1. principal solves the wage contract
            res = optimize.minimize(self.objective_function, [par.wH, par.wL])
            wH, wL = res.x

            # 2. employee chooses