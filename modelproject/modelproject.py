from types import SimpleNamespace
import numpy as np
from scipy import optimize

class PrincipalAgentClass():

    def __init__(self, do_print=True):
        """  setup the model """

        if do_print: print('initializing the model:')

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.pi_0 = 0.3  # probability of high sales when employee is hungover
        par.c = 1.0  # employee's cost of working hard/Not going out
        par.ubar = 5.0  # reservation utility level
        par.rho = 0.5 # CRRA Parameter
        par.e = 1.0 # Level of effort (= 1 when not hungover, =0 when hungover)

        par.pi_1 = 0.7  # probability of high sales when employee is not hungover
        
        # c. solution VED IKK EHELT HVAD DET SKAL VÆRE HER. Vi skal have en vektor der holder løsningen på wH og wL
        sol.wH = np.zeros([1,1]) + np.nan
        sol.wL = np.zeros([1,1]) + np.nan

    def high_sales_prob(self, e):
        """ expected sales given the employee's effort level """

        par = self.par
        if e == 0:
            return par.pi_0
        else:
            return par.pi_1
    
    def expected_wage(self, wH, wL):
        """ calculate the expected wage and the utility of the employee given the wage contract, effort level and probability of high sales"""

        par = self.par

        wage = self.high_sales_prob(1) * wH + (1 - self.high_sales_prob(1)) * wL
        
        return wage
    
    def calc_utility(self,wH,wL,e):
        par = self.par
        utility = self.high_sales_prob(e) * (np.power(wH,(1-par.rho + 1e-12)) /(1-par.rho + 1e-12)-par.c*e)+(1-self.high_sales_prob(e))*(np.power(wL,(1-par.rho + 1e-12)/(1-par.rho + 1e-12))-par.c*e)
        return utility

    def solve(self):
        """ Solving model """

        # a. unpack
        par = self.par
        sol = self.sol

        # b. objective function to be minimized
        obj = lambda x: self.expected_wage(x[0],x[1])

        # c. Setting constraints
        cons = [ {'type': 'ineq', 'fun': lambda x:  self.calc_utility(x[0],x[1],1)-par.ubar}, {'type': 'ineq', 'fun': lambda x: self.calc_utility(x[0],x[1],1)-self.calc_utility(x[0],x[1],0)}]

        # d. setting bounds 
        bounds = ((0,15),(0,15))

        # d. initial guess
        initial_guess = [12.25,2.25] 

        # e. call optimizer
        res=optimize.minimize(obj, initial_guess, constraints=cons, bounds = bounds, method='SLSQP', tol=1e-12)

        # f. store results
        sol.wH = res.x[0]
        sol.wL= res.x[1]
       
        return res