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

        par.highsales = 100
        par.lowsales = 0

        par.pi_1 = 0.7  # probability of high sales when employee is not hungover
        
        # c. solution VED IKK EHELT HVAD DET SKAL VÆRE HER. Vi skal have en vektor der holder løsningen på wH og wL
        sol.wH_higheffort = np.zeros([1,1]) + np.nan
        sol.wL_higheffort = np.zeros([1,1]) + np.nan

        sol.wH_loweffort = np.zeros([1,1]) + np.nan
        sol.wL_loweffort = np.zeros([1,1]) + np.nan

        sol.wH = np.zeros([1,1]) + np.nan
        sol.wL = np.zeros([1,1]) + np.nan
        sol.effort = np.zeros([1,1]) + np.nan

        sol.profit_high_effort = np.zeros([1,1]) + np.nan
        sol.profit_low_effort = np.zeros([1,1]) + np.nan


    def high_sales_prob(self, e):
        """ expected sales given the employee's effort level """

        par = self.par
        if e == 0:
            return par.pi_0
        else:
            return par.pi_1
    
    def expected_wage(self, wH, wL,e):
        """ calculate the expected wage and the utility of the employee given the wage contract, effort level and probability of high sales"""

        par = self.par

        if e==0:
            wage = self.high_sales_prob(0) * wH + (1 - self.high_sales_prob(0)) * wL
        else:
            wage = self.high_sales_prob(1) * wH + (1 - self.high_sales_prob(1)) * wL
        
        return wage
    
    def expected_profit(self, wH, wL, e):
        
        par = self.par 
        if e == 0:
            expect_profit = self.high_sales_prob(0)*par.highsales + (1-self.high_sales_prob(0))*par.lowsales - self.expected_wage(wH, wL,0)
        else: 
            expect_profit = self.high_sales_prob(1)*par.highsales + (1-self.high_sales_prob(1))*par.lowsales - self.expected_wage(wH, wL,1)
        
        return expect_profit

    def calc_utility(self,wH,wL,e):
        par = self.par
        utility = self.high_sales_prob(e) *( wH**(1-par.rho) /(1-par.rho) - par.c*e)+(1-self.high_sales_prob(e))*(wL**(1-par.rho)/(1-par.rho) -par.c*e)
        return utility

    def solve_high_effort(self):
        """ Solving model """

        # a. unpack
        par = self.par
        sol = self.sol

        # b. objective function to be minimized
        obj = lambda x: self.expected_wage(x[0],x[1],1)

        # c. Setting constraints
        cons = [ {'type': 'ineq', 'fun': lambda x:  self.calc_utility(x[0],x[1],1)-par.ubar}, {'type': 'ineq', 'fun': lambda x: self.calc_utility(x[0],x[1],1)-self.calc_utility(x[0],x[1],0)}]

        # d. setting bounds 
        bounds = ((0,100),(0,100))

        # d. initial guess
        initial_guess = [12.25,2.25] 

        # e. call optimizer
        res_high = optimize.minimize(obj, initial_guess, constraints=cons, bounds = bounds, method='SLSQP', tol=1e-12)

        # f. store results
        sol.wH_higheffort = res_high.x[0]
        sol.wL_higheffort = res_high.x[1]
       
        return res_high
    
    def solve_low_effort(self):
        """ Solving model """

        # a. unpack
        par = self.par
        sol = self.sol

        # b. objective function to be minimized
        obj = lambda x: self.expected_wage(x[0],x[1],0)

        # c. Setting constraints
        cons = [ {'type': 'ineq', 'fun': lambda x:  self.calc_utility(x[0],x[1],0)-par.ubar}, {'type': 'ineq', 'fun': lambda x: self.calc_utility(x[0],x[1],0)-self.calc_utility(x[0],x[1],1)}]

        # d. setting bounds 
        bounds = ((0,100),(0,100))

        # d. initial guess
        initial_guess = [12.25,2.25] 

        # e. call optimizer
        res_low = optimize.minimize(obj, initial_guess, constraints=cons, bounds = bounds, method='SLSQP', tol=1e-12)

        # f. store results
        sol.wH_loweffort = res_low.x[0]
        sol.wL_loweffort = res_low.x[1]
       
        return res_low
    
    def solve(self):
        # a. unpack
        par = self.par
        sol = self.sol

        self.solve_high_effort()
        sol.profit_high_effort = self.expected_profit(sol.wH_higheffort, sol.wL_higheffort, 1)

        self.solve_low_effort()
        sol.profit_low_effort = self.expected_profit(sol.wH_loweffort, sol.wL_loweffort, 0)

        if sol.profit_high_effort>sol.profit_low_effort:
            sol.wH = sol.wH_higheffort
            sol.wL = sol.wL_higheffort
            sol.effort = 1 
        
        else: 
            sol.wH = sol.wH_loweffort
            sol.wL = sol.wL_loweffort
            sol.effort = 0


    