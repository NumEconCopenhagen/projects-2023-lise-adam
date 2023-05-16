from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})


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
        
        # c. solution 
        shape = np.zeros([1,1])

        sol.wH_high_effort = shape + np.nan
        sol.wL_high_effort = shape + np.nan
        sol.profit_high_effort = shape + np.nan

        sol.wH_loweffort = shape + np.nan
        sol.wL_loweffort = shape + np.nan
        sol.profit_low_effort = shape + np.nan

        sol.wH = shape + np.nan
        sol.wL = shape + np.nan
        sol.high_effort = shape + np.nan
        sol.profit = shape + np.nan

        # d. illustrate, compare
        par.steps = 20


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
        sol.wH_high_effort = res_high.x[0]
        sol.wL_high_effort = res_high.x[1]
       
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
        sol = self.sol

        self.solve_high_effort()
        sol.profit_high_effort = self.expected_profit(sol.wH_high_effort, sol.wL_high_effort, 1)

        self.solve_low_effort()
        sol.profit_low_effort = self.expected_profit(sol.wH_loweffort, sol.wL_loweffort, 0)

        if sol.profit_high_effort>sol.profit_low_effort:
            sol.wH = sol.wH_high_effort
            sol.wL = sol.wL_high_effort
            sol.high_effort = True 
            sol.profit = sol.profit_high_effort
        
        else: 
            sol.wH = sol.wH_loweffort
            sol.wL = sol.wL_loweffort
            sol.effort = True
            sol.profit = sol.profit_low_effort

    def compare(self, parameter, low, high):
        par = self.par
        sol = self.sol
        steps = par.steps

        grid = np.linspace(low,high,steps)

        wH = np.nan + np.zeros(steps)
        wL = np.nan + np.zeros(steps)
        profit_low_effort = np.nan + np.zeros(steps)
        profit_high_effort = np.nan + np.zeros(steps)
        
        for i, param_value in enumerate(grid):
            par.__dict__[parameter] = param_value
            self.solve()
            wH[i] = sol.wH
            wL[i] = sol.wL 
            profit_low_effort[i] = sol.profit_low_effort
            profit_high_effort[i] = sol.profit_high_effort

        fig = plt.figure(figsize=(2*6,6/1.5))

        ax = fig.add_subplot(1,2,1)
        ax.set_title(f'Wages depending on {str(parameter)}')
        ax.plot(grid,wH,label=r'$w_H$')
        ax.plot(grid,wL, label=r'$w_L$')
        ax.set(xlabel =f'{parameter}')
        ax.legend(frameon=True)

        ax = fig.add_subplot(1,2,2)
        ax.set_title(f'Profits depending on {parameter}')
        ax.plot(grid,profit_low_effort,label='Profit low effort')
        ax.plot(grid,profit_high_effort, label='Profit high effort')
        ax.set(xlabel =f'{parameter}')
        ax.legend(frameon=True)
        fig.tight_layout()



    