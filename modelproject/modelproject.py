from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})


class PrincipalAgentClass():

    def __init__(self, do_print=False):
        """  setup the model """

        if do_print: print('initializing the model:')

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.pi_0 = 0.4  # probability of high sales when employee is hungover
        par.pi_1 = 0.7  # probability of high sales when employee is not hungover
        par.c = 1.0  # employee's cost of providing hard effort/not going out
        par.ubar = 5.0  # reservation utility level
        par.rho = 0.5 # CRRA Parameter
        par.M = 0.3 # cost of monitoring the agent 
        par.highsales = 200 # sales when high
        par.lowsales = 50 # sales when low

        # c. solution 
        shape = np.zeros([1,1])

        sol.wH_high_effort = shape + np.nan # optimal wH, the agent works hard 
        sol.wL_high_effort = shape + np.nan # optimal wL, the agent works hard 
        sol.profit_high_effort = shape + np.nan # resulting profit, the agent works hard 

        sol.wH_loweffort = shape + np.nan # optimal wH, the agent does not work hard 
        sol.wL_loweffort = shape + np.nan # optimal wL,the agent does not work hard 
        sol.profit_low_effort = shape + np.nan # resulting profit, the agent does not work hard 

        sol.wH = shape + np.nan # optimal wH (can both be wH_high_effort or wH_loweffort)
        sol.wL = shape + np.nan # optimal wL (can both be wL_high_effort or wL_loweffort)
        sol.high_effort = shape + np.nan # dummy for optimal to "make" the agent work hard
        sol.profit = shape + np.nan # highest profit achievable 

    def high_sales_prob(self, e):
        """ probability of high sales given the employee's effort level """

        # a. unpack
        par = self.par
        if e == 0:
            return par.pi_0
        else:
            return par.pi_1
    
    def expected_wage(self, wH, wL,e):
        """ calculate the expected wage of the employee given the wage contract\
            and effort level """

        if e==0:
            wage = self.high_sales_prob(0) * wH + (1 - self.high_sales_prob(0)) * wL
        else:
            wage = self.high_sales_prob(1) * wH + (1 - self.high_sales_prob(1)) * wL
        
        return wage
    
    def expected_profit(self, wH, wL, e, monitoring):
        """ calculate the expected proft of the principal given the wage contract,\
            effort level and monitoring decision """

        # a. unpack
        par = self.par 
        if e == 0:
            expect_profit = \
                self.high_sales_prob(0)*par.highsales \
                + (1-self.high_sales_prob(0))*par.lowsales \
                - self.expected_wage(wH, wL,0) - par.M*monitoring
        else: 
            expect_profit = \
                self.high_sales_prob(1)*par.highsales \
                + (1-self.high_sales_prob(1))*par.lowsales \
                - self.expected_wage(wH, wL,1) - par.M*monitoring
        
        return expect_profit

    def calc_utility(self,wH,wL,e):
        """ calculate the expected utility of the agent given the wage contract,\
            effort level """
        
        # a. unpack
        par = self.par
        utility = \
            self.high_sales_prob(e) *( wH**(1-par.rho) /(1-par.rho) - par.c*e) \
            + (1-self.high_sales_prob(e))*(wL**(1-par.rho)/(1-par.rho) -par.c*e)
        
        return utility
    
    def IR_constraint(self,wH,wL,e):
        """ calculate the IR constraint given the wage contract and effort level """
       
        # a. unpack
        par = self.par

        if e==0:
            IR = self.calc_utility(wH,wL,0)-par.ubar
        else:
            IR = self.calc_utility(wH,wL,1)-par.ubar
        return IR 
    
    def IC_constraint(self,wH,wL,high_effort):
        """ calculate the IC constraints given the wage contract"""

        if high_effort==0:
            IC = self.calc_utility(wH,wL,0)-self.calc_utility(wH,wL,1)
        else:
            IC = self.calc_utility(wH,wL,1)-self.calc_utility(wH,wL,0)

        return IC 
        

    def solve_high_effort(self, monitoring=0):
        """ solving model when the principal wants high effort given monitoring decision """

        # a. unpack
        sol = self.sol

        # b. objective function to be minimized
        obj = lambda x: self.expected_wage(x[0],x[1],1)

        # c. setting constraints
        if monitoring == 0:
            cons = [ {'type': 'ineq', 'fun': lambda x:  self.IR_constraint(x[0],x[1],1)}, \
                     {'type': 'ineq', 'fun': lambda x: self.IC_constraint(x[0],x[1],1)}]
        else: 
            cons = [ {'type': 'ineq', 'fun': lambda x:  self.IR_constraint(x[0],x[1],1)}]

        # d. setting bounds 
        bounds = ((0,100),(0,100))

        # d. initial guess
        initial_guess = [12.25,2.25] 

        # e. call optimizer
        res_high = optimize.minimize(obj, initial_guess, constraints=cons, \
                                     bounds = bounds, method='SLSQP', tol=1e-15)

        # f. store results
        sol.wH_high_effort = res_high.x[0]
        sol.wL_high_effort = res_high.x[1]
       
        return res_high
    
    def solve_low_effort(self, monitoring=0):
        """ solving model when the principal wants high effort given monitoring decision """

        # a. unpack
        sol = self.sol

        # b. objective function to be minimized
        obj = lambda x: self.expected_wage(x[0],x[1],0)

        # c. setting constraints
        if monitoring == 0:
            cons = [ {'type': 'ineq', 'fun': lambda x:  self.IR_constraint(x[0],x[1],0)}, \
                     {'type': 'ineq', 'fun': lambda x: self.IC_constraint(x[0],x[1],0)}]
        else: 
            cons = [ {'type': 'ineq', 'fun': lambda x:  self.IR_constraint(x[0],x[1],0)}]

        # d. setting bounds 
        bounds = ((0,100),(0,100))

        # d. initial guess
        initial_guess = [12.25,2.25] 

        # e. call optimizer
        res_low = optimize.minimize(obj, initial_guess, constraints=cons, \
                                    bounds = bounds, method='SLSQP', tol=1e-15)

        # f. store results
        sol.wH_loweffort = res_low.x[0]
        sol.wL_loweffort = res_low.x[1]
       
        return res_low
    
    def solve(self, monitoring = 0):
        """ solving model """
        
        # a. unpack
        sol = self.sol

        # b. calculate profit when the principal wants high effort given monitoring decision
        self.solve_high_effort(monitoring)
        sol.profit_high_effort = self.expected_profit(sol.wH_high_effort,sol.wL_high_effort,1,monitoring)

        # c. calculate profit when the principal wants low effort given monitoring decision
        self.solve_low_effort(monitoring)
        sol.profit_low_effort = self.expected_profit(sol.wH_loweffort,sol.wL_loweffort,0,monitoring)

        # d. compare profit and store the solution with the highest profit
        if sol.profit_high_effort>sol.profit_low_effort:
            sol.wH = sol.wH_high_effort
            sol.wL = sol.wL_high_effort
            sol.high_effort = True 
            sol.profit = sol.profit_high_effort
        
        else: 
            sol.wH = sol.wH_loweffort
            sol.wL = sol.wL_loweffort
            sol.high_effort = True
            sol.profit = sol.profit_low_effort
        
        return sol.wH, sol.wL, sol.high_effort, sol.profit

    def compare(self, parameter,low,high,steps):
        """ produce graphs with wages and profit depending on parameter values """
        
        # a. unpack
        par = self.par
        sol = self.sol

        # b. producing grids and vectors to store the results
        grid = np.linspace(low,high,steps)
        wH = np.nan + np.zeros(steps)
        wL = np.nan + np.zeros(steps)
        profit_low_effort = np.nan + np.zeros(steps)
        profit_high_effort = np.nan + np.zeros(steps)
        
        # c. solve the model for different parameter values
        for i, param_value in enumerate(grid):
            setattr(par, parameter, param_value)
            self.solve()
            wH[i] = sol.wH
            wL[i] = sol.wL 
            profit_low_effort[i] = sol.profit_low_effort
            profit_high_effort[i] = sol.profit_high_effort

        # d. plotting
        fig = plt.figure(figsize=(2*6,6/1.5))

        ax = fig.add_subplot(1,2,1)
        ax.set_title(f'Wages depending on {parameter}')
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