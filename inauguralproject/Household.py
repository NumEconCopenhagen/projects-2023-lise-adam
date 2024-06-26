from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """ 
        # The function that is called, when the model is formed

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        opt = self.opt = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.omega = 0.5 
        par.epsilon_w = 1.0
        par.epsilon_m = 1.0
        par.gender = 0.0 #dummy for whether we use model in Question 5 or not.

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        sol.alpha_estimate=np.nan
        sol.sigma_estimate=np.nan
        sol.epsilon_w_estimate=np.nan
        sol.epsilon_m_estimate=np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """
        par = self.par

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma==0:
            H = min(HM,HF)

        elif par.sigma==1:
            H = HM**(1-par.alpha)*HF**par.alpha

        elif par.sigma<1: 
            # avoid diving with zero
            H = ((1-par.alpha)*(HM+1e-8) **((par.sigma-1)/par.sigma) + par.alpha * (HF+1e-8)**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        else:
            H = ((1-par.alpha)*HM **((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_w = 1+1/par.epsilon_w
        epsilon_m = 1+1/par.epsilon_m

        TM = LM+HM
        TF = LF+HF            
        disutility = par.nu*(TM**epsilon_m/epsilon_m+TF**epsilon_w/epsilon_w)

        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        opt = self.opt

        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt


    def solve(self):
        """ solve model continously """
        sol = self.sol
        
        # a. objective function: negative since we minimize
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])

        # b. setting constraints
        cons = [{'type': 'ineq', 'fun': lambda x:  x[0]+x[1] -24},
        {'type': 'ineq', 'fun': lambda x: x[2]+x[3]-24}]
        
        # c. bounds on hours 
        bounds = [(0,24)]*4
            
        # d. initial guess
        initial_guess = [6,6,6,6] 

        # e. call optimizer
        res = optimize.minimize(obj, initial_guess, method='SLSQP', bounds=bounds)

        # f. store results
        sol.LM = res.x[0]
        sol.HM = res.x[1]
        sol.LF = res.x[2]
        sol.HF = res.x[3]

        return res
            

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
        opt = self.opt

        # a. solving model for different female wages (using either a discrete or continous solution)
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF
            if discrete==True:
                self.solve_discrete()
                sol.LM_vec[i] = opt.LM
                sol.HM_vec[i] = opt.HM
                sol.LF_vec[i] = opt.LF
                sol.HF_vec[i] = opt.HF
            if discrete==False:
                self.solve()
                sol.LM_vec[i] = sol.LM
                sol.HM_vec[i] = sol.HM
                sol.LF_vec[i] = sol.LF
                sol.HF_vec[i] = sol.HF

    def run_regression(self):
        """ run regression """
        par = self.par
        sol = self.sol

        # a. creating log transformatons
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        sol = self.sol
        par = self.par
        
        # a. different initial guesses for our models
        if par.gender==1:
            init = [0.5, 1, 1]
        else:
            init = [0.5,0.5]
    
        # b. call optimizer
        res=optimize.minimize(self.objective_regression, x0 = init, method = 'Nelder-Mead' )

        # c. storing results
        if par.gender==1:
            sol.sigma_estimate = res.x[0]
            sol.epsilon_w_estimate = res.x[1]
            sol.epsilon_m_estimate = res.x[2]

        else:
            sol.alpha_estimate = res.x[0]
            sol.sigma_estimate = res.x[1]
        

                    
    def objective_regression(self, x):
        par = self.par
        sol = self.sol

        # a. runinning regression
        if par.gender==1:
            par.sigma = x[0]
            par.epsilon_w = x[1]
            par.epsilon_m = x[2]

        else:
            par.alpha = x[0]
            par.sigma = x[1]

        self.solve_wF_vec()
        self.run_regression()

        # b. defining the deviation function we minimize
        obj = (sol.beta0-par.beta0_target)**2 + (sol.beta1-par.beta1_target)**2

        return obj