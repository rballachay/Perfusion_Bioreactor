#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:40:38 2021

@author: RileyBallachay

Note: See references for values taken from literature
"""

import numpy as np
from scipy.integrate import odeint

"""
# Questions for Coffman:
    1. What is the typical start-up time associated with a perfusion reactor

"""
class chemostat:
    
    def __init__(self):
        "Initialize chemostat parameters"
        
        # Weight of a single cell
        self.CELL = 250*10**(-12) # g/cell [4]
        
        # Glucose concentration of media coming in
        self.c0 = 25 # g/L - [0]
        
        # Flow rate of media coming in
        self.F = 12000/4/24 # L/hr
        
        # Alpha: perfusion rate
        self.a = .5 # - [0]
        
        # Beta: bleed rate
        self.bleed = .3 # - [0]
        
        # Re-calcualte the perfusion rate based on the bleed rate
        # this assumes that bleed is taken off the perfusion
        if self.bleed!=0:
            self.a=self.a-self.bleed
        
        # Yield of biomass on glucose
        self.Yc_s = 1000*1.7*10**6 * self.CELL
        
        # Inoculum density in perfusion bioreactor
        self.Cc0= 1000*20*10**6*self.CELL # g/L - [5] 
        
        # Initial glucose concentration in bioreactor
        self.Cs0 = 50 # g/L - [0]
        
        # Maximum growth rate of CHO cells
        self.mu_max = 0.04 # h-1 - [2]
        
        # Monod constant of CHO cells
        self.Ks = 0.664 # g/L - [2]
        
        # Volume of a single bioreactor
        self.V = 2000 # - [0]
        
        # Yield of lactate on glucose
        self.Yt_s = 1.75 # - [3]
        
        # Alpha in protein production equation
        self.alpha = (7.65*10**(-7))/self.CELL # - [2]
        
        # Beta in protein production equation
        self.beta = 7.65*10**(-8)/self.CELL # - [2]
        
        # Average cell death rate 
        self.kd = 0.001 # - h-1 [0]
        
        # Average toxicity of lactate as rate
        self.kt = 0.0001 # - h-1 [0]
        
        # Maximum oxygen consumption rate
        self.maxX = (120*10**6) * self.CELL # g/L - [1]
        
        # kLa based on [1]
        self.kLa = 57 # h-1 [1]
        
        # Maintain oxygen level at 50% saturation
        self.min = 0.5*8*10**-3 # g/L - [0]
        
        # Specific oxygen uptake rate based on values
        self.qo2 = self.kLa*self.min/self.maxX # - units [1]
        
        # Batch time based on start-up and total run time
        self.start_up = 100 # hr
        self.batch_time = self.start_up + 24*20 # hr
        
    def __chemoSolve(self, z,t): 
        "System of ODEs to be solved for"
        
        # Separate into cell, substrate, product and toxic 
        Cc, Cs, Cp, Ct = z
        
        # Growth and death rate
        rg = self.mu_max*Cs*Cc/(self.Ks+Cs)
        rd = (self.kd + self.kt*Ct)*Cc
        
        # All ODEs
        dCc = (self.a*self.F*self.Cc0*Cc-(1+self.a)*self.F*Cc+self.V*(rg-rd))/self.V
        dCs = (self.F*self.c0+self.a*self.F*Cs-self.V*rg/self.Yc_s-(1+self.a)*self.F*Cs)/self.V
        dCp = (self.V*(self.alpha*dCc+self.beta*Cc)*10**-6 - self.F*Cp)/self.V
        dCt = (self.Yt_s/self.Yc_s)*rg - self.F*Ct/self.V
        
        return [dCc,dCs,dCp,dCt]
    
    
    def solve_ODE(self):
        "Solve system of ODEs using odeint method"
        
        self.t = np.linspace(0, self.batch_time, self.batch_time) 
    
        self.sol = odeint(self.__chemoSolve,(self.Cc0,chemo.Cs0,0,0),self.t)
        
        return self.sol
    
    def plot_data(self):


chemo = chemostat()
sol = solve_ODE()

t = np.linspace(0, 24*20+100, 24*20+100)
z = sol
import matplotlib.pyplot as plt
plt.plot(t, z)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=True)
plt.title('Lotka-Volterra System')
plt.show()



"""[0] No reference. TBD.
"""

"""[1] ICB Framework. Jon Coffman: https://docs.google.com/spreadsheets/d/1GWc_qEpuogR91A4IbRGJYSOTvQOhvvLv6U1eSSDQhbQ/edit?usp=sharing
"""

"""[2]López-Meza, J., Araíz-Hernández, D., Carrillo-Cocom, L.M. et al. Using simple models to describe the kinetics of growth, 
glucose consumption, and monoclonal antibody formation in naive and infliximab producer CHO cells. Cytotechnology 68, 1287–1300 
(2016). https://doi.org/10.1007/s10616-015-9889-2
"""

"""[3]Buchsteiner, Maria et al. “Improving culture performance and antibody production in CHO cell culture processes by reducing 
the Warburg effect.” Biotechnology and bioengineering vol. 115,9 (2018): 2315-2327. doi:10.1002/bit.26724
"""

"""[4] Xie, L., Wang, D.I.C. Applications of improved stoichiometric model in medium design and fed-batch cultivation of animal 
cells in bioreactor. Cytotechnology 15, 17–29 (1994). https://doi-org.ezproxy.library.ubc.ca/10.1007/BF00762376
"""

"""[5] Yongky, Andrew et al. “Process intensification in fed-batch production bioreactors using non-perfusion seed cultures.” 
mAbs vol. 11,8 (2019): 1502-1514. doi:10.1080/19420862.2019.1652075
"""