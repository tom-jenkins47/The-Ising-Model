import os
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from Methods import *

class Simulation:
    
    """
    Class to perform Ising model simulation.
    Initialises number of sweeps, J, output file name, temperature, system size and dynamic to be used.
    """
    
    def __init__(self, outfile):
        
        self.nsweeps = 10000
        self.J = 1.0
        self.SetParams()
        self.setSpin()
        self.outfile = outfile
    
    def SetParams(self):
        
        """
        Initialises user-defined parameters.
        """
        
        self.kT = float(input('Temperature: '))
        self.N = int(input('System size: '))
        self.Dynamic = int(input('What spin dynamics are to be used? Press 0 for Glauber or 1 for Kawasaki: '))
        
    def setSpin(self):
        
        spin = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                
                r=random.random()
                if(r<0.5):
                    spin[i,j] = -1
                else:
                    spin[i,j] = 1
                    
        self.spin = spin
        
    def UseGlauber(self):
        
        """
        Utilises Glauber spin dynamics to perform the Metropolis test and update the system.
        """
        
        SD = SpinDynamics(self.spin, self.kT)
        deltaE, i, j = SD.Glauber() # Get energy and trial site using Glauber
        
        # Perform Metropolis test
        if deltaE <= 0:
            self.spin[i,j] = -self.spin[i,j]
        
        else: 
            r = random.random()
            if r < +np.exp(-deltaE/self.kT):
                self.spin[i,j] = -self.spin[i,j]

        
    def UseKawasaki(self):
        
        """
        Utilises Kawasaki spin dynamics to perform the Metropolis test and update the system.
        """
        
        SD = SpinDynamics(self.spin, self.kT)
        deltaE, Ai, Aj, Bi, Bj = SD.Kawasaki() # Get energy and trial sites using Kawasaki
        
        # Perform Metropolis test
        if deltaE <= 0:
            self.spin[Ai,Aj] = -self.spin[Ai,Aj]
            self.spin[Bi,Bj] = -self.spin[Bi,Bj]
            
        else:
            r = random.random()
            if r < +np.exp(-deltaE/self.kT):
                self.spin[Ai,Aj] = -self.spin[Ai,Aj]
                self.spin[Bi,Bj] = -self.spin[Bi,Bj]
                
                
    def update(self):
        
        for i in range(self.N):
            for j in range(self.N):
                
                if self.Dynamic == 0:
                    self.UseGlauber()
                    
                elif self.Dynamic == 1:       
                    self.UseKawasaki()
                    
                else:
                    raise ValueError('Select 0 for Glauber or 1 for Kawasaki.')
        

    def RunSim(self, write=False):
        
        """
        Runs a full simulation across 10000 sweeps.
        Writes observables to the output file, and animates the system every 10 sweeps.
        """
        
        Nxy = self.N**2
        
        # Only initialise magnetisation lists if Glauber is selected
        if self.Dynamic == 0:
        
            mags = []
            mags2 = []                            
        
        energies = []                            
        energies2 = []
        
         
        # Iterate through whole grid, across defined number of sweeps
        for n in range(self.nsweeps): 

            # Wait 200 sweeps for equilibration
            if n%10 == 0 and n > 200:
                
                # Determine observable values and write to lists
                Ecalc = CalcEnergy(self.spin)
                totalE = Ecalc.totalE()
                M = np.sum(self.spin)
                
                if self.Dynamic == 0:
        
                    mags.append(M)
                    mags2.append(M**2)
                    
                energies.append(totalE)
                energies2.append(totalE**2)
                
        # Determine average observable values
        if self.Dynamic == 0:
            
            abs_mags = [abs(element) for element in mags]  
            abs_mags_av = np.mean(abs_mags)                         
            mags_av = np.mean(mags)                                 
            mags2_av = np.mean(mags2)     
            sus = (mags2_av-(mags_av**2))/(Nxy*self.kT)                        
        
        energies_av = np.mean(energies)                             
        energies2_av = np.mean(energies2)
        
        C = (energies2_av-(energies_av**2))/(Nxy*(self.kT**2))
        C_err = CalcError(energies, self.kT, Nxy)
        
        
        if write:
            
            # Write observables to output file
            if self.Dynamic == 0:
                
                file_exists = os.path.isfile(f'{self.outfile}')
                with open(f'{self.outfile}', 'a+') as f:
                    if not file_exists: # Only write column headers if file does not already exist
                        f.write('Temperature, Energy, Absolute Magnetisation, Susceptibility, Specific Heat, Delta Specific Heat\n')
                    f.write(f'{self.kT}, {energies_av}, {abs_mags_av}, {sus}, {C}, {C_err}\n')
                    
            else:
                
                file_exists = os.path.isfile(f'{self.outfile}')
                with open(f'{self.outfile}', 'a+') as f:
                    if not file_exists:
                        f.write('Temperature, Energy, Specific Heat, Delta Specific Heat\n')
                    f.write(f'{self.kT}, {energies_av}, {C}, {C_err}\n')
                
                
            print(f'Observables written to {self.outfile}')
        
        
class Animation:
    
    def __init__(self):
        
        # Set up a simulation to be animated
        
        self.sim = Simulation('dumpfile.txt')
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.imshow(self.sim.spin, cmap='gray')
        self.ani = None
        
    def run(self):
        
        # Run the animation, updating every 25ms

        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=25, blit=True)
        plt.show()

    def animate(self, frames):
        
        # Animation function
        
        self.sim.update()
        self.plot.set_data(self.sim.spin)
    
        return (self.plot,)
        
                              
def main():
    
    #outfile_glauber = 'observables_glauber_single.txt'
    #outfile_kawasaki = 'observables_kawasaki_single.txt'
    
    #sim = Simulation(outfile_glauber)
    
    #tstart = time.time()
    
    #sim.RunSim()
    
    #print(f'Elapsed time: {(time.time() - tstart)/60} minutes.')
    
    anim = Animation()
    anim.run()
      
if __name__ == "__main__":
    main()
    
    
    
    
                        
                        
        
                        
                        
                        
                    
                    
                    
            


        
        
        
        
        