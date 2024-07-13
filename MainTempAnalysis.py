import os
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from Methods import *
matplotlib.use('TKAgg') # Opens GUI for simulation


class Simulation:
    
    """
    Class to perform Ising model simulation.
    Initialises number of sweeps, J, output file name, temperature, system size and dynamic to be used.
    """
    
    def __init__(self, outfile, kT, N, Dynamic):
        
        self.nsweeps = 10000
        self.J = 1.0
        self.outfile = outfile
        self.kT = kT
        self.N = N
        self.Dynamic = Dynamic
        
    def UseGlauber(self, spin):
        
        """
        Utilises Glauber spin dynamics to perform the Metropolis test and update the system.
        """
        
        SD = SpinDynamics(spin, self.kT)
        deltaE, i, j = SD.Glauber() # Get energy and trial site using Glauber
        
        # Perform Metropolis test
        if deltaE <= 0:
            spin[i,j] = -spin[i,j]
        
        else: 
            r = random.random()
            if r < +np.exp(-deltaE/self.kT):
                spin[i,j] = -spin[i,j]
            
        return spin
        
    def UseKawasaki(self, spin):
        
        """
        Utilises Kawasaki spin dynamics to perform the Metropolis test and update the system.
        """
        
        SD = SpinDynamics(spin, self.kT)
        deltaE, Ai, Aj, Bi, Bj = SD.Kawasaki() # Get energy and trial sites using Kawasaki
        
        # Perform Metropolis test
        if deltaE <= 0:
            spin[Ai,Aj] = -spin[Ai,Aj]
            spin[Bi,Bj] = -spin[Bi,Bj]
            
        else:
            r = random.random()
            if r < +np.exp(-deltaE/self.kT):
                spin[Ai,Aj] = -spin[Ai,Aj]
                spin[Bi,Bj] = -spin[Bi,Bj]
            
        return spin
              
    def RunSim(self):
        
        """
        Runs a full simulation across 10000 sweeps.
        Writes observables to the output file, and animates the system every 10 sweeps.
        """
        
        Nx = self.N
        Ny = self.N
        Nxy = Nx*Ny
        
        # Only initialise magnetisation lists if Glauber is selected
        if self.Dynamic == 0:
        
            mags = []
            mags2 = []                            
        
        energies = []                            
        energies2 = []
        
        # Initialise spins randomly
        spin = np.zeros((Nx,Ny))
        
        for i in range(Nx):
            for j in range(Ny):
                
                r=random.random()
                if(r<0.5):
                    spin[i,j] = -1
                else:
                    spin[i,j] = 1
        
        # Set up animation
        anim = plt.imshow(spin, animated=True, origin='lower')
        
        # Iterate through whole grid, across defined number of sweeps
        for n in range(self.nsweeps): 
            for i in range(Nx):
                for j in range(Ny):
                    
                    if self.Dynamic == 0:
                        spin = self.UseGlauber(spin)
                        
                    elif self.Dynamic == 1:       
                        spin = self.UseKawasaki(spin)
                        
                    else:
                        raise ValueError('Select 0 for Glauber or 1 for Kawasaki.')
                        
            if n%10 == 0:
                
                # Update animation every 10 sweeps
                plt.cla()
                anim = plt.imshow(spin, animated=True, origin='lower')
                plt.draw()
                plt.pause(0.0001)
             
            # Wait 200 sweeps for equilibration
            if n%10 == 0 and n > 200:
                
                # Determine observable values and write to lists
                Ecalc = CalcEnergy(spin)
                totalE = Ecalc.totalE()
                M = np.sum(spin)
                
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
        
        
def TempAnalysis(outfile):
    
    """
    Function to run simulations for a range of temperature values.
    """
    
    N = int(input('System size: '))
    Dynamic = int(input('What spin dynamics are to be used? Press 0 for Glauber or 1 for Kawasaki: '))
    
    T_range = np.arange(1.0, 3.1, 0.1)
    T_range = np.round(T_range, 2) # Round temperature values
    
    # Run a simulation at each temperature step
    for kT in T_range:
        
        sim = Simulation(outfile, kT, N, Dynamic)
        sim.RunSim()
        
        print(f'Temperature value {kT} completed.')
        
                              
def main():
    
    outfile_glauber = 'observables_glauber.txt'
    outfile_kawasaki = 'observables_kawasaki.txt'
    
    tstart = time.time()
    
    # Change argument below depending on which dynamic is going to be used
    TempAnalysis(outfile_glauber)
    
    print(f'Elapsed time: {(time.time() - tstart)/60} minutes.')
    
if __name__ == "__main__":
   
    main()
    
      
