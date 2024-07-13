import numpy as np 
import random

class CalcEnergy:
    
    """
    Class used to perform energy-related calculations.
    Initialises spin config., J and grid parameters (Nx, ly).
    """
    
    def __init__(self, spin):
        
        self.spin = spin
        self.Nx = spin.shape[0]
        self.Ny = spin.shape[1]
        self.J = 1.0
        
    def deltaE(self, i, j):
        
        """
        Returns delta E for a given site.
        """
        
        return 2*self.J*self.spin[i,j]*(self.spin[(i+1)%self.Nx,j]\
        + self.spin[i,(j+1)%self.Ny] + self.spin[(i-1)%self.Nx,j] 
        + self.spin[i,(j-1)%self.Ny])
            
    def totalE(self):
        
        """
        Returns total system energy for a given spin state.
        """
        
        E = 0.0
    
        for i in range(self.Nx):
            for j in range(self.Ny):
                
                Si = self.spin[i,j]
                Sj = self.spin[(i+1)%self.Nx,j] + self.spin[i,(j+1)%self.Ny]\
                + self.spin[(i-1)%self.Nx,j] + self.spin[i,(j-1)%self.Ny]
                E += -self.J*Si*Sj
        
        return 0.5*E # Factor accounts for nearest neighbours
    
class SpinDynamics:
    
    """
    Class to determine delta E using either Glauber or Kawasaki spin dynamics.
    Initialises spin config., J, grid parameters (Nx, ly) and temperature (kT).
    """
    
    def __init__(self, spin, kT):
        
        self.spin = spin
        self.Nx = spin.shape[0]
        self.Ny = spin.shape[1]
        self.kT = kT
        self.J = 1.0
        
    def Glauber(self):
        
        """
        Determines delta E for a given site using the Glauber algorithm.
        Returns delta E, and the coordinates of the trial site (i, j).
        """
        
        # Select a random site
        i = np.random.randint(0,self.Nx)
        j = np.random.randint(0,self.Ny)
        
        # Calculate site delta energy 
        calcE = CalcEnergy(self.spin)
        deltaE = calcE.deltaE(i, j)
        
        return deltaE, i, j
    
    def Kawasaki(self):
        
        """
        Determines delta E using the Kawasaki algorithm.
        Returns delta E, and the coordinates of the 2 trial sites (Ai, Aj) and (Bi, Bj).
        """
        
        # Select 2 random sites
        Ai = np.random.randint(0, self.Nx)
        Aj = np.random.randint(0, self.Ny)
        Bi = np.random.randint(0, self.Nx)
        Bj = np.random.randint(0, self.Ny)
        
        # If sites are the same, select another
        while self.spin[Ai,Aj] == self.spin[Bi,Bj]:
            
            Bi = np.random.randint(0,self.Nx)
            Bj = np.random.randint(0,self.Ny)
            
        # Get both site delta energies
        calcEA = CalcEnergy(self.spin)
        calcEB = CalcEnergy(self.spin)
        deltaEA = calcEA.deltaE(Ai, Aj)
        deltaEB = calcEB.deltaE(Bi, Bj)
        
        # Correction to account for nearest neighbour interactions
        correction = 4*self.J*self.spin[Ai,Aj]*self.spin[Bi,Bj]
        
        # Nearest neighbour threshold
        threshold = (abs(Ai%self.Nx - Bi%self.Nx) + abs(Aj%self.Ny - Bj%self.Ny))
        
        if threshold < 2:
            deltaE = deltaEA + deltaEB - correction     
            
        else:
            deltaE = deltaEA + deltaEB
        
        return deltaE, Ai, Aj, Bi, Bj
    
def CalcError(x, kT, Nxy):
    
    """
    Function to determine the error in an input list using the bootstrap resampling method.
    Takes a list x, temperature (kT) and number of sites (N) as input.
    Returns the standard deviation of the list of determined specific heat values.
    ## For this to be used for susceptibility, change kT**2 to kT ##
    """
    
    Cs = []
    
    for i in range(100): # Use 100 resamples, decrease for smaller input size
        
      y = [random.choice(x) for item in x]  # Randomly draw samples                                                 
      C = (np.mean(np.square(y))-(np.mean(y)**2))/(Nxy*(kT**2)) # Calc. specific heat      
      Cs.append(C)

    return np.std(Cs)

