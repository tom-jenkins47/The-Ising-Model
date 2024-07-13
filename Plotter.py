import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""
Program to plot relevant graphs from output data files.
"""


def plotGlauber():
    
    data = np.loadtxt('observables_glauber.txt', skiprows=1, delimiter= ',')
    
    plt.plot(data[:, 0], data[:, 1], marker='s', color='k')
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.title('Energy vs. Temperature (Glauber)')
    plt.show()
    
    plt.plot(data[:, 0], data[:, 2], marker='s', color='k')
    plt.xlabel('Temperature')
    plt.ylabel('Absolute Magnetisation')
    plt.title('Absolute Magnetisation vs. Temperature (Glauber)')
    plt.show()
    
    plt.plot(data[:, 0], data[:, 3], marker='s', color='k')
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility')
    plt.title('Susceptibility vs. Temperature (Glauber)')
    plt.show()
    
    plt.errorbar(data[:, 0], data[:, 4], yerr=data[:, 5], marker='s', color='k')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.title('Specific Heat vs. Temperature (Glauber)')
    plt.show()
    
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]
    
    
def plotKawasaki():
    
    data = np.loadtxt('observables_kawasaki.txt', skiprows=1, delimiter=',')
    
    plt.plot(data[:, 0], data[:, 1], marker='s', color='k')
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.title('Energy vs. Temperature (Kawasaki)')
    plt.show()
    
    plt.errorbar(data[:, 0], data[:, 2], yerr=data[:, 3], marker='s', color='k')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.title('Specific Heat vs. Temperature (Kawasaki)')
    plt.show()
    
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

plotGlauber()
plotKawasaki()

    

    

    
    
 
    
    

