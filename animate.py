import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
 
# References
# https://gist.github.com/neale/e32b1f16a43bfdc0608f45a504df5a84
# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation
 
# ANIMATION FUNCTION
def func(num, dataSet, line, d2, l2):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, :num])    
    line.set_3d_properties(dataSet[2, :num])    
    l2.set_data(d2[0:2, :num])    
    l2.set_3d_properties(d2[2, :num])    
    return [line, l2]
 
 


def animate(dataSet, dataSet2, numDataPoints, interval=5):
    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = Axes3D(fig)

    # NOTE: Can't pass empty arrays into 3d version of plot()
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0] # For line plot
    line2 = plt.plot(dataSet2[0], dataSet2[1], dataSet2[2], lw=2, c='r')[0] # For line plot

    # AXES PROPERTIES]
    # ax.set_xlim3d([limit0, limit1])
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('Z(t)')
    ax.set_title('Trajectory of electron for E vector along [120]')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line, dataSet2, line2), interval=interval, blit=True)
    #line_ani.save(r'AnimationNew.mp4')


    plt.show()

if __name__ == '__main__':
    # THE DATA POINTS
    t = np.arange(0,20,0.2) # This would be the z-axis ('t' means time here)
    x = np.cos(t)-1
    y = 1/2*(np.cos(2*t)-1)
    dataSet = np.array([x, y, t])
    numDataPoints = len(t)
    
    animate(dataSet, numDataPoints)
