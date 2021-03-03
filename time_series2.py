import argparse
from rossler_map import RosslerMap
import numpy as np
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = 10000 // self.delta_t

        self.rosler_nn = torch.load('model.checkpoint')
        self.initial_condition = np.array(value.init)

    def full_traj(self): 
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 
        initial_condition = self.initial_condition
        y, traj = [], []
        self.rosler_nn.eval()
        with torch.no_grad():
            current = torch.from_numpy(self.initial_condition)[None, :]
            for _ in tqdm.tqdm(range(1000000)):
                if len(y) > 999999:
                    break
                y.append(current[0, 1])
                traj.append(current[0])
                current = self.rosler_nn(current)

        y = np.array(y)
        traj = np.array(torch.stack(traj))

        #if your delta_t is different to 1e-2 then interpolate y
        #in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        assert y.shape == (1000000,)
        return y, traj

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
        assert y.shape == (1000000,)
          
        np.save('y_traj.npy',y)
        
    
if __name__ == '__main__':
    delta_t = 1e-2
    ROSSLER = Rossler_model(delta_t)
    y, traj = ROSSLER.full_traj()
    ROSSLER.save_traj(y)
    np.save('our_traj.npy', traj)

    Niter = 1000000
    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array([-5.75, -1.6,  0.02])
    realtraj, t = ROSSLER_MAP.full_traj(Niter, INIT)
    np.save('real_traj.npy', realtraj)

