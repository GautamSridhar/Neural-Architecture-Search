import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np
import time
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

#torch.manual_seed(1)    # reproducible

class Simple2d(nn.Module):

    def forward(self, t, x):
        return torch.mm(x**3, true_A)


class LotkaVolterra(nn.Module):

    def __init__(self, theta):
        super(LotkaVolterra,self).__init__()
        self.theta=theta

    def forward(self, t, x):
        # a = 1.
        # b = 0.1
        # c = 1.5
        # d = 0.75
        a, b, c, d = self.theta[0], self.theta[1], self.theta[2], self.theta[3]

        return torch.tensor([ a*x[0] -   b*x[0]*x[1],
                             -c*x[1] + d*b*x[0]*x[1] ])


class FHN(nn.Module):

    def __init__(self, theta):
        super(FHN, self).__init__()
        self.theta = theta

    def forward(self, t, x):
        # theta1 = 0.2
        # theta2 = 0.2
        # theta3 = 3.0
        theta1, theta2, theta3 = self.theta[0], self.theta[1], self.theta[2]
        return torch.tensor([theta3 * (x[0] - x[0]**3 / 3.0 + x[1]),
                            - 1.0 / theta3 * (x[0] - theta1 + theta2 * x[1])])


class Lorenz63(nn.Module):

    def __init__(self, theta):
        super(Lorenz63, self).__init__()
        self.theta = theta

    def forward(self, t, x):

        theta = self.theta

        return torch.tensor([theta[0] * (x[1] - x[0]),
                             theta[1] * x[0] - x[1] - x[0] * x[2],
                             x[0] * x[1] - theta[2] * x[2]
                            ])


class Lorenz96(nn.Module):

    def __init__(self, theta):
        super(Lorenz96, self).__init__()
        self.theta = theta

    def forward(self, t, x):
        theta = self.theta

        f = []
        for n in range(100):
            f.append((x[(n+1) % 100] - x[(n+98) % 100]) * x[(n+99) % 100]
                     - x[n % 100] + theta)

        return torch.tensor(f)


class ChemicalReactionSimple(nn.Module):

    def __init__(self, theta):
        super(ChemicalReactionSimple, self).__init__()
        self.theta = theta

    def forward(self, t, x):
        theta = self.theta

        firstODE = theta[0] - theta[1]*x[0]*x[1]**2

        secondODE = theta[1]*x[0]*x[1]**2 - theta[2]*x[1]

        return torch.tensor([firstODE, secondODE])


class Chemostat(nn.Module):

    def __init__(self, nSpecies, D, S0, theta):
        super(Chemostat, self).__init__()
        self.nSpecies = nSpecies
        self.D = D
        self.S0 = S0
        self.theta = theta

    def forward(self, t, x):
        theta = self.theta
        assert theta.size == 3*self.nSpecies
        # assert x_in.size == 1 + self.nSpecies

        if type(x) is np.ndarray:
            x = torch.from_numpy(x.T)

        Cetas = torch.from_numpy(theta[:self.nSpecies])
        Vs = torch.from_numpy(theta[self.nSpecies:2*self.nSpecies])
        Ks = torch.from_numpy(theta[-self.nSpecies:])

        xFis = self._MMTerm(torch.ones_like(Vs)*x[-1], Vs, Ks)*x[:self.nSpecies]

        bacDerivs = -self.D*x[:self.nSpecies] \
                  + xFis*Cetas

        subDeriv = self.D*(self.S0 - x[-1]) \
                 - torch.sum(xFis)

        return torch.from_numpy(np.squeeze(np.concatenate([bacDerivs.reshape([1, -1]),

                                          np.asarray(subDeriv).reshape([1, 1])],

                                         axis=1)))

    def _MMTerm(self, S, VMax, KM):

        """
        Describes the uptake of nutrients by the bacterium using a
        Michaelis-Menten term.
        Parameters
        ----------
        S:      scalar or vector of size n
                curent substrate concentration
        VMax:   scalar or vector of size n
                Michaelis-Menten multiplicative constant
        KM:     scalar or vector of size n
                Michaelis-Menten denominator constant
        """
        VMax = VMax
        S = S
        KM = KM
        return VMax*S/(S+KM)


class Clock(nn.Module):

    def __init__(self, theta):

        """
        Initializes the clock model as given by "Methods and Models in
        Mathematical Biology" by Müller and Kuttler, section 5.2.7, using n=3.

        """
        super(Clock,self).__init__()
        self.theta = theta
    

    def forward(self, t, x):

        # TODO: adapt
        """
        Returns the vector of state derivatives of the clock system.
        Parameters
        ----------
        x:      vector of length 7
                states of the clock system.           
        """
        theta = np.asarray(self.theta)
        assert theta.size == 17
        # assert x.size() == 7
        a1 = theta[0]
        a3 = theta[1]
        a4 = theta[2]
        gammaC = theta[3]
        gammaE = theta[4]
        gamma3 = theta[5]
        gamma4 = theta[6]
        d1 = theta[7]
        d2 = theta[8]
        pi1P = theta[9]
        pi1M = theta[10]
        pi2P = theta[11]
        pi2M = theta[12]
        pi3P = theta[13]
        pi3M = theta[14]
        b1 = theta[15]
        k1 = theta[16]

        xc = x[0]
        xe = x[1]
        l = x[2]
        r = x[3]
        y1 = x[4]
        y2 = x[5]
        y3 = x[6]

        xcDeriv = a1*l - (gammaC + d1)*xc + d2*xe - pi1P*r*xc + pi1M*y1
        xeDeriv = d1*xc - d2*xe - gammaE*xe
        lDeriv = a3 - gamma3*l + b1*y3/(1+b1/k1*y3)
        rDeriv = a4 - gamma4*r -pi1P*r*xc + pi1M*y1
        y1Deriv = pi1P*r*xc - pi1M*y1 - 2*pi2P*y1**2 + 2*pi2M*y2 - pi3P*y1*y2 + pi3M*y3
        y2Deriv = pi2P*y1**2 - pi2M*y2 + pi3M*y3 - pi3P*y1*y2
        y3Deriv = pi3P*y1*y2 - pi3M*y3

        

        return torch.tensor([xcDeriv,
                             xeDeriv,
                             lDeriv,
                             rDeriv,
                             y1Deriv,
                             y2Deriv,
                             y3Deriv])

class ProteinTransduction(nn.Module):

    def __init__(self, theta):
        super(ProteinTransduction, self).__init__()
        self.theta = theta

    def forward(self, t, x):
        theta = np.asarray(self.theta)
        f = [- theta[0] * x[0] - theta[1] * x[0] * x[2] + theta[2] * x[3],
             theta[0] * x[0],
             - theta[1] * x[0] * x[2] + theta[2] * x[3] +
             theta[4] * x[4] / (theta[5] + x[4]),
             theta[1] * x[0] * x[2] - theta[2] * x[3] - theta[3] * x[3],
             theta[3] * x[3] - theta[4] * x[4] / (theta[5] + x[4])]
        return torch.tensor(f)


def generate_data(datfunc, true_x0,t,method=None):
    with torch.no_grad():
        true_x = odeint(datfunc, true_x0, t, method=method)
    return true_x

def get_batch(data_size, batch_time, batch_size, true_y, true_der, t):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    batch_der = torch.stack([true_der[s + i] for i in range(batch_time)], dim=0)
    return batch_y0, batch_t, batch_y, batch_der
