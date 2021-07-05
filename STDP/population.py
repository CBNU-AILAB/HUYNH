import torch
import torch.nn as nn


class BasePopulation(nn.Module):
    def __init__(self):
        super().__init__()

    def get(self, name):        
        return self.__dict__['_buffers'][name].numpy()

class IFPopulation(BasePopulation):
    def __init__(self, neurons, dt=1., v_th=0.9, rest=0., tau_ref=0., count_spike=False):
        super().__init__()
        self.neurons = neurons

        self.register_buffer('dt', torch.tensor(dt))        #timestep
        self.register_buffer('v', torch.zeros(neurons))    # potential
        self.register_buffer('v_th', torch.tensor(v_th))    # hard threshold
        self.register_buffer('rest', torch.tensor(rest))    # resting potential
        self.register_buffer('refrac', torch.zeros(neurons))    # list of refactory periods
        self.register_buffer('tau_ref', torch.tensor(tau_ref))  # refactory period
        self.register_buffer('s', torch.zeros(neurons))         # spike list of neurons in layer
        self.v.fill_(self.rest)
        if count_spike == True:
            self.count_spike = count_spike
            self.register_buffer('spikecount', torch.zeros(neurons))  # spike list of neurons in layer

    def reset(self):
        self.v[:] = self.rest

    def forward(self, x):
        self.v += (self.refrac <= 0).float() * x

        self.refrac -= self.dt                                          #reduce the refactory

        self.s[:] = self.v >= self.v_th                                # update spike if potential larger threshold

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)        # set refactory period if a neuron spiked
        self.v.masked_fill_(self.s.bool(), self.rest)               # reset potential if larger than threshold

        if self.count_spike:
            self.spikecount += self.s

        return self.s, self.v

class HomoeostasisLIFPopulation(BasePopulation):
    def __init__(self, neurons, dt=1., tau_rc=100., v_th=-52., theta_plus=0.05, tau_theta= 1e7, rest=-65., tau_ref=5., frozen_threshold= False, count_spike = True):
        super().__init__()

        self.neurons = neurons
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        self.register_buffer('threshold', torch.tensor(tau_ref))
        self.register_buffer('theta_plus', torch.tensor(theta_plus))
        self.register_buffer('theta', torch.zeros(neurons))  # list of theta
        self.register_buffer('tau_theta', torch.tensor(tau_theta))  # tau of theta



        self.v.fill_(self.rest)
        self.threshold.fill_(self.v_th)

        self.frozen_th = frozen_threshold
        self.count_spike = count_spike

        if self.count_spike ==True:
            self.register_buffer('spikecount', torch.zeros(neurons))

    def reset(self):
        self.v[:] = self.rest
        if self.count_spike ==True:
            self.spikecount.fill_(0.)

    def forward(self, x):
        #Decay potential
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest
        # Update potential to the neurons potential
        self.v += (self.refrac <= 0).float() * x
        # Update refactory period
        self.refrac -= self.dt
        #output spikes update
        self.s[:] = self.v >= self.v_th + self.theta

        #update spikecount
        if self.count_spike == True:
            self.spikecount += (self.s>0).float() * 1

        if self.frozen_th == False:
            #update new theta if neurons spike
            self.theta += (self.s>0).float() * self.theta_plus
            #decay theta if neurons no spikes
            self.theta += (self.s < 1).float() * torch.exp(-(self.dt/self.tau_theta))

        # reset refactory period if neurons spike
        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        #reset potential
        self.v.masked_fill_(self.s.bool(), self.rest)
        return self.s, self.v


class LIFPopulation2(BasePopulation):
    def __init__(self, neurons, dt=1., tau_rc=100., v_th=-52., rest=-65., tau_ref=5.):
        super().__init__()

        self.neurons = neurons
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        self.register_buffer('spikecount', torch.zeros(neurons))

        self.v.fill_(self.rest)

    def reset(self):
        self.v[:] = self.rest
        self.spikecount.fill_(0)

    def forward(self, x):
        #Decay potential
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest
        # Update potential to the neurons potential
        self.v += (self.refrac <= 0).float() * x
        # Update refactory period
        self.refrac -= self.dt
        #output spikes
        self.s[:] = self.v >= self.v_th
        #update spikecount
        self.spikecount += (self.s>0).float() * 1
        # reset refactory period if neurons spike
        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        #reset potential
        self.v.masked_fill_(self.s.bool(), self.rest)
        return self.s, self.v

