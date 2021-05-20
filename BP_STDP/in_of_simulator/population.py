from new_simulator.graph import *
import numpy as np

class PopulationIF:
    def __init__(self, in_neurons, neurons,name,  q=1, resting=0, threshold=1, tau_ref=0.0):
        self.in_neurons = in_neurons
        self.neurons = neurons
        self.q = q
        self.u_r = resting
        self.th = threshold
        self.tau_ref = tau_ref
        self.a = [np.zeros(self.in_neurons)]            # time-course of input current
        self.I = [np.zeros(self.neurons)]               # input current
        self.u = [np.zeros(self.neurons)]               # membrane potential
        self.s = [np.zeros(self.neurons)]               # firing time of neurons
        self.ref = np.zeros(self.neurons)               # refactory period of neurons
        self.t_hat = np.zeros(self.neurons)             # rising time of neurons
        self.name = name
        self.W = CTX.create_node(np.zeros((self.neurons, self.in_neurons)), name='{}/{}'.format(self.name, 'W'))
        self.reset_parameters()
        self.reset_variables()

    def reset_parameters(self):
        # self.W[:] = np.random.uniform(low=-0.5, high=0.5, size=(self.neurons, self.in_neurons))
        self.W.value[:] = np.random.normal(0., 1.0, size=(self.neurons, self.in_neurons))

    def reset_variables(self):
        self.a = [np.zeros(self.in_neurons)]  # time-course of input current
        self.I = [np.zeros(self.neurons)]  # input current
        self.u = [np.zeros(self.neurons)]  # membrane potential
        self.s = [np.zeros(self.neurons)]  # firing times of neurons
        self.ref = np.zeros(self.neurons)  # refactory period of neurons
        self.t_hat = np.zeros(self.neurons)  # rising time of neurons


    def __call__(self, t, x):
        # Determine the pre-synaptic psc
        self.psc(t=t,
                 s=x,
                 q=self.q,
                 a=self.a,
                 name='{}/{}'.format(self.name, 'psc'))

        self.current(t=t,
                     W=self.W,
                     a=self.a,
                     I=self.I,
                     name='{}/{}'.format(self.name, 'current'))
        self.integrate(t=t,
                       I=self.I,
                       u=self.u,
                       t_hat=self.t_hat,
                       ref=self.ref,
                       u_r=self.u_r,
                       name='{}/{}'.format(self.name, 'integrate'))
        self.fire(t=t,
                  u=self.u,
                  t_hat=self.t_hat,
                  s=self.s,
                  ref=self.ref,
                  th=self.th,
                  u_r=self.u_r,
                  tau_ref=self.tau_ref,
                  name='{}/{}'.format(self.name, 'fire'))
        self.refractory(t=t,
                        ref=self.ref,
                        s=self.s,
                        name='{}/{}'.format(self.name, 'ref'))
        return self.s

    @CTX.primitive
    def psc(self, t, s, q, a, name):
        """
        a(t-t_) = q*d(t-t_)
        :param t: time
        :param s: presynaptic spikes
        :param q: charge
        :param a: postsynaptic currents (output)
        :return:
        """
        aa = np.zeros(self.in_neurons)
        aa[:] = q * s[-1]               # q* by the last elements of s
        a.append(aa)
        return a
    @CTX.primitive
    def current(self, t, W, a, I, name):
        """
        :param t: time
        :param W: weights
        :param a: PSC
        :param I: input currents (output)
        :return:
        """
        II = np.zeros(self.neurons)
        II[:] = np.dot(W, a[-1])
        I.append(II)
    @CTX.primitive
    def integrate(self, t, I, u, t_hat, ref, u_r, name):
        """
        :param t:
        :param I:
        :param u:
        :param t_hat:
        :param ref:
        :param u_r:
        :return:
        """
        uu = np.zeros(self.neurons)
        for i in range(uu.shape[0]):
            if ref[i] <= 0:
                tt = int(t*1000)
                tt_hat = int(t_hat[i]*1000)
                for s in range(tt-tt_hat+1):
                    uu[i] += I[tt-s][i]
        u.append(uu)
    @CTX.primitive
    def fire(self, t, u, t_hat, s, ref, th, u_r, tau_ref, name):
        """
        :param t:
        :param u:
        :param t_hat:
        :param s:
        :param ref:
        :param th:
        :param u_r:
        :param tau_ref:
        :return:
        """
        ss = np.zeros(self.neurons)
        for i in range(ss.shape[0]):
            if ref[i] <= 0 and u[-1][i] > th:
                ss[i] = 1
                t_hat[i] = t
                u[-1][i] = u_r
                ref[i] += tau_ref
        s.append(ss)

    @CTX.primitive
    def refractory(self, t, ref, s, name):
        """
        :param t:
        :param ref:
        :param s:
        :return:
        """
        for i in range(ref.shape[0]):
            if s[-1][i] == 0 and ref[i] > 0:
                ref[i] -= 0.001
