import torch


class BaseLearning:
    def __init__(self):
        pass


class ReSuMe(BaseLearning):
    def __init__(self):
        super().__init__()

class bp_stdp(BaseLearning):
    def __init__(self,
                 lr=0.0005,
                 ):
        super().__init__()

    def __desired_fr_generation(self):
        return

class PostPre(BaseLearning):
    def __init__(self,
                 connection,
                 lr=(1e-4, 1e-2)):
        super().__init__()

        self.connection = connection
        # lr은 tuple 타입으로 크기가 2가 된다. lr[0]는 presynaptic spike에 대한 weight change에서 사용되고
        # lr[1]은 postsynaptic spike에 대한 weight change에서 사용된다.
        self.lr = lr

    def __call__(self):
        # Compute weight changes for presynaptic spikes
        s_pre = self.connection.source.s.unsqueeze(1)
        x_post = self.connection.target.x.unsqueeze(0)
        self.connection.w -= self.lr[0] * torch.transpose(torch.mm(s_pre, x_post), 0, 1)
        # Compute weight changes for postsynaptic spikes
        s_post = self.connection.target.s.unsqueeze(1)
        x_pre = self.connection.source.x.unsqueeze(0)
        self.connection.w += self.lr[1] * torch.mm(s_post, x_pre)

class PostPre2(BaseLearning):
    def __init__(self,
                 connection,
                 tau_pre=20.,
                 tau_post = 20.,
                 pre_lr =0.0001,
                 post_lr = 0.001,
                 dt =1,
                 w_max =1.,
                 offset = 1,
                 ):
        super().__init__()

        self.connection = connection
        self.w_max = torch.tensor(w_max)
        self.tau_pre = torch.tensor(tau_pre)
        self.tau_post = torch.tensor(tau_post)
        self.pre_lr = torch.tensor(pre_lr)
        self.post_lr = torch.tensor(post_lr)
        self.dt = torch.tensor(dt)
        self.x_pre = torch.zeros(1,self.connection.source.neurons)
        self.x_pre_tar = torch.zeros(1,self.connection.source.neurons)
        self.x_post = torch.zeros(self.connection.target.neurons,1)
        self.x_post_tar = torch.zeros(self.connection.target.neurons,1)
        self.offset = offset


    def reset(self):
        self.x_pre = torch.zeros(1, self.connection.source.neurons)
        self.x_pre_tar = torch.zeros(1, self.connection.source.neurons)
        self.x_post = torch.zeros(self.connection.target.neurons, 1)
        self.x_post_tar = torch.zeros(self.connection.target.neurons, 1)

    def __call__(self):

        self.__updateTrace()
        self.__WeightsUpdate()


    def __WeightsUpdate(self):
        self.delta_weight = torch.zeros_like(self.connection.w)

         # reshape spike vectors according to the shape of weights
        s_pre = torch.reshape(self.connection.source.s, [1, len(self.connection.source.s)])
        s_post = torch.reshape(self.connection.target.s, [len(self.connection.target.s), 1])

        # Update weight at pre-spike timing
        self.connection.w += (s_pre>0).float() * (-self.pre_lr) * self.x_post* (self.connection.w**self.offset)

        #Update weight at post-spike timing
        self.connection.w += (s_post>0).float() * (self.post_lr)* (self.x_pre - self.x_pre_tar) * ((self.w_max-self.connection.w)**self.offset)

        pass


    def __updateTrace(self):

        s_pre = torch.reshape(self.connection.source.s, [1, len(self.connection.source.s)])
        s_post = torch.reshape(self.connection.target.s, [len(self.connection.target.s), 1])

        self.x_pre_tar *= torch.exp(-(self.dt/self.tau_pre))
        self.x_pre_tar+= (s_pre>0).float() *1
        self.x_pre = (s_pre>0).float() *self.x_pre_tar

        self.x_post_tar *= torch.exp(-(self.dt/self.tau_post))
        self.x_post_tar += (s_post>0).float() *1
        self.x_post = (s_post>0).float() *self.x_post_tar



