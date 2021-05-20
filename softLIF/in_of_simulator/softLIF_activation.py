import torch # import main library
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters

class soft_LIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, currents):
        '''
        :param inputs:
        :param **kwargs:
        :param x: input tensor
        :return: softLIF firing-rate
        '''
        # define parameters
        ref = Parameter(torch.tensor(0.001))
        v_th = Parameter(torch.tensor(1.0))
        tau = Parameter(torch.tensor(0.05))
        gamma = Parameter(torch.tensor(0.02))

        # save for re-use in backward
        ctx.save_for_backward(currents)

        # Calculate the rate
        j = gamma * torch.log1p(torch.exp((currents - v_th) / gamma))
        output = torch.zeros_like(currents)
        output[j > 0] = 1. / (ref + tau * torch.log1p(1. / j[j > 0]))
        return output


    @staticmethod
    def backward(ctx, grad_output):
        # stored information from forward step
        currents, = ctx.saved_tensors

        # define parameter
        ref = Parameter(torch.tensor(0.001))
        v_th = Parameter(torch.tensor(1.0))
        tau = Parameter(torch.tensor(0.05))
        gamma = Parameter(torch.tensor(0.02))
        gain = Parameter(torch.tensor(1.))

        grad_input = grad_output.clone()

        y = currents - torch.tensor(1.)
        y = y / gamma
        j = torch.tensor(y)
        j[y<34.] = gamma*torch.log1p(torch.exp(y[y<34.]))
        yy = y[j>0]
        jj = j[j>0]
        vv = torch.tensor(1.)/(ref+ tau*torch.log1p(v_th/ jj))

        grad_input[j>0]= (gain * tau * vv* vv ) / (torch.tensor(1.) * jj * (jj+1) * (1+torch.exp(-yy)))

        # print(out_gradient)

        return grad_input
