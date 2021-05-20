import argparse
from population import *
from InputDataLoader import DataLoader
from encoder import BP_STDP_LabelEncoder, PoissonEncoder
from new_simulator.graph import *


class SNN:
    def __init__(self):
        self.pop1 = PopulationIF(in_neurons=28*28, neurons=500,resting=0., threshold=0.9, tau_ref=0., name='pop1')
        self.pop2 = PopulationIF(in_neurons=500, neurons=150,resting=0., threshold=0.9, tau_ref=0., name='pop2')
        self.pop3 = PopulationIF(in_neurons=150, neurons=10,resting=0., threshold=12.5, tau_ref=0., name='pop3')

    def __call__(self, t, x):
        return self._forward(t, x)

    def _forward(self, t, x):
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].reshape(-1)
        else:
            x = x.reshape(-1)
        o = self.pop1(t, x)
        o = self.pop2(t, o)
        o = self.pop3(t, o)
        return o

def app(opt):
    train_loader = DataLoader(data=opt.data, batch_size=opt.batch_size, shuffle=False, train=True, dataset='MNIST')

    # encoder_img = PoissonEncoder2(max_fr=250.)
    encoder_img = PoissonEncoder()
    encoder_lb = BP_STDP_LabelEncoder(opt.num_classes, frequency=250)

    model = SNN()
    # non-graph version
    loss = graph_version.LossBP_STDP()
    # optimizer = non_graph_version.Optimizer_BP_STDP(W=model.pop2.W, delta_W=loss.delta_W, lr=0.01)
    num_steps = int(opt.simulation_times/opt.time_steps)
    print(num_steps)
    frame_step = -1
    for step in range(1, num_steps):
        frame_step+=1
        t = step * opt.time_steps
        print(t)
        print(frame_step)

        # image.shape: [1, 1, 28, 28]
        # label.shape: [1]
        if step % int(opt.time_frames/opt.time_steps) == 0:
            # time = np.arange(0, 10,1)
            # rasterplot(time, model.pop3.s)
            frame_step=0
            # breakpoint()
            train_loader.next()
            model.pop1.reset_parameters()
            model.pop2.reset_parameters()
            model.pop3.reset_parameters()
            encoder_lb = BP_STDP_LabelEncoder(opt.num_classes, frequency=250)


        image, label = train_loader()
        image = image.numpy()
        label = label.numpy()


        image = np.squeeze(image, 0)  # image.shape: [1, 28, 28]

        spiked_image = encoder_img(image, trace= True)
        spiked_label= encoder_lb(frame_step, label, trace=True)
        # print(spiked_label)

        # simulating network
        model(t, spiked_image)


        l = loss(y=spiked_label, y_hat=model.pop3.s, name='loss')
        print(l)

        loss.backward()
        # optimizer.step()

        # graph = CTX.graph
        for k, v in CTX.graph.items():
            print("k: {}\tv: {}".format(k, v))
        print('--------')

        # breakpoint()


    a = np.array(encoder_lb.records)
    d = np.array(spiked_label)
    s1 = np.array(model.pop1.s)
    s2 = np.array(model.pop2.s)
    s3 = np.array(model.pop3.s)
    time = np.arange(0, num_steps)

    print(time.size)
    print(s1.shape)

    rasterplot(time, s3)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--simulation_times', default=1.0, type=float)  # 600s
    parser.add_argument('--time_steps', default=0.001, type=float)  # 1ms
    parser.add_argument('--time_frames', default=0.01, type=float)  # 10ms

    app(parser.parse_args())
