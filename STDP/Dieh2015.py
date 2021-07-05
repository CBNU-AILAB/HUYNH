import argparse
import torch
import torchvision
import pickle
from n3ml_latest.n3ml.encoder import PoissonEncoder
from n3ml_latest.n3ml.model import Diehl2015

def app(opt):
    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)
    # Define an encoder to generate spike train for an image
    encoder = PoissonEncoder(opt.time_interval)
    # Define a model
    model = Diehl2015()
    # Training Model
    n_img =0
    for epoch in range(opt.num_epochs):
        for images, labels in train_loader:
            n_img+=1
            print(n_img)
            if n_img==1:
                iefile = open('Dieh_Weight_h' + str(n_img), 'wb')
                pickle.dump(model.xe.w, iefile)

                th_file = open('Dieh_threshold_h' + str(n_img), 'wb')
                pickle.dump(model.exc.theta, th_file)


            # Initialize a model
            model.reset()
            # Encode images into spiked_images
            spiked_images = encoder(images)
            spiked_images = spiked_images.view(opt.time_interval, opt.batch_size, -1)

            spiked_images = spiked_images.cuda()        # passing to GPU
            # print(spiked_images.shape)
            # breakpoint()

            # Train a model
            for time in range(opt.time_interval):
                model(spiked_images[time])
                model.update()

            # reset trace
            model.ie.reset_trace()
            # save weights
            if n_img % 5000==0:
                iefile = open('Dieh_Weight_h' + str(n_img), 'wb')
                pickle.dump(model.xe.w, iefile)
                th_file = open('Dieh_threshold_h' + str(n_img), 'wb')
                pickle.dump(model.exc.theta, th_file)
        # save model weights
        iefile1 = open ('Dieh_Weight_epoch'+ str(epoch), 'wb')
        pickle.dump(model.xe.w, iefile1)
        th_file = open('Dieh_threshold_epoch' + str(n_img), 'wb')
        pickle.dump(model.exc.theta, th_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_step', default=1, type=int)         # 1ms
    parser.add_argument('--time_interval', default=350, type=int)   # 350ms
    parser.add_argument('--num_epochs', default=1, type=int)
    app(parser.parse_args())
