import argparse
import torch
import torchvision
import pickle
from n3ml_latest.n3ml.encoder import PoissonEncoder
from n3ml_latest.n3ml.model import Diehl2015, Diehl2015_Inference


def app(opt):
    # Load MNIST train set
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)

    # Load MNIST test set
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)

    # load trained_weight, trained_threshold and Define the inference model

    #Load weight
    weight_file = open('Dieh_Weight_epoch0', 'rb')
    weight  = pickle.load(weight_file)

    #load threshold
    th_file = open('Dieh_threshold_epoch60000', 'rb')
    frozen_threshold = pickle.load(th_file)

    model = Diehl2015_Inference(trained_weight=weight, trained_threshold= frozen_threshold)


    # Define an encoder to generate spike train for an image
    encoder = PoissonEncoder(opt.time_interval)

    # Inference consists of 2 steps: labelling for exc neurons and prediction

    # Step 1. Labelling exc neurons

    #initialize the assigned_labels
    assigned_labels = -1 * torch.ones(model.exc.neurons)

    #intialize the max_spike_count
    max_spike_count = torch.zeros(model.exc.neurons)

    n_img =0
    for images, labels in train_loader:
        n_img+=1
        # Initialize a model
        model.reset()
        # Encode images into spiked_images
        spiked_images = encoder(images)
        spiked_images = spiked_images.view(opt.time_interval, opt.batch_size, -1)
        spiked_images = spiked_images.cuda()        # passing to GPU

        #feedforward and count number of output spikes
        for time in range(opt.time_interval):
            model(spiked_images[time])

        # assign label for neurons
        assigned_labels = (model.exc.spikecount > max_spike_count).float() * labels

        # update maximum spike count
        max_spike_count = (model.exc.spikecount > max_spike_count).float() *model.exc.spikecount

    # save assigned_labels
    iefile1 = open ('assign_labeled', 'wb')
    pickle.dump(assigned_labels, iefile1)


    '''===================='''

    # Step 2. Prediction

    # generate the spike-frame
    import pandas as pd
    predicted_results = pd.DataFrame(torch.zeros((10, 10)))
    pd.set_option('display.max_columns', predicted_results.shape[0] + 1)
    pd.set_option('display.max_rows', predicted_results.shape[0] + 1)

    # initial average spikecount array
    average_spikecount = torch.zeros(model.exc.neurons)

    n_img =0
    for images, labels in test_loader:
        n_img+=1
        # Initialize a model
        model.reset()
        # Encode images into spiked_images
        spiked_images = encoder(images)
        spiked_images = spiked_images.view(opt.time_interval, opt.batch_size, -1)
        spiked_images = spiked_images.cuda()        # passing to GPU

        #feedforward and count number of output spikes
        for time in range(opt.time_interval):
            model(spiked_images[time])

        # compute average spikecount

        for i in range(10):
            average_spikecount[i] = (torch.sum((assigned_labels == i).float() * model.exc.spikecount))/(torch.sum((assigned_labels==i).float()*1))

        # predict label based on the highest spkecount
        predicted_label = torch.argmax(average_spikecount)
        predicted_results[labels][predicted_label] += 1

   
    print(predicted_results)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_step', default=1, type=int)         # 1ms
    parser.add_argument('--time_interval', default=350, type=int)   # 250ms
    parser.add_argument('--num_epochs', default=30, type=int)
    app(parser.parse_args())
