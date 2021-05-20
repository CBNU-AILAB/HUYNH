import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from SpikeProb.reference import *
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def make_spikes(time_intervals, time_steps, f):
    o = np.zeros((int(time_intervals/time_steps)+1, f.shape[0]))
    for i in range(f.shape[0]):
        if f[i] > 0:
            o[int(f[i]/time_steps), i] = 1
    return o

def spikeplot(time_intervals, time_steps, spiked_data, f1, f2, **kwargs):
    times = np.arange(int(time_intervals/time_steps)+1)

    spikes = []
    spikes.append(make_spikes(time_intervals, time_steps, spiked_data))
    spikes.append(make_spikes(time_intervals, time_steps, f1))
    spikes.append(make_spikes(time_intervals, time_steps, f2))

    # rasterplot(times, spikes[0])

    fig, axes = plt.subplots(3, sharex=True)

    kwargs.setdefault('linestyle', 'None')
    kwargs.setdefault('marker', '|')

    for i in range(len(spikes)):
        if i == 2:
            axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        n_spike, n_neuron = spikes[i].shape
        spiketimes = []
        for j in range(n_neuron):
            spiketimes.append(times[spikes[i][:, j] > 0].ravel())
        spiketimes = np.array(spiketimes)
        indexes = np.zeros(n_neuron, dtype=np.int)
        for t in range(times.shape[0]):
            for k in range(spiketimes.shape[0]):
                if spiketimes[k].shape[0] <= 0:
                    continue
                if indexes[k] < spiketimes[k].shape[0] and times[t] == spiketimes[k][indexes[k]]:
                    axes[i].plot(spiketimes[k][indexes[k]], k+1, 'k', **kwargs)

                    plt.draw()
                    plt.pause(0.002)

                    indexes[k] += 1

    plt.pause(2000)











path = '.'    #test dataset path
def load_iris_dataset():
    iris = load_iris()
    return iris['data'], iris['target']


def get_statistics(inputs):
    min_vals = np.amin(inputs, axis=0)
    max_vals = np.amax(inputs, axis=0)
    return min_vals, max_vals



if __name__ == '__main__':

    #Load IRIS data
    tables, labels = load_iris_dataset()


    #min & max data
    min_vals, max_vals = get_statistics(tables)

    file1 = open(path + 'trained_data', 'rb')
    train_data = pickle.load(file1)

    file2 = open(path + 'test_data', 'rb')
    test_data = pickle.load(file2)

    file3 = open(path + 'trained_label', 'rb')
    train_label = pickle.load(file3)

    file4 = open(path + 'test_label', 'rb')
    test_label = pickle.load(file4)



    n_epochs = 200
    # Declare parameters

    tau = 7  # ms
    num_terminals = 16
    min_delay = 1
    max_delay = 16
    resting = 0.
    hidden_neurons = 10
    output_neurons = 3
    input_neurons = 50
    simulation_time = 25
    threshold = 1
    learning_rate = 0.0075
    inhibitory_neuron_position = 0

    encoding_neurons = 12
    coding_interval = 4
    outside_interval = 5
    timestep=1

    ref={}


    #Weight_loading

    file1 = open(path + 'trained_weight_path', 'rb')
    weights = pickle.load(file1)


    weight = {}
    weight['con1']= weights['con1']
    weight['con2']= weights['con2']



    # Declare connection-delays
    delay ={}
    delay['con1']= np.zeros((input_neurons, num_terminals))
    delay['con2']= np.zeros((hidden_neurons, num_terminals))
    for i in range(num_terminals):
        delay['con1'][:,i] = i+1
        delay['con2'][:,i] = i+1

    # Declare voltage
    voltages={}

    # Declare firing time
    output_firing_times = {}


    #Declare target_firing_time of output layer
    target_firing_times = np.zeros((output_neurons,1))

    #Declare error
    error = np.array(0.)

    # Declare derivatives
    derivaties = {}
    derivaties['output'] = np.zeros(shape=weight['con2'].shape)
    derivaties['hidden'] = np.zeros(shape=weight['con1'].shape)

    # Declare upstream derivatives
    upstream_derivaties = {}
    upstream_derivaties['output'] = np.zeros((output_neurons, 1))
    upstream_derivaties['hidden'] = np.zeros((hidden_neurons, 1))



    # Declare desired and non-desired firing time
    labeled_firing_time = 10
    non_labeled_firing_time = 20

    predicted_label=[]

    draw_spikes = {}

    for sample_index, training_sample in enumerate (test_data):        #loop data

        label_sample = test_label[sample_index]

        '''================================================================
         ============================Encode data============================
         ==================================================================='''

        # Encode training data to firing_times
        firing_times = population_encoder(training_sample, encoding_neurons, min_vals, max_vals, outside_interval, coding_interval, simulation_time)

        output_firing_times['input'] = np.zeros(shape=(firing_times.shape[0] * firing_times.shape[1] + 2))
        output_firing_times['input'][:firing_times.shape[0] * firing_times.shape[1]] = firing_times.flatten()
        output_firing_times['input'][-2:].fill(0)
        # print(output_firing_times['input'])
        print(output_firing_times['input'])
        draw_spikes['input'] = output_firing_times['input']* 0.001


        '''================================================================
          ============================Forward============================
          ==================================================================='''
        spike_response={}
        spike_response['input'] = np.zeros(shape=input_neurons * num_terminals)
        spike_response['hidden'] = np.zeros(shape=hidden_neurons * num_terminals)

        voltages['hidden'] = np.full((hidden_neurons, 1), resting)
        voltages['output'] = np.full((output_neurons, 1), resting)

        output_firing_times['output'] = np.zeros(shape=output_neurons)
        output_firing_times['output'].fill(-1)

        output_firing_times['hidden'] = np.zeros(shape=hidden_neurons)
        output_firing_times['hidden'].fill(-1)

        for t in range(0, simulation_time, timestep):
            spike_response['input'] = compute_spike_responses(t, output_firing_times['input'], spike_response['input'], tau, num_terminals,delay['con1'])
            # print(spike_response['input'].shape)


            voltages['hidden'] = compute_voltage(spike_response['input'], weight['con1'], voltages['hidden'])

            # print(voltages['hidden'].shape)


            output_firing_times['hidden'] = compute_firing_times(t, voltages['hidden'], output_firing_times['hidden'], threshold)
            # print(output_firing_times['hidden'].shape)


            spike_response['hidden'] = compute_spike_responses(t, output_firing_times['hidden'], spike_response['hidden'], tau, num_terminals, delay['con2'])
            spike_response['hidden'][inhibitory_neuron_position:num_terminals] *= -1

            # print(spike_response['hidden'].shape)

            voltages['output'] = compute_voltage(spike_response['hidden'], weight['con2'], voltages['output'])
            # print(voltages['output'].shape)

            output_firing_times['output'] = compute_firing_times(t, voltages['output'], output_firing_times['output'], threshold)

        draw_spikes['hidden']= output_firing_times['hidden']*0.001
        draw_spikes['output'] = output_firing_times['output'] * 0.001

        for i in range (len(draw_spikes['input'])):
                if draw_spikes['input'][i] > 0.025:
                    draw_spikes['input'][i] = -1

        print(draw_spikes['input'])



        spikeplot(time_intervals=0.05, time_steps=0.001, spiked_data= draw_spikes['input'], f1= draw_spikes['hidden'], f2= draw_spikes['output'] )
        breakpoint()


        # Check label


        _label = np.argmin(output_firing_times['output'])

        predicted_label.append(_label)




    print(predicted_label)

    print(test_label)
    count=0
    for p_index in range (len(test_label)):
        if test_label[p_index] == predicted_label[p_index]:
            count+=1

    print(count)
    per = count/75 * 100

    print(per)

