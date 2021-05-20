from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from SpikeProb.reference import *
import pickle

path = '/.'   # set the data_path (training data and testing data
def load_iris_dataset():
    iris = load_iris()
    return iris['data'], iris['target']


def get_statistics(inputs):
    min_vals = np.amin(inputs, axis=0)
    max_vals = np.amax(inputs, axis=0)
    return min_vals, max_vals
 
if __name__ == '__main__':

#     #Load IRIS data
#     tables, labels = load_iris_dataset()

#     #cross validation
#     train_data, test_data, train_label, test_label = train_test_split(
#         tables, labels, test_size=0.5)
  
  
    # determine min and max data
    min_vals, max_vals = get_statistics(tables)



    indexes = np.arange(tables.shape[0])

    #define the training epochs
    n_epochs = 400
    
    # Declare parameters

    tau = 7  # ms
    num_terminals = 16
    min_delay = 1
    max_delay = 16
    resting = 0.
    hidden_neurons = 10
    output_neurons = 3
    input_neurons = 50
    simulation_time = 24
    threshold = 1
    learning_rate = 0.0075
    inhibitory_neuron_position = 0

    encoding_neurons = 12
    coding_interval = 4
    outside_interval = 5
    timestep=1



    #Weight_initialization
    weight = {}
    # weight['con1']= np.full((hidden_neurons,num_terminals *input_neurons), 0.009)
    # weight['con2']= np.full((output_neurons,num_terminals*hidden_neurons), 0.03)

    weight['con1'] = np.random.uniform(0.24, 0.25, (hidden_neurons, num_terminals * input_neurons))
    weight['con2'] = np.random.uniform(0.155, 0.16, (output_neurons, num_terminals * hidden_neurons))



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
    print(derivaties['output'].shape)
    derivaties['hidden'] = np.zeros(shape=weight['con1'].shape)
    print(derivaties['hidden'].shape)

    # Declare upstream derivatives
    upstream_derivaties = {}
    upstream_derivaties['output'] = np.zeros((output_neurons, 1))
    upstream_derivaties['hidden'] = np.zeros((hidden_neurons, 1))



    # Declare desired and non-desired firing time
    labeled_firing_time = 5
    non_labeled_firing_time = 20


    '''==========================================================================
       ================================Training =================================
       =========================================================================='''


    for epoch in range (1, n_epochs+1):                 #loop epoch

        print('==============Start epoch ' + str(epoch)+ ' ====================')

        for sample_index, training_sample in enumerate (train_data):        #loop data

            label_sample = train_label[sample_index]

            '''================================================================
             ============================Encode data============================
             ==================================================================='''

            # Encode training data to firing_times
            firing_times = population_encoder(training_sample, encoding_neurons, min_vals, max_vals, outside_interval, coding_interval, simulation_time)

            output_firing_times['input'] = np.zeros(shape=(firing_times.shape[0] * firing_times.shape[1] + 2))
            output_firing_times['input'][:firing_times.shape[0] * firing_times.shape[1]] = firing_times.flatten()
            output_firing_times['input'][-2:].fill(0)

            
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
                print(t)

                # print(output_firing_times['input'].shape)
                spike_response['input'] = compute_spike_responses(t, output_firing_times['input'], spike_response['input'], tau, num_terminals,delay['con1'])
                # print(output_firing_times['input'].shape)
                # print(spike_response['input'].shape)


                voltages['hidden'] = compute_voltage(spike_response['input'], weight['con1'], voltages['hidden'])

                print(voltages['hidden'])


                output_firing_times['hidden'] = compute_firing_times(t, voltages['hidden'], output_firing_times['hidden'], threshold)
                print(output_firing_times['hidden'])

                spike_response['hidden'] = compute_spike_responses(t, output_firing_times['hidden'], spike_response['hidden'], tau, num_terminals, delay['con2'])
                spike_response['hidden'][inhibitory_neuron_position:num_terminals] *= -1

                # print(spike_response['hidden'].shape)

                voltages['output'] = compute_voltage(spike_response['hidden'], weight['con2'], voltages['output'])
                print(voltages['output'])

                output_firing_times['output'] = compute_firing_times(t, voltages['output'], output_firing_times['output'], threshold)
                print(output_firing_times['output'])
                print('==============')

                
            '''=================================learning============================================'''

            target_firing_times = Update_target_firing_times(label_sample, labeled_firing_time, non_labeled_firing_time, target_firing_times)
            # print(target_firing_times)


            # Determine error
            error = Update_error(error, output_firing_times['output'], target_firing_times)
            # print(output_firing_times['output'])
            # print(error)

            # Compute upstream-derivative of output-layer
            upstream_derivaties['output'] = Compute_output_upstream_gradient(output_firing_times['output'], output_firing_times['hidden'], delay['con2'], target_firing_times, weight['con2'], tau,
                                                                             upstream_derivaties['output'])

            upstream_derivaties['hidden'] = compute_hidden_upstream_derivatives(output_firing_times['output'], output_firing_times['hidden'], output_firing_times['input'],
                                                                               delay['con2'], weight['con2'], delay['con1'], weight['con1'],
                                                                                upstream_derivaties['hidden'], tau, upstream_derivaties['output'])

            # print(upstream_derivaties['hidden'])

            # Compute output gradient
            derivaties['output'] = compute_output_gradient(output_firing_times['output'], output_firing_times['hidden'], delay['con2'], upstream_derivaties['output'], tau, derivaties['output'],
                                                           learning_rate)

            derivaties['hidden'] = compute_output_gradient(output_firing_times['hidden'], output_firing_times['input'], delay['con1'], upstream_derivaties['hidden'], tau, derivaties['hidden'],
                                                           learning_rate)

            # Update output-weight
            weight['con2'] = update_weights(weight['con2'], derivaties['output'])

            # Update hidden-weight
            weight['con1'] = update_weights(weight['con1'], derivaties['hidden'])
            # breakpoint()


        '''========================Finish epoch==========================='''


        print('finish epoch ' + str(epoch))

        file = open(path + 'weights_epoch' + str(epoch), 'wb')
        pickle.dump(weight, file)
