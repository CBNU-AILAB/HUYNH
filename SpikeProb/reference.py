import numpy as np
from scipy.stats import norm

def population_encoder(table, num_neurons, min_vals, max_vals, max_firing_time, threshold, simulation_time):
    num_rows = min_vals.shape[0]
    num_cols = num_neurons

    beta = 1.5
    locs = np.zeros(shape=(num_rows, num_cols))
    scales = np.zeros(shape=(num_rows))

    for i in range(locs.shape[0]):
        for j in range(locs.shape[1]):
            locs[i, j] = min_vals[i]+(2*(j+1)-3)*(max_vals[i]-min_vals[i])/(2*(num_neurons-2))
        scales[i] = (max_vals[i]-min_vals[i])/(beta*(num_neurons-2))

    responses = np.zeros(shape=(num_rows, num_cols))
    for i in range(responses.shape[0]):
        for j in range(responses.shape[1]):
            responses[i, j] = norm.pdf(table[i], locs[i, j], scales[i])

    # print(responses)

    max_responses = norm.pdf(0, scale=scales)

    firing_times = transform_firing_times(responses, max_responses, max_firing_time)
    firing_times[firing_times > threshold] = simulation_time+1

    return firing_times

def transform_firing_times(responses, max_response, max_firing_time):
    num_rows, num_cols = responses.shape

    max_response = np.tile(max_response, (num_cols, 1))
    max_response = np.transpose(max_response, (1, 0))

    firing_times = responses * max_firing_time / max_response
    firing_times = firing_times - max_firing_time
    firing_times = firing_times * -1.0
    firing_times = np.around(firing_times)

    return np.array(firing_times)

def compute_spike_responses(t, firing_times, spike_responses, tau, num_terminals, delays):

    _firing_times = np.tile(firing_times, (num_terminals, 1))
    _firing_times = np.transpose(_firing_times, (1, 0))
    print(_firing_times.shape)
    print(delays.shape)
    _t = t - _firing_times - delays
    x = _t / tau
    y = np.exp(1 - _t / tau)
    y = x * y
    y[(_firing_times < 0) | (y < 0)] = 0
    spike_responses[:] = y.flatten()
    return spike_responses

def compute_voltage(spike_responses, weights, voltages):

    spike_responses = np.reshape(spike_responses, (len(spike_responses),1))
    voltages[:] = np.matmul(weights, spike_responses)
    return voltages

def compute_firing_times(t, voltages, firing_times, action_threshold):

    voltages = voltages.flatten()

    firing_times[(firing_times < 0) & (voltages > action_threshold)] = t
    return firing_times

def Update_target_firing_times(label,  labeled_firing_time, non_label_firing_time, target_firing_time):
    # print(desired_firing_time[label])
    target_firing_time.fill(non_label_firing_time)
    target_firing_time[label].fill(labeled_firing_time)

    return target_firing_time

def Update_error(error, firing_times, target_firing_times):
    _y = firing_times[firing_times > 0]
    y = target_firing_times[target_firing_times > 0]
    res = _y - y
    res = res ** 2
    res = np.sum(res)
    res /= 2.0
    error.fill(res)
    return error

def Compute_output_upstream_gradient(firing_times,
                                     pre_firing_times,
                                     delays,
                                     target_firing_times,
                                     weights,
                                     tau,
                                     output_upstream_derivatives):

    num_hidden_neurons, num_terminals = delays.shape
    num_output_neurons = firing_times.shape[0]

    _weights = np.reshape(weights, (num_output_neurons, num_hidden_neurons, num_terminals))


    for j in range(num_output_neurons):
        numerator = target_firing_times[j] - firing_times[j]

        # Compute denominator
        weighted_sum = 0.0

        for i in range(num_hidden_neurons):
            for l in range(num_terminals):
                derivative = Compute_derivative_spike_response(firing_times[j],
                                                               pre_firing_times[i],
                                                               delays[i, l],
                                                               tau)
                if i ==0:
                    derivative *= -1
                update_weights
                weighted_sum += _weights[j, i, l] * derivative

        denominator = weighted_sum

        if denominator != 0.0:
            output_upstream_derivatives[j] = numerator / denominator
        else:
            output_upstream_derivatives[j] = 0.0

    return output_upstream_derivatives


def Compute_derivative_spike_response(firing_time, pre_firing_time, delay, tau):
    t = firing_time - pre_firing_time - delay
    if t <= 0:
        return 0
    elif t>0:
        y = np.exp(1-t/tau)/tau
        y = y - (t*(np.exp(1-t/tau)))/(tau**2)
        return y


def compute_hidden_upstream_derivatives(post_firing_times,
                                        firing_times,
                                        pre_firing_times,
                                        post_delays,
                                        post_weights,
                                        pre_delays,
                                        pre_weights,
                                        hidden_upstream_derivatives,
                                        tau,
                                        output_upstream_derivatives,):


    num_neurons = firing_times.shape[0]
    pre_num_neurons = pre_firing_times.shape[0]
    print('pre_num_neurons is ' + str(pre_num_neurons))
    print('preshape is ' + str(pre_firing_times))
    post_num_neurons = post_firing_times.shape[0]
    num_terminals = pre_delays.shape[1]


    _pre_weights = np.reshape(pre_weights, (num_neurons, pre_num_neurons, num_terminals))
    _post_weights = np.reshape(post_weights, (post_num_neurons, num_neurons, num_terminals))


    for i in range(num_neurons):
        numerator = 0.0

        for j in range(post_num_neurons):
            temp = 0.0

            for k in range(num_terminals):
                derivative = Compute_derivative_spike_response(post_firing_times[j],
                                                               firing_times[i],
                                                               post_delays[i, k],
                                                               tau)
                if i ==0:
                    derivative *= -1

                temp += _post_weights[j, i, k] * derivative

            numerator += output_upstream_derivatives[j] * temp

        denominator = 0.0

        for h in range(pre_num_neurons):
            for l in range(num_terminals):
                derivative = Compute_derivative_spike_response(firing_times[i],
                                                               pre_firing_times[h],
                                                               pre_delays[h, l],
                                                               tau)
                denominator += _pre_weights[i, h, l] * derivative

        if denominator != 0.0:
            hidden_upstream_derivatives[i] = numerator / denominator
        else:
            hidden_upstream_derivatives[i] = 0.0

    return hidden_upstream_derivatives



def compute_output_gradient(firing_times,           # t_a_j
                            pre_firing_times,       #t_a_i
                            delays,                 # dk
                            output_upstream_derivatives,        #gradient
                            tau,
                            output_derivatives,
                            learning_rate):
    num_output_neurons = firing_times.shape[0]
    num_hidden_neurons, num_terminals = delays.shape


    # print("output_derivatives's shape: {}".format(output_derivatives.shape))

    _output_derivatives = output_derivatives.reshape(num_output_neurons, num_hidden_neurons, num_terminals)


    # print("output_derivatives's shape: {}".format(_output_derivatives.shape))

    for j in range(num_output_neurons):
        for i in range(num_hidden_neurons):
            for k in range(num_terminals):
                r = compute_spike_response(firing_times[j], pre_firing_times[i], delays[i, k], tau)
                _output_derivatives[j, i, k] = (-learning_rate) * r * output_upstream_derivatives[j]

    output_derivatives[:] = _output_derivatives.reshape(num_output_neurons, num_hidden_neurons * num_terminals)

    return output_derivatives

def compute_spike_response(t_j, t_i, d_k, tau):
    if t_i < 0:
        return 0
    if t_j < 0:
        return 0
    t = (t_j - t_i - d_k)

    if t <= 0:
        return 0
    # t > 0
    x = t / tau
    r = np.exp((1 - t)/tau)
    r = x * r
    return r

def update_weights(weights, derivatives):
    weights[:] = weights + derivatives
    weights[weights < 0.0] = 0.0
    return weights
