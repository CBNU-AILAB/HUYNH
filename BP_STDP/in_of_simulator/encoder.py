import numpy as np
class Encoder:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
        
class PoissonEncoder(Encoder):
    def __init__(self):
        super(PoissonEncoder, self).__init__()
        self.records = []

    def __call__(self, image, trace=False):
        # image.size: [c, h, w]
        # spike_image.size: [c, h, w]
        c, h, w = image.shape
        spike_image = np.random.uniform(size=(c, h, w))
        for i in range(c):
            for j in range(h):
                for k in range(w):
                    if spike_image[i, j, k] < image[i, j, k]*0.001:
                        spike_image[i, j, k] = 1
                    else:
                        spike_image[i, j, k] = 0
        self.records.append(spike_image)
        if trace:
            return self.records
        return spike_image

class BP_STDP_LabelEncoder(Encoder):
    def __init__(self, num_classes, frequency):
        super(BP_STDP_LabelEncoder, self).__init__()
        self.num_classes = num_classes
        self.sub_interval = 1000/frequency
        self.records = None

    def __call__(self, t, label, trace=False):
        print(label)
        o = np.zeros(self.num_classes)
        if t > 0 and t% self.sub_interval == 0:
            o[label] = 1
        self.records = o
        if trace:
            return self.records
        return o, self.sub_interval
