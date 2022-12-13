def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]




class ucr():
    def __init__(self):
        super(ucr, self).__init__()
        # data parameters
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1

        # features
        self.mid_channels = 32
        self.final_out_channels = 128

        self.dropout = 0.35
        self.features_len = 20

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 32


class Epilepsy(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.mid_channels = 32
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 24

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128


class HAR(object):
    def __init__(self):
        # model configs
        self.input_channels = 9
        self.kernel_size = 8
        self.stride = 1
        self.mid_channels = 32
        self.final_out_channels = 128

        self.num_classes = 6
        self.dropout = 0.35
        self.features_len = 18

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128


class EEG(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.mid_channels = 32
        self.final_out_channels = 128
        self.num_classes = 5
        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3
        self.features_len = 127

        # training configs
        self.num_epoch = 40


        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128