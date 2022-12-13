import torch
from torch import nn
from copy import deepcopy
from .helpers import SEBasicBlock
from .helpers import MultiHeadedAttention, TCE, PositionwiseFeedForward, EncoderLayer


def get_network_class(network_name):
    """Return the algorithm class with the given name."""
    if network_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(network_name))
    return globals()[network_name]


class classifier(nn.Module):
    def __init__(self, configs, hparams):
        super(classifier, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        # print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        predictions = self.logits(x_flat)
        return predictions


##########################################################################################

class cnn1d_fe(nn.Module):
    def __init__(self, configs):
        super(cnn1d_fe, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.aap(x)
        # print(x.shape)
        return x


class cnn1d_temporal(nn.Module):
    def __init__(self, hparams):
        super(cnn1d_temporal, self).__init__()

    def forward(self, x):
        return x

