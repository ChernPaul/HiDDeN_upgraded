import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization

# current layers max number = 8
PROBABILITIES = (float(0.2), float(0.25), float(0.2), float(0.15), float(0.2), float(0.15), float(0.2), float(0.25))


def independent_apply(noise_layers,
                      probabilities=PROBABILITIES):
    layers_num = noise_layers.__len__()
    samples = np.random.random_sample(layers_num-1)
    layers_to_apply = [Identity()]
    for i in range(1, layers_num, 1):
        if samples[i-1] <= probabilities[i-1]:
            layers_to_apply.append(noise_layers[i])
    return layers_to_apply


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def remove_noise_layer(self, index):
        self.noise_layers.pop(index)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)

    # def forward(self, encoded_and_cover):
    #     layers_to_apply = independent_apply(self.noise_layers)
    #     for layer in layers_to_apply:
    #         layer(encoded_and_cover)
    #     return encoded_and_cover
