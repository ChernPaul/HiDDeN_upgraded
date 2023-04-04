import argparse
import re
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop

from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression

from noise_layers.sharp import Sharp
from noise_layers.gauss_blur import Gauss_blur

def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))

# оставляем фиктивным сигма 2 дабы не особо париться
def parse_gauss(gauss_command):
    matches = re.match(r'gauss\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', gauss_command)
    (kernel_h, kernel_w), (sigma1, sigma2) = parse_pair(matches.groups())
    return Gauss_blur((kernel_h, kernel_w), sigma1)


def parse_sharp(sharp_command):
    matches = re.match(r'sharp\((\d+\.*\d*,\d+\.*\d*,\d+\.*\d*)\)', sharp_command)
    params = matches.groups()[0].split(',')
    radius = float(params[0])
    percent = float(params[1])
    threshold = float(params[2])
    return Sharp(radius, percent, threshold)


class NoiseArgParser(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )

    @staticmethod
    def parse_cropout_args(cropout_args):
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command[:len('cropout')] == 'cropout':
                layers.append(parse_cropout(command))
            elif command[:len('crop')] == 'crop':
                layers.append(parse_crop(command))
            elif command[:len('dropout')] == 'dropout':
                layers.append(parse_dropout(command))
            elif command[:len('resize')] == 'resize':
                layers.append(parse_resize(command))
            elif command[:len('jpeg')] == 'jpeg':
                layers.append('JpegPlaceholder')
            elif command[:len('quant')] == 'quant':
                layers.append('QuantizationPlaceholder')
            elif command[:len('gauss')] == 'gauss':
                layers.append(parse_gauss(command))
            elif command[:len('sharp')] == 'sharp':
                layers.append(parse_sharp(command))
            elif command[:len('identity')] == 'identity':
                # We are adding one Identity() layer in Noiser anyway
                pass

            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)
