import math

import torch.nn as nn

from src.classes.deep_learning.modules.common import Conv


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def parse_model(layout: dict, logger):
    """
    This function constructs a PyTorch model from a model dictionary. It supports a variety of
    layer types and configurations, including backbones and heads, and allows for custom
    network architectures.

    :param layout: dictionary that contains the model configuration
    :param logger: ogger object that is used to print debug information
    :return:  nn.Sequential model, sorted list of layers
    """
    logger.info(f'{"idx":>1}{"from":>18}{"n":>3}{"params":>10}  {"module":<40}{"arguments":<30}')
    input_channels: list[int] = [layout['class_number']]

    layers, save, output_channel = [], [], input_channels[-1]  # layers, savelist, ch out
    for idx, (fr, num_repetitions, module, params) in enumerate(layout['backbone'] + layout['head']):
        # Evaluate the module if read as string, else pass directly (e.g. "Conv" -> Conv module)
        module = eval(module) if isinstance(module, str) else module
        # Evaluate parameters if read as strings, else pass them directly (e.g. "1" -> 1)
        params = [eval(param) if isinstance(param, str) else param for param in params]

        if module in [nn.Conv2d, Conv, nn.MaxPool2d, nn.Linear]:
            in_channel, output_channel = input_channels[fr], params[0]
        elif module is nn.BatchNorm2d:
            params = [input_channels[fr]]
        elif module in [nn.ReLU, nn.Flatten]:
            params = []
        else:
            output_channel = input_channels[fr]
        # applied num times, or just once
        module_applied = (nn.Sequential(*[module(*params) for _ in range(num_repetitions)])
                          if num_repetitions > 1 else module(*params))
        module_type = str(module)[8:-2].replace('__main__.', '')  # module type
        parameter_number = sum([x.numel() for x in module_applied.parameters()])  # number params
        # attach index, 'from' index, type, number params
        module_applied.layer_index, module_applied.from_layer, \
            module_applied.type, module_applied.parameter_number = idx, fr, module_type, parameter_number
        logger.info(f'{idx:>3}{fr:>18}{num_repetitions:>3}{parameter_number:10.0f}  {module_type:<40}{params}')  # print
        layers.append(module_applied)
        if idx == 0:
            input_channels = []
        input_channels.append(output_channel)
    return nn.Sequential(*layers)
