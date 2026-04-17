"""Fault modeling using fault propagation analysis."""

import torch
import torch.nn as nn

try:
  from brevitas.nn import QuantConv2d
  from brevitas.quant_tensor import QuantTensor
except:
  QuantConv2d = None
  QuantTensor = None

#from math import ceil, floor
#from random import randint, choice
# .fault_modeling_transient import FaultSA_WS_Conv
from . import fault_modeling_transient as ft
from . import fault_modeling_permanent as fp

FAULT_CLASS_DICT = {
    "FP_OS_T": ft.FaultSA_OS_Conv, "FP_WS_T": ft.FaultSA_WS_Conv, "FP_IS_T": ft.FaultSA_IS_Conv,
    "Q_OS_T": ft.QFaultSA_OS_Conv, "Q_WS_T": ft.QFaultSA_WS_Conv, "Q_IS_T": ft.QFaultSA_IS_Conv,
    "FP_OS_P": fp.FaultSA_OS_Conv, "FP_WS_P": fp.FaultSA_WS_Conv, "FP_IS_P": fp.FaultSA_IS_Conv,
    "Q_OS_P": fp.QFaultSA_OS_Conv, "Q_WS_P": fp.QFaultSA_WS_Conv, "Q_IS_P": fp.QFaultSA_IS_Conv
    }

class Conv2d_FI(nn.Conv2d):
    def __init__(self, in_size, sa_size, fault_class, stuck_at, *args, **kwargs):
        super(Conv2d_FI, self).__init__(*args, **kwargs)
        self.in_size = in_size
        self.sa_size = sa_size
        self.fault = fault_class(self.sa_size, self.in_size, self.in_channels, self.out_channels, 
                                 self.kernel_size[0], self.stride[0], self.padding[0], stuck_at)
        
    def forward(self, input_activation: torch.tensor) -> torch.tensor:
        # Forward pass for the layer
        out_activation = super(Conv2d_FI, self).forward(input_activation)
        
        # Error injection
        out_activation = self.fault.inject_errors(input_activation, self.weight, out_activation)

        return out_activation


class Conv2d_(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_, self).__init__(*args, **kwargs)
        
    def forward(self, input_activation: torch.tensor) -> torch.tensor:
        self.in_size = input_activation.size()[2]
        return super(Conv2d_, self).forward(input_activation)


def compute_channel_view_shape(tensor: torch.Tensor, channel_dim: int):
    broadcast_shape = [1] * len(tensor.size())
    broadcast_shape[channel_dim] = -1
    return tuple(broadcast_shape)

class QuantConv2d_FI(QuantConv2d):
    def __init__(self, in_size, sa_size, fault_class, stuck_at, *args, **kwargs):
        super(QuantConv2d_FI, self).__init__(*args, **kwargs)
        self.in_size = in_size
        self.sa_size = sa_size
        self.fault = fault_class(self.sa_size, self.in_size, self.in_channels, self.out_channels, 
                                 self.kernel_size[0], self.stride[0], self.padding[0], stuck_at)

    def quant_output_scale_impl(self, inp, quant_input_scale, quant_weight_scale):
        output_scale_shape = compute_channel_view_shape(inp, channel_dim=1)
        output_scale = quant_weight_scale.view(output_scale_shape)
        output_scale = output_scale * quant_input_scale.view(output_scale_shape)
        return output_scale

    def forward(self, input_activation):
        inp = self.unpack_input(input_activation)

        quant_input = self.input_quant(inp)
        quant_weight = self.quant_weight(quant_input)
        
        # Quantized input and weight
        qx = torch.round(inp/quant_input.scale)
        qw = torch.round(self.weight/quant_weight.scale)

        compute_output_quant_tensor = isinstance(quant_input, QuantTensor) and isinstance(
            quant_weight, QuantTensor)
        if not (compute_output_quant_tensor or
                self.output_quant.is_quant_enabled) and self.return_quant_tensor:
            raise RuntimeError("QuantLayer is not correctly configured")

        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)
            # Quantized bias
            qb = torch.round(self.bias/quant_bias.scale)
        else:
            quant_bias = None
            qb = None
        
        output_tensor = self.inner_forward_impl(qx, qw, qb)
        output_scale = self.quant_output_scale_impl(inp, quant_input.scale, quant_weight.scale)
        
        # Error injection
        output_tensor = self.fault.inject_errors(qx, qw, output_tensor)
        output_tensor = output_tensor * output_scale

        quant_output = self.output_quant(output_tensor)
        return self.pack_output(quant_output)


class QuantConv2d_(QuantConv2d):
    def __init__(self, *args, **kwargs):
        super(QuantConv2d_, self).__init__(*args, **kwargs)
        
    def forward(self, input_activation):
        self.in_size = input_activation.size()[2]
        return super(QuantConv2d_, self).forward(input_activation)


def substitute_conv_layers(model: nn.Module):
    """Substitute nn.Conv2d layers with the custom class that keeps the input size."""
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            new_layer = Conv2d_(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None),
                padding_mode=layer.padding_mode
            )
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            # Replace the module in the model
            setattr(model, name, new_layer)
        elif list(layer.children()) != []:
            substitute_conv_layers(layer)
        
    return model

def add_fi_layer(model: nn.Module, layer: nn.Module, parent_layer, name, sa_size, ftype, stuck_at):
    """Substitute selected layer with FI variant."""
    
    if isinstance(layer, Conv2d_):
        fi_layer = Conv2d_FI(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=(layer.bias is not None),
            padding_mode=layer.padding_mode,
            sa_size=sa_size,
            in_size=layer.in_size,
            fault_class=FAULT_CLASS_DICT[ftype],
            stuck_at=stuck_at
        )
        fi_layer.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            fi_layer.bias.data = layer.bias.data.clone()
        # Replace the module in the model
        setattr(parent_layer, name, fi_layer)

    return fi_layer


def substitute_qconv_layers(model: nn.Module):
    """Substitute nn.Conv2d/qnn.QuantConv2d layers with the custom class that keeps the input size."""
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            new_layer = Conv2d_(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None),
                padding_mode=layer.padding_mode
            )
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            # Replace the module in the model
            setattr(model, name, new_layer)
        elif isinstance(layer, QuantConv2d):
            new_layer = QuantConv2d_(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None),
                padding_mode=layer.padding_mode,
                weight_quant=(layer.weight_quant is not None),
                bias_quant=(layer.bias_quant is not None),
                input_quant=(layer.input_quant is not None),
                output_quant=(layer.output_quant is not None),
                return_quant_tensor=layer.return_quant_tensor,
                device=(layer.device is not None),
                dtype=(layer.dtype is not None)
            )
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            # Replace the module in the model
            setattr(model, name, new_layer)
        elif list(layer.children()) != []:
            substitute_conv_layers(layer)
        
    return model


def add_qfi_layer(model: nn.Module, layer: nn.Module, parent_layer, name, sa_size, ftype, stuck_at):
    """Substitute selected layer with FI variant."""
    if isinstance(layer, Conv2d_):
        fi_layer = Conv2d_FI(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=(layer.bias is not None),
            padding_mode=layer.padding_mode,
            sa_size=sa_size,
            in_size=layer.in_size,
            fault_class=FAULT_CLASS_DICT[ftype],
            stuck_at=stuck_at
        )
        fi_layer.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            fi_layer.bias.data = layer.bias.data.clone()
        # Replace the module in the model
        setattr(parent_layer, name, fi_layer)
    elif isinstance(layer, QuantConv2d_):
        fi_layer = QuantConv2d_FI(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=(layer.bias is not None),
            padding_mode=layer.padding_mode,
            weight_quant=(layer.weight_quant is not None),
            bias_quant=(layer.bias_quant is not None),
            input_quant=(layer.input_quant is not None),
            output_quant=(layer.output_quant is not None),
            return_quant_tensor=layer.return_quant_tensor,
            device=(layer.device is not None),
            dtype=(layer.dtype is not None),
            sa_size=sa_size,
            in_size=layer.in_size,
            fault_class=FAULT_CLASS_DICT[ftype],
            stuck_at=stuck_at
        )
        fi_layer.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            fi_layer.bias.data = layer.bias.data.clone()
        # Replace the module in the model
        setattr(parent_layer, name, fi_layer)

    return fi_layer