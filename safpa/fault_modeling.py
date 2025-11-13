"""Fault modeling using fault propagation analysis."""

import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor

from math import ceil, floor
from random import randint, choice

class FaultSA_Conv():
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding):
        super(FaultSA_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sa_size = sa_size
        self.in_size = in_size

        self.out_h = int((in_size + 2*self.padding - self.kernel_size) / self.stride + 1)
        self.out_w = self.out_h

        self.dmm_width = self.kernel_size*self.kernel_size*self.in_channels
        self.dmm_height = self.out_h * self.out_w
        kmm_height = self.out_channels
        self.latency = self.dmm_width + self.sa_size - 1 + self.sa_size - 1
        self.h_nsteps = ceil(self.dmm_height/self.sa_size)
        self.k_nsteps = ceil(kmm_height/self.sa_size)
        
        self.errors_func = [self.inject_errors_ireg, self.inject_errors_wreg, self.inject_errors_psum, self.inject_errors_mult]
        self.inject_errors = self.inject_errors_ireg
        #self.generate_fault = self.generate_fault_ireg

    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(0, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size,self.out_channels-self.k_step*self.sa_size-2)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.out_h*self.out_w-self.h_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.kernel_size*self.kernel_size*self.in_channels+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bit = randint(0, 7) if bit is None else bit

        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if bit==7 else self.epsilon

        self.c_out_start = self.k_step * self.sa_size + self.pe_col + 1
        c_out_end = (self.k_step+1) * self.sa_size
        self.c_out_end = c_out_end if (self.out_channels >= c_out_end) else self.out_channels

        w_f = self.timestamp - self.pe_col - self.pe_row
        w_kf = w_f % (self.kernel_size*self.kernel_size)
        self.w_cf = floor(w_f/(self.kernel_size*self.kernel_size))
        self.w_if = floor(w_kf/self.kernel_size)
        self.w_jf = w_kf % self.kernel_size

        sw_f = self.h_step * self.sa_size + self.pe_row
        self.y_if = floor(sw_f/self.out_w)
        self.y_jf = sw_f % self.out_w

        self.x_if = self.y_if * self.stride + self.w_if - self.padding
        self.x_jf = self.y_jf * self.stride + self.w_jf - self.padding
        
        self.skip = True if (sw_f >= self.dmm_height) else False
        self.skip = True if (w_f < 0 or w_f >= self.dmm_width) else self.skip
        self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.kernel_size*self.kernel_size*self.in_channels+self.pe_row+self.pe_col) else self.skip
        
        # Check if x is a padded 0
        self.x_is_pad = True if (not 0 <= self.x_if < self.in_size) or (not 0 <= self.x_jf < self.in_size) else False

    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        sign_coeff = torch.full((batch,), 1, device=device)
        
        if not self.x_is_pad:
            for i in range(batch):
                sign_coeff[i] = (int(x[i,self.w_cf,self.x_if,self.x_jf]) >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, -1, 1)

        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)

        for c in range(self.c_out_start, self.c_out_end):
            y[:,c,self.y_if,self.y_jf] += sign_coeff * epsilon_tensor * weights[c,self.w_cf,self.w_if,self.w_jf]

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(0, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size,self.out_channels-self.k_step*self.sa_size-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.out_h*self.out_w-self.h_step*self.sa_size-2)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.kernel_size*self.kernel_size*self.in_channels+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bit = randint(0, 7) if bit is None else bit

        self.epsilon = -2**self.bit
        self.epsilon = -self.epsilon if bit==7 else self.epsilon

        self.c_out = self.k_step * self.sa_size + self.pe_col

        w_f = self.timestamp - self.pe_col - self.pe_row
        w_kf = w_f % (self.kernel_size*self.kernel_size)
        self.w_cf = floor(w_f/(self.kernel_size*self.kernel_size))
        self.w_if = floor(w_kf/self.kernel_size)
        self.w_jf = w_kf % self.kernel_size

        self.sw_start = self.h_step * self.sa_size + self.pe_row
        sw_end = (self.h_step + 1) * self.sa_size
        self.sw_end = sw_end if (self.out_h*self.out_w > sw_end) else (self.out_h*self.out_w - 1)

        self.y_if_start = floor(self.sw_start/self.out_w)
        self.y_jf_start = self.sw_start % self.out_w

        self.y_if_end = floor(self.sw_end/self.out_w)
        self.y_jf_end = self.sw_end % self.out_w

        self.y_nlines = self.y_if_end - self.y_if_start

        self.x_if_start = self.y_if_start * self.stride + self.w_if - self.padding
        self.x_jf_start = self.y_jf_start * self.stride + self.w_jf - self.padding

        self.x_if_end = self.y_if_end * self.stride + self.w_if - self.padding
        self.x_jf_end = self.y_jf_end * self.stride + self.w_jf - self.padding

        # Check if x is a padded 0
        is_top_pad = True if (self.x_if_start < 0) else False
        is_left_pad = True if (self.x_jf_start < 0) else False
        is_sbottom_pad = True if (self.x_if_start >= self.in_size) else False
        is_ebottom_pad = True if (self.x_if_end >= self.in_size) else False
        is_sright_pad = True if (self.x_jf_start >= self.in_size) else False
        is_eright_pad = True if (self.x_jf_end >= self.in_size) else False
        
        self.y_jf_start = self.padding if is_left_pad else self.y_jf_start
        self.y_jf_start = 0 if is_sright_pad else self.y_jf_start
        self.x_jf_start = 0 if is_left_pad else self.x_jf_start
        self.x_jf_start = (self.kernel_size - self.padding - 1) if is_sright_pad else self.x_jf_start
        
        self.y_if_start = self.padding if is_top_pad else self.y_if_start
        self.y_if_start = self.y_if_start + 1 if is_sright_pad else self.y_if_start
        self.x_if_start = 0 if is_top_pad else self.x_if_start
        self.x_if_start = self.x_if_start + 1 if is_sright_pad else self.x_if_start
        
        self.y_jf_end = self.y_jf_end - self.padding if is_eright_pad else self.y_jf_end
        self.y_jf_end = self.out_w - self.padding - 1 if is_ebottom_pad else self.y_jf_end
        self.x_jf_end = self.x_jf_end - self.padding if is_eright_pad else self.x_jf_end
        self.x_jf_end = self.in_size - 1 if is_ebottom_pad else self.x_jf_end
        
        self.y_if_end = self.y_if_end - self.padding if is_ebottom_pad else self.y_if_end
        self.x_if_end = self.x_if_end - self.padding if is_ebottom_pad else self.x_if_end
        
        self.y_col_last = self.out_w if ((self.out_w - 1) * self.stride + self.w_jf - self.padding) < self.in_size else self.out_w - self.padding
        self.y_nlines = self.y_nlines - 1 if is_top_pad or is_sbottom_pad or is_ebottom_pad else self.y_nlines
        self.y_nlines = self.y_nlines - 1 if is_sright_pad else self.y_nlines
        self.y_nlines = -1 if (self.c_out >= self.out_channels) else self.y_nlines
        self.y_nlines = -1 if (self.timestamp < self.pe_row + self.pe_col) else self.y_nlines
        self.y_nlines = -1 if (self.timestamp >= self.kernel_size*self.kernel_size*self.in_channels+self.pe_row+self.pe_col) else self.y_nlines

    def inject_errors_wreg(self, x, weights, y):
        if (self.y_nlines < 0):
            return y
        
        sign_coeff = (int(weights[self.c_out,self.w_cf,self.w_if,self.w_jf]) >> self.bit) & 1
        sign_coeff = -1 if sign_coeff == 1 else 1
        
        y_col_start = self.y_jf_start
        x_col_start = self.x_jf_start
        if (self.y_nlines != 0):
            y_col_start = self.y_jf_start
            x_col_start = self.x_jf_start
            for row in range(self.y_nlines):
                for i, col in enumerate(range(y_col_start, self.y_col_last)):
                    y[:,self.c_out,self.y_if_start+row,col] += self.epsilon * sign_coeff * x[:,self.w_cf,self.x_if_start+row*self.stride,x_col_start+i*self.stride]
                x_col_start = self.w_jf - self.padding
                y_col_start = self.padding if (x_col_start < 0) else 0
                x_col_start = 0 if (x_col_start < 0) else x_col_start

        for i, col in enumerate(range(y_col_start, self.y_jf_end+1)):
            y[:,self.c_out,self.y_if_end,col] += self.epsilon * sign_coeff * x[:,self.w_cf,self.x_if_end,x_col_start+i*self.stride]

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(0, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size,self.out_channels-self.k_step*self.sa_size-2)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.out_h*self.out_w-self.h_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.kernel_size*self.kernel_size*self.in_channels+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bsize = 32
        self.bit = randint(0, self.bsize-1) if bit is None else bit
        self.sign = choice([1,-1])
        
        self.epsilon = 2**self.bit * self.sign
        
        self.c_out = self.k_step * self.sa_size + self.pe_col
        w_f = self.timestamp - self.pe_col - self.pe_row
        
        sw_f = self.h_step * self.sa_size + self.pe_row
        self.y_if = floor(sw_f/self.out_w)
        self.y_jf = sw_f % self.out_w
        
        self.skip = True if (sw_f >= self.dmm_height) else False
        self.skip = True if (w_f < 0 or w_f >= self.dmm_width) else self.skip
        self.skip = True if (self.c_out >= self.out_channels) else self.skip
        self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.kernel_size*self.kernel_size*self.in_channels+self.pe_row+self.pe_col) else self.skip
        
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        y[:,self.c_out,self.y_if,self.y_jf] += self.epsilon
        return y
    
    def generate_fault_mult(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(0, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size,self.out_channels-self.k_step*self.sa_size-2)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.out_h*self.out_w-self.h_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.kernel_size*self.kernel_size*self.in_channels+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bsize = 16
        self.bit = randint(0, self.bsize-1) if bit is None else bit
        self.sign = choice([1,-1])
        
        self.epsilon = 2**self.bit * self.sign
        
        self.c_out = self.k_step * self.sa_size + self.pe_col
        w_f = self.timestamp - self.pe_col - self.pe_row
        
        sw_f = self.h_step * self.sa_size + self.pe_row
        self.y_if = floor(sw_f/self.out_w)
        self.y_jf = sw_f % self.out_w
        
        self.skip = True if (sw_f >= self.dmm_height) else False
        self.skip = True if (w_f < 0 or w_f >= self.dmm_width) else self.skip
        self.skip = True if (self.c_out >= self.out_channels) else self.skip
        self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.kernel_size*self.kernel_size*self.in_channels+self.pe_row+self.pe_col) else self.skip
        
    def inject_errors_mult(self, x, weights, y):
        if self.skip:
            return y
        
        y[:,self.c_out,self.y_if,self.y_jf] += self.epsilon
        return y
    

class Conv2d_FI(nn.Conv2d):
    def __init__(self, in_size, sa_size, *args, **kwargs):
        super(Conv2d_FI, self).__init__(*args, **kwargs)
        self.in_size = in_size
        self.sa_size = sa_size
        self.fault = FaultSA_Conv(self.sa_size, self.in_size, self.in_channels, self.out_channels, self.kernel_size[0], self.stride[0], self.padding[0])
        
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

class QuantConv2d_FI(qnn.QuantConv2d):
    def __init__(self, in_size, sa_size, *args, **kwargs):
        super(QuantConv2d_FI, self).__init__(*args, **kwargs)
        self.in_size = in_size
        self.sa_size = sa_size
        self.fault = FaultSA_Conv(self.sa_size, self.in_size, self.in_channels, self.out_channels, self.kernel_size[0], self.stride[0], self.padding[0])

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


class QuantConv2d_(qnn.QuantConv2d):
    def __init__(self, *args, **kwargs):
        super(QuantConv2d_, self).__init__(*args, **kwargs)
        
    def forward(self, input_activation):
        self.in_size = input_activation.size()[2]
        return super(QuantConv2d_, self).forward(input_activation)


def substitute_conv_layers(model: nn.Module):
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
            new_layer.bias.data = layer.bias.data.clone()
            # Replace the module in the model
            setattr(model, name, new_layer)
        elif isinstance(layer, qnn.QuantConv2d):
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
            new_layer.bias.data = layer.bias.data.clone()
            # Replace the module in the model
            setattr(model, name, new_layer)
        elif list(layer.children()) != []:
            substitute_conv_layers(layer)
        
    return model


def add_fi_layer(model: nn.Module, layer: nn.Module, parent_layer, name, sa_size):
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
            in_size=layer.in_size
        )
        fi_layer.weight.data = layer.weight.data.clone()
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
            in_size=layer.in_size
        )
        fi_layer.weight.data = layer.weight.data.clone()
        fi_layer.bias.data = layer.bias.data.clone()
        # Replace the module in the model
        setattr(parent_layer, name, fi_layer)

    return fi_layer