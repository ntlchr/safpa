# -*- coding: utf-8 -*-
"""Fault modeling using fault propagation analysis for transient faults."""

import torch
import torch.nn as nn

from math import ceil, floor
from random import randint
import struct


def floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

def floatToBitsTensor(f, batch):
    res = torch.zeros((batch,), dtype=int, device=f.device)
    for i in range(batch):
        s = struct.pack('>f', f[i])
        res[i] = struct.unpack('>l', s)[0]
    return res

class FaultSA_WS_Conv():
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at=None):
        super(FaultSA_WS_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sa_size = sa_size
        self.in_size = in_size
        self.bsize_regs = 32
        self.bsize_psum = 32

        self.out_h = int((in_size + 2*self.padding - self.kernel_size) / self.stride + 1)
        self.out_w = self.out_h

        self.dmm_width = self.kernel_size*self.kernel_size*self.in_channels
        self.dmm_height = self.out_h * self.out_w
        kmm_height = self.out_channels
        self.latency = self.dmm_height + self.sa_size - 1 + self.sa_size - 1    # WS
        self.h_nsteps = ceil(self.dmm_width/self.sa_size)   # WS
        self.k_nsteps = ceil(kmm_height/self.sa_size)
        
        self.errors_func = [self.inject_errors_ireg, self.inject_errors_wreg, self.inject_errors_psum]
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
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-self.k_step*self.sa_size-2)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-self.h_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.dmm_height+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)

        self.c_out_start = self.k_step * self.sa_size + self.pe_col + 1    # WS
        c_out_end = (self.k_step+1) * self.sa_size                         # WS
        self.c_out_end = c_out_end if (self.out_channels >= c_out_end) else self.out_channels

        w_f = self.h_step*self.sa_size + self.pe_row                   # WS
        w_kf = w_f % (self.kernel_size*self.kernel_size)
        self.w_cf = floor(w_f/(self.kernel_size*self.kernel_size))
        self.w_if = floor(w_kf/self.kernel_size)
        self.w_jf = w_kf % self.kernel_size

        sw_f = self.timestamp - self.pe_col - self.pe_row               # WS
        self.y_if = floor(sw_f/self.out_w)
        self.y_jf = sw_f % self.out_w
        
        self.x_if = self.y_if * self.stride + self.w_if - self.padding
        self.x_jf = self.y_jf * self.stride + self.w_jf - self.padding
        
        self.skip = True if (sw_f < 0 or sw_f >= self.dmm_height) else False    # WS
        self.skip = True if (w_f >= self.dmm_width) else self.skip              # WS
        self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.dmm_height+self.pe_row+self.pe_col) else self.skip
        
        # Check if x is a padded 0
        self.x_is_pad = True if (not 0 <= self.x_if < self.in_size) or (not 0 <= self.x_jf < self.in_size) else False

    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device

        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        if self.x_is_pad:
            if self.bit_type == 0:
                epsilon_tensor = torch.full((batch,), 0, device=device)
            elif self.bit_type == 1:
                epsilon_tensor = torch.pow(torch.full((batch,), 2.0, device=device), epsilon_tensor-127)
            else:
                epsilon_tensor *= torch.pow(torch.full((batch,), 2.0, device=device), -126)
        else:
            x_int = floatToBitsTensor(x[:,self.w_cf,self.x_if,self.x_jf], batch)
            sign_coeff = (x_int >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, -1, 1)
            if self.bit_type == 0:
                epsilon_tensor *= sign_coeff * x[:,self.w_cf,self.x_if,self.x_jf]
            elif self.bit_type == 1:
                epsilon_tensor = ((torch.pow(torch.full((batch,), 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * x[:,self.w_cf,self.x_if,self.x_jf]).to(torch.float)
            else:
                exponent = (x_int >> 23) & 0xFF
                epsilon_tensor *= (torch.pow(torch.full((batch,), 2.0, device=device), exponent - 127)) * sign_coeff

        for c in range(self.c_out_start, self.c_out_end):
            y[:,c,self.y_if,self.y_jf] += epsilon_tensor * weights[c,self.w_cf,self.w_if,self.w_jf]

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-self.k_step*self.sa_size-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-self.h_step*self.sa_size-2)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.dmm_height+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)

        self.c_out = self.k_step * self.sa_size + self.pe_col           # WS

        w_f = self.h_step*self.sa_size + self.pe_row                    # WS
        w_kf = w_f % (self.kernel_size*self.kernel_size)
        self.w_cf = floor(w_f/(self.kernel_size*self.kernel_size))
        self.w_if = floor(w_kf/self.kernel_size)
        self.w_jf = w_kf % self.kernel_size

        self.sw_start = self.timestamp - self.pe_col - self.pe_row      # WS
        self.sw_end = self.out_h*self.out_w - 1                         # WS

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
        self.y_nlines = self.y_nlines - self.padding if is_top_pad or is_sbottom_pad or is_ebottom_pad else self.y_nlines
        self.y_nlines = self.y_nlines - 1 if is_sright_pad else self.y_nlines
        self.y_nlines = -1 if (self.c_out >= self.out_channels) else self.y_nlines
        self.y_nlines = -1 if (self.timestamp >= self.dmm_height+self.pe_row+self.pe_col) else self.y_nlines
        self.y_nlines = -1 if (self.w_cf >= self.in_channels) else self.y_nlines    # WS

    def inject_errors_wreg(self, x, weights, y):
        if (self.y_nlines < 0):
            return y

        w_int = floatToBits(weights[self.c_out,self.w_cf,self.w_if,self.w_jf])
        sign_coeff = (w_int >> self.bit) & 1
        sign_coeff = -1 if sign_coeff == 1 else 1

        if self.bit_type == 0:
            epsilon_error = self.epsilon * sign_coeff * weights[self.c_out,self.w_cf,self.w_if,self.w_jf]
        elif self.bit_type == 1:
            epsilon_error = ((2.0**(sign_coeff*self.epsilon) - 1) * weights[self.c_out,self.w_cf,self.w_if,self.w_jf].to(torch.double)).to(torch.float)
        else:
            exponent = (w_int >> 23) & 0xFF
            epsilon_error = self.epsilon * 2**(exponent - 127) * sign_coeff

        y_col_start = self.y_jf_start
        x_col_start = self.x_jf_start
        if (self.y_nlines != 0):
            y_col_start = self.y_jf_start
            x_col_start = self.x_jf_start
            for row in range(self.y_nlines):
                for i, col in enumerate(range(y_col_start, self.y_col_last)):
                    y[:,self.c_out,self.y_if_start+row,col] += epsilon_error * x[:,self.w_cf,self.x_if_start+row*self.stride,x_col_start+i*self.stride]
                x_col_start = self.w_jf - self.padding
                y_col_start = self.padding if (x_col_start < 0) else 0
                x_col_start = 0 if (x_col_start < 0) else x_col_start

        for i, col in enumerate(range(y_col_start, self.y_jf_end+1)):
            y[:,self.c_out,self.y_if_end,col] += epsilon_error * x[:,self.w_cf,self.x_if_end,x_col_start+i*self.stride]

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(0, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-self.k_step*self.sa_size-2)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-self.h_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.dmm_height+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bit = randint(0, self.bsize_psum-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        self.c_out = self.k_step * self.sa_size + self.pe_col
        self.w_f = self.h_step * self.sa_size + self.pe_row          # WS
        
        self.sw_f = self.timestamp - self.pe_col - self.pe_row       # WS
        self.y_if = floor(self.sw_f/self.out_w)
        self.y_jf = self.sw_f % self.out_w
        
        self.skip = True if (self.sw_f < 0 or self.sw_f >= self.dmm_height) else False
        self.skip = True if (self.w_f >= self.dmm_width) else self.skip
        self.skip = True if (self.c_out >= self.out_channels) else self.skip
        self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.dmm_height+self.pe_row+self.pe_col) else self.skip
        
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
        psum = x_unf[:,self.w_f-self.pe_row:self.w_f+1,self.sw_f].matmul(weights.view(weights.size(0), -1)[self.c_out,self.w_f-self.pe_row:self.w_f+1])
        
        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        #p_int = floatToBitsTensor(psum, batch)
        p_int = psum.view(torch.int32)
        sign_coeff = (p_int >> self.bit) & 1
        sign_coeff = torch.where(sign_coeff == 1, -1, 1)
        if self.bit_type == 0:
            epsilon_tensor *= sign_coeff * psum
        elif self.bit_type == 1:
            epsilon_tensor = ((torch.pow(torch.full((batch,), 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * psum).to(torch.float)
        else:
            exponent = (p_int >> 23) & 0xFF
            epsilon_tensor *= (torch.pow(torch.full((batch,), 2.0, device=device), exponent - 127)) * sign_coeff
        
        y[:,self.c_out,self.y_if,self.y_jf] += epsilon_tensor
        return y

class QFaultSA_WS_Conv(FaultSA_WS_Conv):
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at=None):
        super(QFaultSA_WS_Conv, self).__init__(sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding)
        self.bsize_regs = 8
        self.bsize_psum = 32
    
    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_ireg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device

        sign_coeff = torch.full((batch,), 1, device=device)
        if not self.x_is_pad:
            sign_coeff = (x[:,self.w_cf,self.x_if,self.x_jf].to(torch.int32) >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, -1, 1)

        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        for c in range(self.c_out_start, self.c_out_end):
            y[:,c,self.y_if,self.y_jf] += sign_coeff * epsilon_tensor * weights[c,self.w_cf,self.w_if,self.w_jf]

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_wreg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_wreg(self, x, weights, y):
        if (self.y_nlines < 0):
            return y

        sign_coeff = (int(weights[self.c_out,self.w_cf,self.w_if,self.w_jf]) >> self.bit) & 1
        sign_coeff = -1 if sign_coeff == 1 else 1
        
        epsilon_error = self.epsilon * sign_coeff

        y_col_start = self.y_jf_start
        x_col_start = self.x_jf_start
        if (self.y_nlines != 0):
            y_col_start = self.y_jf_start
            x_col_start = self.x_jf_start
            for row in range(self.y_nlines):
                for i, col in enumerate(range(y_col_start, self.y_col_last)):
                    y[:,self.c_out,self.y_if_start+row,col] += epsilon_error * x[:,self.w_cf,self.x_if_start+row*self.stride,x_col_start+i*self.stride]
                x_col_start = self.w_jf - self.padding
                y_col_start = self.padding if (x_col_start < 0) else 0
                x_col_start = 0 if (x_col_start < 0) else x_col_start

        for i, col in enumerate(range(y_col_start, self.y_jf_end+1)):
            y[:,self.c_out,self.y_if_end,col] += epsilon_error * x[:,self.w_cf,self.x_if_end,x_col_start+i*self.stride]

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_psum(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
        psum = x_unf[:,self.w_f-self.pe_row:self.w_f+1,self.sw_f].matmul(weights.view(weights.size(0), -1)[self.c_out,self.w_f-self.pe_row:self.w_f+1])
        
        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        sign_coeff = (psum.to(torch.int32) >> self.bit) & 1
        sign_coeff = torch.where(sign_coeff == 1, -1, 1)
        
        y[:,self.c_out,self.y_if,self.y_jf] += epsilon_tensor * sign_coeff
        return y

class FaultSA_IS_Conv():
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at=None):
        super(FaultSA_IS_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sa_size = sa_size
        self.in_size = in_size
        self.bsize_regs = 32
        self.bsize_psum = 32

        self.out_h = int((in_size + 2*self.padding - self.kernel_size) / self.stride + 1)
        self.out_w = self.out_h

        self.dmm_width = self.kernel_size*self.kernel_size*self.in_channels
        self.dmm_height = self.out_h * self.out_w
        self.kmm_height = self.out_channels
        self.latency = self.kmm_height + self.sa_size - 1 + self.sa_size - 1    # IS
        self.h_nsteps = ceil(self.dmm_height/self.sa_size)                      # IS
        self.k_nsteps = ceil(self.dmm_width/self.sa_size)                       # IS
        
        self.errors_func = [self.inject_errors_ireg, self.inject_errors_wreg, self.inject_errors_psum]
        self.inject_errors = self.inject_errors_ireg
        #self.generate_fault = self.generate_fault_ireg

    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.latency-1) if timestamp is None else timestamp # IS
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.dmm_height-self.h_step*self.sa_size-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-self.k_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.kmm_height+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)

        self.c_out_start = self.timestamp - self.pe_col - self.pe_row # IS
        self.c_out_end = self.out_channels

        w_f = self.k_step * self.sa_size + self.pe_row + 1                 # IS
        w_kf = w_f % (self.kernel_size*self.kernel_size)
        self.w_cf = floor(w_f/(self.kernel_size*self.kernel_size))
        self.w_if = floor(w_kf/self.kernel_size)
        self.w_jf = w_kf % self.kernel_size

        sw_f = self.h_step * self.sa_size + self.pe_col             # IS
        self.y_if = floor(sw_f/self.out_w)
        self.y_jf = sw_f % self.out_w

        self.x_if = self.y_if * self.stride + self.w_if - self.padding
        self.x_jf = self.y_jf * self.stride + self.w_jf - self.padding
        
        self.skip = True if (sw_f >= self.dmm_height) else False
        self.skip = True if (w_f < 0 or w_f >= self.dmm_width) else self.skip
        #self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.kmm_height+self.pe_row+self.pe_col) else self.skip
        
        # Check if x is a padded 0
        self.x_is_pad = True if (not 0 <= self.x_if < self.in_size) or (not 0 <= self.x_jf < self.in_size) else False

    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device

        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        if self.x_is_pad:
            if self.bit_type == 0:
                epsilon_tensor = torch.full((batch,), 0, device=device)
            elif self.bit_type == 1:
                epsilon_tensor = torch.pow(torch.full((batch,), 2.0, device=device), epsilon_tensor-127)
            else:
                epsilon_tensor *= torch.pow(torch.full((batch,), 2.0, device=device), -126)
        else:
            x_int = floatToBitsTensor(x[:,self.w_cf,self.x_if,self.x_jf], batch)
            sign_coeff = (x_int >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, -1, 1)
            if self.bit_type == 0:
                epsilon_tensor *= sign_coeff * x[:,self.w_cf,self.x_if,self.x_jf]
            elif self.bit_type == 1:
                epsilon_tensor = ((torch.pow(torch.full((batch,), 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * x[:,self.w_cf,self.x_if,self.x_jf]).to(torch.float)
            else:
                exponent = (x_int >> 23) & 0xFF
                epsilon_tensor *= (torch.pow(torch.full((batch,), 2.0, device=device), exponent - 127)) * sign_coeff

        for c in range(self.c_out_start, self.c_out_end):
            y[:,c,self.y_if,self.y_jf] += epsilon_tensor * weights[c,self.w_cf,self.w_if,self.w_jf]

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(0, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size,self.dmm_height-self.h_step*self.sa_size-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.dmm_width-self.k_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.kmm_height+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)

        self.c_out = self.timestamp - self.pe_col - self.pe_row         # IS

        w_f = self.k_step * self.sa_size + self.pe_row + 1                 # IS
        w_kf = w_f % (self.kernel_size*self.kernel_size)
        self.w_cf = floor(w_f/(self.kernel_size*self.kernel_size))
        self.w_if = floor(w_kf/self.kernel_size)
        self.w_jf = w_kf % self.kernel_size

        self.sw_start = self.h_step * self.sa_size + self.pe_col             # IS
        sw_end = (self.h_step + 1) * self.sa_size
        self.sw_end = sw_end if (self.out_h*self.out_w > sw_end) else (self.out_h*self.out_w - 1)

        self.y_if_start = floor(self.sw_start/self.out_w)
        self.y_jf_start = self.sw_start % self.out_w

        self.y_if_end = floor(self.sw_end/self.out_w)
        self.y_jf_end = self.sw_end % self.out_w

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
        
        self.y_nlines = self.y_if_end - self.y_if_start
        
        self.y_col_last = self.out_w if ((self.out_w - 1) * self.stride + self.w_jf - self.padding) < self.in_size else self.out_w - self.padding
        #self.y_nlines = self.y_nlines - 1 if is_top_pad or is_sbottom_pad or is_ebottom_pad else self.y_nlines
        #self.y_nlines = self.y_nlines - 1 if is_sright_pad else self.y_nlines
        self.y_nlines = -1 if is_sbottom_pad else self.y_nlines
        self.y_nlines = -1 if (self.c_out >= self.out_channels) else self.y_nlines
        self.y_nlines = -1 if (self.timestamp < self.pe_row + self.pe_col) else self.y_nlines
        self.y_nlines = -1 if (self.timestamp >= self.kmm_height+self.pe_row+self.pe_col) else self.y_nlines
        self.y_nlines = -1 if (self.w_cf >= self.in_channels) else self.y_nlines    # IS

    def inject_errors_wreg(self, x, weights, y):
        if (self.y_nlines < 0):
            return y

        w_int = floatToBits(weights[self.c_out,self.w_cf,self.w_if,self.w_jf])
        sign_coeff = (w_int >> self.bit) & 1
        sign_coeff = -1 if sign_coeff == 1 else 1

        if self.bit_type == 0:
            epsilon_error = self.epsilon * sign_coeff * weights[self.c_out,self.w_cf,self.w_if,self.w_jf]
        elif self.bit_type == 1:
            epsilon_error = ((2.0**(sign_coeff*self.epsilon) - 1) * weights[self.c_out,self.w_cf,self.w_if,self.w_jf].to(torch.double)).to(torch.float)
        else:
            exponent = (w_int >> 23) & 0xFF
            epsilon_error = self.epsilon * 2**(exponent - 127) * sign_coeff

        y_col_start = self.y_jf_start
        x_col_start = self.x_jf_start
        if (self.y_nlines != 0):
            y_col_start = self.y_jf_start
            x_col_start = self.x_jf_start
            for row in range(self.y_nlines):
                for i, col in enumerate(range(y_col_start, self.y_col_last)):
                    y[:,self.c_out,self.y_if_start+row,col] += epsilon_error * x[:,self.w_cf,self.x_if_start+row*self.stride,x_col_start+i*self.stride]
                x_col_start = self.w_jf - self.padding
                y_col_start = self.padding if (x_col_start < 0) else 0
                x_col_start = 0 if (x_col_start < 0) else x_col_start

        for i, col in enumerate(range(y_col_start, self.y_jf_end+1)):
            y[:,self.c_out,self.y_if_end,col] += epsilon_error * x[:,self.w_cf,self.x_if_end,x_col_start+i*self.stride]

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
            self.timestamp = randint(0, self.latency-1) if timestamp is None else timestamp
        else:
            self.pe_col = randint(0, min(self.sa_size,self.dmm_height-self.h_step*self.sa_size-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.dmm_width-self.k_step*self.sa_size-1)) if pe_row is None else pe_row
            self.timestamp = randint(self.pe_col+self.pe_row, self.kmm_height+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        
        self.bit = randint(0, self.bsize_psum-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        self.c_out = self.timestamp - self.pe_col - self.pe_row                 # IS
        self.w_f = self.k_step * self.sa_size + self.pe_row                  # IS
        
        self.sw_f = self.h_step * self.sa_size + self.pe_col + 1             # IS
        self.y_if = floor(self.sw_f/self.out_w)
        self.y_jf = self.sw_f % self.out_w
        
        self.skip = True if (self.sw_f >= self.dmm_height) else False
        self.skip = True if (self.w_f < 0 or self.w_f >= self.dmm_width) else self.skip
        self.skip = True if (self.c_out >= self.out_channels) else self.skip
        self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.kmm_height+self.pe_row+self.pe_col) else self.skip
        
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
        psum = x_unf[:,self.w_f-self.pe_row:self.w_f+1,self.sw_f].matmul(weights.view(weights.size(0), -1)[self.c_out,self.w_f-self.pe_row:self.w_f+1])
        
        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        #p_int = floatToBitsTensor(psum, batch)
        p_int = psum.view(torch.int32)
        sign_coeff = (p_int >> self.bit) & 1
        sign_coeff = torch.where(sign_coeff == 1, -1, 1)
        if self.bit_type == 0:
            epsilon_tensor *= sign_coeff * psum
        elif self.bit_type == 1:
            epsilon_tensor = ((torch.pow(torch.full((batch,), 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * psum).to(torch.float)
        else:
            exponent = (p_int >> 23) & 0xFF
            epsilon_tensor *= (torch.pow(torch.full((batch,), 2.0, device=device), exponent - 127)) * sign_coeff
        
        
        y[:,self.c_out,self.y_if,self.y_jf] += epsilon_tensor
        return y

class QFaultSA_IS_Conv(FaultSA_IS_Conv):
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at=None):
        super(QFaultSA_IS_Conv, self).__init__(sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding)
        self.bsize_regs = 8
        self.bsize_psum = 32
    
    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_ireg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device

        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        sign_coeff = torch.full((batch,), 1, device=device)
        if not self.x_is_pad:
            sign_coeff = (x[:,self.w_cf,self.x_if,self.x_jf].to(torch.int32) >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, -1, 1)

        for c in range(self.c_out_start, self.c_out_end):
            y[:,c,self.y_if,self.y_jf] += sign_coeff * epsilon_tensor * weights[c,self.w_cf,self.w_if,self.w_jf]

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_wreg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_wreg(self, x, weights, y):
        if (self.y_nlines < 0):
            return y

        sign_coeff = (int(weights[self.c_out,self.w_cf,self.w_if,self.w_jf]) >> self.bit) & 1
        sign_coeff = -1 if sign_coeff == 1 else 1
        
        epsilon_error = self.epsilon * sign_coeff

        y_col_start = self.y_jf_start
        x_col_start = self.x_jf_start
        if (self.y_nlines != 0):
            y_col_start = self.y_jf_start
            x_col_start = self.x_jf_start
            for row in range(self.y_nlines):
                for i, col in enumerate(range(y_col_start, self.y_col_last)):
                    y[:,self.c_out,self.y_if_start+row,col] += epsilon_error * x[:,self.w_cf,self.x_if_start+row*self.stride,x_col_start+i*self.stride]
                x_col_start = self.w_jf - self.padding
                y_col_start = self.padding if (x_col_start < 0) else 0
                x_col_start = 0 if (x_col_start < 0) else x_col_start

        for i, col in enumerate(range(y_col_start, self.y_jf_end+1)):
            y[:,self.c_out,self.y_if_end,col] += epsilon_error * x[:,self.w_cf,self.x_if_end,x_col_start+i*self.stride]

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_psum(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
        psum = x_unf[:,self.w_f-self.pe_row:self.w_f+1,self.sw_f].matmul(weights.view(weights.size(0), -1)[self.c_out,self.w_f-self.pe_row:self.w_f+1])
        
        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)

        sign_coeff = (psum.to(torch.int32) >> self.bit) & 1
        sign_coeff = torch.where(sign_coeff == 1, -1, 1)
        
        y[:,self.c_out,self.y_if,self.y_jf] += epsilon_tensor * sign_coeff
        return y

class FaultSA_OS_Conv():
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at=None):
        super(FaultSA_OS_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sa_size = sa_size
        self.in_size = in_size
        self.bsize_regs = 32
        self.bsize_psum = 32

        self.out_h = int((in_size + 2*self.padding - self.kernel_size) / self.stride + 1)
        self.out_w = self.out_h

        self.dmm_width = self.kernel_size*self.kernel_size*self.in_channels
        self.dmm_height = self.out_h * self.out_w
        kmm_height = self.out_channels
        self.latency = self.dmm_width + self.sa_size - 1 + self.sa_size - 1
        self.h_nsteps = ceil(self.dmm_height/self.sa_size)
        self.k_nsteps = ceil(kmm_height/self.sa_size)
        
        self.errors_func = [self.inject_errors_ireg, self.inject_errors_wreg, self.inject_errors_psum]
        self.inject_errors = self.inject_errors_ireg
        #self.generate_fault = self.generate_fault_ireg

    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        self.h_step = randint(0, self.h_nsteps-1) if h_step is None else h_step
        self.k_step = randint(0, self.k_nsteps-1) if k_step is None else k_step
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-self.k_step*self.sa_size-2)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.out_h*self.out_w-self.h_step*self.sa_size-1)) if pe_row is None else pe_row
        self.timestamp = randint(self.pe_col+self.pe_row, self.kernel_size*self.kernel_size*self.in_channels+self.pe_col+self.pe_row-1) if timestamp is None else timestamp
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)

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

        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        if self.x_is_pad:
            if self.bit_type == 0:
                epsilon_tensor = torch.full((batch,), 0, device=device)
            elif self.bit_type == 1:
                epsilon_tensor = torch.pow(torch.full((batch,), 2.0, device=device), epsilon_tensor-127)
            else:
                epsilon_tensor *= torch.pow(torch.full((batch,), 2.0, device=device), -126)
        else:
            x_int = floatToBitsTensor(x[:,self.w_cf,self.x_if,self.x_jf], batch)
            sign_coeff = (x_int >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, -1, 1)
            if self.bit_type == 0:
                epsilon_tensor *= sign_coeff * x[:,self.w_cf,self.x_if,self.x_jf]
            elif self.bit_type == 1:
                epsilon_tensor = ((torch.pow(torch.full((batch,), 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * x[:,self.w_cf,self.x_if,self.x_jf]).to(torch.float)
            else:
                exponent = (x_int >> 23) & 0xFF
                epsilon_tensor *= (torch.pow(torch.full((batch,), 2.0, device=device), exponent - 127)) * sign_coeff

        for c in range(self.c_out_start, self.c_out_end):
            y[:,c,self.y_if,self.y_jf] += epsilon_tensor * weights[c,self.w_cf,self.w_if,self.w_jf]

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
        
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)

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

        w_int = floatToBits(weights[self.c_out,self.w_cf,self.w_if,self.w_jf])
        sign_coeff = (w_int >> self.bit) & 1
        sign_coeff = -1 if sign_coeff == 1 else 1

        if self.bit_type == 0:
            epsilon_error = self.epsilon * sign_coeff * weights[self.c_out,self.w_cf,self.w_if,self.w_jf]
        elif self.bit_type == 1:
            epsilon_error = ((2.0**(sign_coeff*self.epsilon) - 1) * weights[self.c_out,self.w_cf,self.w_if,self.w_jf].to(torch.double)).to(torch.float)
        else:
            exponent = (w_int >> 23) & 0xFF
            epsilon_error = self.epsilon * 2**(exponent - 127) * sign_coeff

        y_col_start = self.y_jf_start
        x_col_start = self.x_jf_start
        if (self.y_nlines != 0):
            y_col_start = self.y_jf_start
            x_col_start = self.x_jf_start
            for row in range(self.y_nlines):
                for i, col in enumerate(range(y_col_start, self.y_col_last)):
                    y[:,self.c_out,self.y_if_start+row,col] += epsilon_error * x[:,self.w_cf,self.x_if_start+row*self.stride,x_col_start+i*self.stride]
                x_col_start = self.w_jf - self.padding
                y_col_start = self.padding if (x_col_start < 0) else 0
                x_col_start = 0 if (x_col_start < 0) else x_col_start

        for i, col in enumerate(range(y_col_start, self.y_jf_end+1)):
            y[:,self.c_out,self.y_if_end,col] += epsilon_error * x[:,self.w_cf,self.x_if_end,x_col_start+i*self.stride]

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
        
        self.bit = randint(0, self.bsize_psum-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        self.c_out = self.k_step * self.sa_size + self.pe_col
        self.w_f = self.timestamp - self.pe_col - self.pe_row
        
        self.sw_f = self.h_step * self.sa_size + self.pe_row
        self.y_if = floor(self.sw_f/self.out_w)
        self.y_jf = self.sw_f % self.out_w
        
        self.skip = True if (self.sw_f >= self.dmm_height) else False
        self.skip = True if (self.w_f < 0 or self.w_f >= self.dmm_width) else self.skip
        self.skip = True if (self.c_out >= self.out_channels) else self.skip
        self.skip = True if (self.timestamp < self.pe_row + self.pe_col) else self.skip
        self.skip = True if (self.timestamp >= self.kernel_size*self.kernel_size*self.in_channels+self.pe_row+self.pe_col) else self.skip
        
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
        psum = x_unf[:,self.w_f-self.pe_row:self.w_f+1,self.sw_f].matmul(weights.view(weights.size(0), -1)[self.c_out,self.w_f-self.pe_row:self.w_f+1])
        
        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        #p_int = floatToBitsTensor(psum, batch)
        p_int = psum.view(torch.int32)
        sign_coeff = (p_int >> self.bit) & 1
        sign_coeff = torch.where(sign_coeff == 1, -1, 1)
        if self.bit_type == 0:
            epsilon_tensor *= sign_coeff * psum
        elif self.bit_type == 1:
            epsilon_tensor = ((torch.pow(torch.full((batch,), 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * psum).to(torch.float)
        else:
            exponent = (p_int >> 23) & 0xFF
            epsilon_tensor *= (torch.pow(torch.full((batch,), 2.0, device=device), exponent - 127)) * sign_coeff
        
        
        y[:,self.c_out,self.y_if,self.y_jf] += epsilon_tensor
        return y

class QFaultSA_OS_Conv(FaultSA_OS_Conv):
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at=None):
        super(QFaultSA_OS_Conv, self).__init__(sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding)
        self.bsize_regs = 8
        self.bsize_psum = 32
    
    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_ireg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        sign_coeff = torch.full((batch,), 1, device=device)
        if not self.x_is_pad:
            sign_coeff = (x[:,self.w_cf,self.x_if,self.x_jf].to(torch.int32) >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, -1, 1)

        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)
        for c in range(self.c_out_start, self.c_out_end):
            y[:,c,self.y_if,self.y_jf] += sign_coeff * epsilon_tensor * weights[c,self.w_cf,self.w_if,self.w_jf]

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_wreg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
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
        super().generate_fault_psum(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
        psum = x_unf[:,self.w_f-self.pe_row:self.w_f+1,self.sw_f].matmul(weights.view(weights.size(0), -1)[self.c_out,self.w_f-self.pe_row:self.w_f+1])
        
        epsilon_tensor = torch.full((batch,), self.epsilon, device=device)

        sign_coeff = (psum.to(torch.int32) >> self.bit) & 1
        sign_coeff = torch.where(sign_coeff == 1, -1, 1)
        
        y[:,self.c_out,self.y_if,self.y_jf] += epsilon_tensor * sign_coeff
        return y