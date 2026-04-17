# -*- coding: utf-8 -*-
"""Fault modeling using fault propagation analysis for permanent faults."""

import torch
import torch.nn as nn
import numpy

import os
from math import ceil, floor
from random import randint
import struct

import ctypes


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
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at):
        super(FaultSA_WS_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sa_size = sa_size
        self.in_size = in_size
        self.stuck_at = stuck_at
        self.bsize_regs = 32
        self.bsize_psum = 32
        
        self.sign_value_one = 0 if stuck_at == 1 else -1
        self.sign_value_zero = 1 if stuck_at == 1 else 0

        self.out_h = int((in_size + 2*self.padding - self.kernel_size) / self.stride + 1)
        self.out_w = self.out_h

        self.dmm_width = self.kernel_size*self.kernel_size*self.in_channels
        self.dmm_height = self.out_h * self.out_w
        kmm_height = self.out_channels
        self.latency = self.dmm_height + self.sa_size - 1 + self.sa_size - 1
        self.h_nsteps = ceil(self.dmm_width/self.sa_size)   # WS
        self.k_nsteps = ceil(kmm_height/self.sa_size)
        
        self.errors_func = [self.inject_errors_ireg, self.inject_errors_wreg, self.inject_errors_psum]
        self.inject_errors = self.inject_errors_ireg
        
        self.lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "permanent_fault_lib.so"))
        self.inject_errors_ireg_cfn = self.lib.inject_errors_ireg_ws
        self.inject_errors_wreg_cfn = self.lib.inject_errors_wreg_ws
        #self.generate_fault = self.generate_fault_ireg

    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-1)) if pe_row is None else pe_row
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        c_out_start = [(i * self.sa_size + self.pe_col + 1) for i in range(self.k_nsteps)]     # PF
        c_out_start = [x for x in c_out_start if not x >= self.out_channels]              # PF
        self.c_out_start = numpy.array(c_out_start)
        c_out_end = [((i + 1) * self.sa_size) for i in range(self.k_nsteps)]                        # PF
        self.c_out_end = numpy.array(list(map(lambda x: x if self.out_channels >= x else self.out_channels, c_out_end))) # PF

        w_f = [(i * self.sa_size + self.pe_row) for i in range(self.h_nsteps)]      # PF
        w_f = [x for x in w_f if not x >= self.dmm_width]                           # PF
        w_kf = list(map(lambda x: x % (self.kernel_size*self.kernel_size), w_f))    # PF
        self.w_cf = numpy.array(list(map(lambda x: floor(x/(self.kernel_size*self.kernel_size)), w_f)))  # PF
        self.w_if = numpy.array(list(map(lambda x: floor(x/self.kernel_size), w_kf)))            # PF
        self.w_jf = numpy.array(list(map(lambda x: x % self.kernel_size, w_kf)))                 # PF
        
        self.skip = True if not w_f else False  # PF
        self.skip = True if not c_out_start else self.skip     # PF

    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(device='cpu', non_blocking=True)
        y_cpu = y.to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        pc_out_start = self.c_out_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pc_out_end = self.c_out_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.inject_errors_ireg_cfn(px, pw, py, self.bit, self.bit_type, pc_out_start, pc_out_end, pw_cf, pw_if, pw_jf,
                                self.kernel_size, self.in_size, self.out_w, self.in_channels, batch, len(self.w_cf),
                                len(self.c_out_start), self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-1)) if pe_row is None else pe_row
        
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        c_out = [(i * self.sa_size + self.pe_col) for i in range(self.k_nsteps)]    # PF
        c_out = [x for x in c_out if not x >= self.out_channels]                    # PF
        self.c_out = numpy.array(c_out)                                             # PF
        
        w_f = [(i * self.sa_size + self.pe_row) for i in range(self.h_nsteps)]      # PF
        w_f = [x for x in w_f if not x >= self.dmm_width]                           # PF
        w_kf = list(map(lambda x: x % (self.kernel_size*self.kernel_size), w_f))    # PF
        self.w_cf = numpy.array(list(map(lambda x: floor(x/(self.kernel_size*self.kernel_size)), w_f)))  # PF
        self.w_if = numpy.array(list(map(lambda x: floor(x/self.kernel_size), w_kf)))            # PF
        self.w_jf = numpy.array(list(map(lambda x: x % self.kernel_size, w_kf)))                 # PF
        
        self.skip = True if not w_f else False             # PF
        self.skip = True if not c_out else self.skip       # PF

    def inject_errors_wreg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(device='cpu', non_blocking=True)
        y_cpu = y.to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        pc_out = self.c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.inject_errors_wreg_cfn(px, pw, py, self.bit, self.bit_type, pc_out, pw_cf, pw_if, pw_jf, self.kernel_size, 
                                  self.in_size, self.out_w, self.in_channels, batch, len(self.w_cf), len(self.c_out), 
                                  self.padding, self.stride, self.out_channels, self.stuck_at)
        
        y = y_cpu.to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-1)) if pe_row is None else pe_row
        
        self.bit = randint(0, self.bsize_psum-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        self.c_out = [(i * self.sa_size + self.pe_col) for i in range(self.k_nsteps)]   # PF
        self.c_out = [x for x in self.c_out if not x >= self.out_channels]              # PF
        
        self.w_f = [(i * self.sa_size + self.pe_row) for i in range(self.h_nsteps)]      # PF
        self.w_f = [x for x in self.w_f if not x >= self.dmm_width]                      # PF
        
        self.skip = True if not self.w_f else False             # PF
        self.skip = True if not self.c_out else self.skip       # PF
        
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        epsilon_shape = (batch,len(self.c_out),self.dmm_height)
        for w_f in self.w_f:
            # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
            psum = x_unf[:,w_f-self.pe_row:w_f+1,:].transpose(1, 2).matmul(
                weights.view(weights.size(0), -1)[self.c_out,w_f-self.pe_row:w_f+1].t()).transpose(1, 2)
            
            epsilon_tensor = torch.full(epsilon_shape, self.epsilon, device=device)
            #p_int = floatToBitsTensor(psum, batch)
            p_int = psum.view(torch.int32)
            sign_coeff = (p_int >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, self.sign_value_one, self.sign_value_zero)
            if self.bit_type == 0:
                epsilon_tensor *= sign_coeff * psum
            elif self.bit_type == 1:
                epsilon_tensor = ((torch.pow(torch.full(epsilon_shape, 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * psum).to(torch.float)
            else:
                exponent = (p_int >> 23) & 0xFF
                epsilon_tensor *= (torch.pow(torch.full(epsilon_shape, 2.0, device=device), exponent - 127)) * sign_coeff
    
            y[:,self.c_out,:,:] += epsilon_tensor.view(batch, len(self.c_out), self.out_h, self.out_w)
        
        return y

class QFaultSA_WS_Conv(FaultSA_WS_Conv):
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at):
        super(QFaultSA_WS_Conv, self).__init__(sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at)
        self.bsize_regs = 8
        self.bsize_psum = 32
        
        self.qinject_errors_ireg_cfn = self.lib.qinject_errors_ireg_ws
        self.qinject_errors_wreg_cfn = self.lib.qinject_errors_wreg_ws
    
    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_ireg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        y_cpu = y.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        
        pc_out_start = self.c_out_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pc_out_end = self.c_out_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.qinject_errors_ireg_cfn(px, pw, py, self.bit, self.epsilon, pc_out_start, pc_out_end, pw_cf, pw_if, pw_jf,
                                self.kernel_size, self.in_size, self.out_w, self.in_channels, batch, len(self.w_cf),
                                len(self.c_out_start), self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(dtype=torch.float32).to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_wreg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_wreg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        y_cpu = y.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        
        pc_out = self.c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.qinject_errors_wreg_cfn(px, pw, py, self.bit, self.epsilon, pc_out, pw_cf, pw_if, pw_jf,
                                self.kernel_size, self.in_size, self.out_w, self.in_channels, batch, len(self.w_cf), 
                                len(self.c_out), self.padding, self.stride, self.out_channels, self.stuck_at)
        
        y = y_cpu.to(dtype=torch.float32).to(device=device, non_blocking=True)
        torch.cuda.synchronize()

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
        
        epsilon_shape = (batch,len(self.c_out),self.dmm_height)
        for w_f in self.w_f:
            # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
            psum = x_unf[:,w_f-self.pe_row:w_f+1,:].transpose(1, 2).matmul(
                weights.view(weights.size(0), -1)[self.c_out,w_f-self.pe_row:w_f+1].t()).transpose(1, 2)
            
            sign_coeff = (psum.to(torch.int32) >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, self.sign_value_one, self.sign_value_zero)
            
            epsilon_tensor = torch.full(epsilon_shape, self.epsilon, device=device)
            epsilon_tensor *= sign_coeff
    
            y[:,self.c_out,:,:] += epsilon_tensor.view(batch, len(self.c_out), self.out_h, self.out_w)
        
        return y


class FaultSA_IS_Conv():
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at):
        super(FaultSA_IS_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sa_size = sa_size
        self.in_size = in_size
        self.stuck_at = stuck_at
        self.bsize_regs = 32
        self.bsize_psum = 32
        
        self.sign_value_one = 0 if stuck_at == 1 else -1
        self.sign_value_zero = 1 if stuck_at == 1 else 0

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
        
        self.lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "permanent_fault_lib.so"))
        self.inject_errors_ireg_cfn = self.lib.inject_errors_ireg_is
        self.inject_errors_wreg_cfn = self.lib.inject_errors_wreg_is
        #self.generate_fault = self.generate_fault_ireg

    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.dmm_height-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_width-1)) if pe_row is None else pe_row
            
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        w_f = [(i * self.sa_size + self.pe_row + 1) for i in range(self.k_nsteps)]  # PF
        w_f = [x for x in w_f if not x >= self.dmm_width]                           # PF
        w_kf = list(map(lambda x: x % (self.kernel_size*self.kernel_size), w_f))    # PF
        self.w_cf = numpy.array(list(map(lambda x: floor(x/(self.kernel_size*self.kernel_size)), w_f)))     # PF
        self.w_if = numpy.array(list(map(lambda x: floor(x/self.kernel_size), w_kf)))                       # PF
        self.w_jf = numpy.array(list(map(lambda x: x % self.kernel_size, w_kf)))                            # PF
        
        sw_f = [(i * self.sa_size + self.pe_col) for i in range(self.h_nsteps)]     # PF
        sw_f = [x for x in sw_f if not x >= self.dmm_height]                        # PF
        self.y_if = numpy.array(list(map(lambda x: floor(x/self.out_w), sw_f)))     # PF
        self.y_jf = numpy.array(list(map(lambda x: x % self.out_w, sw_f)))          # PF
        
        self.skip = True if not sw_f else False                     # PF
        self.skip = True if not w_f else self.skip                  # PF

    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device

        x_cpu = x.to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(device='cpu', non_blocking=True)
        y_cpu = y.to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        py_if = self.y_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_jf = self.y_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        self.inject_errors_ireg_cfn(px, pw, py, self.bit, self.bit_type, pw_cf, pw_if, pw_jf, len(self.w_cf), 
                               py_if, py_jf, len(self.y_if), self.kernel_size, self.in_size, self.out_w,
                               self.in_channels, batch, self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size,self.dmm_height-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.dmm_width-1)) if pe_row is None else pe_row
        
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)

        w_f = [(i * self.sa_size + self.pe_row + 1) for i in range(self.k_nsteps)]  # PF
        w_f = [x for x in w_f if not x >= self.dmm_width]                           # PF
        w_kf = list(map(lambda x: x % (self.kernel_size*self.kernel_size), w_f))    # PF
        self.w_cf = numpy.array(list(map(lambda x: floor(x/(self.kernel_size*self.kernel_size)), w_f)))     # PF
        self.w_if = numpy.array(list(map(lambda x: floor(x/self.kernel_size), w_kf)))                       # PF
        self.w_jf = numpy.array(list(map(lambda x: x % self.kernel_size, w_kf)))                            # PF
        
        sw_start = [(i * self.sa_size + self.pe_col) for i in range(self.h_nsteps)]     # PF
        sw_start = [x for x in sw_start if not x >= self.dmm_height]                    # PF
        self.sw_start = numpy.array(sw_start)                                           # PF
        sw_end = [((i + 1) * self.sa_size) for i in range(self.h_nsteps)]               # PF
        self.sw_end = numpy.array(list(map(lambda x: x if self.dmm_height >= x else self.dmm_height-1, sw_end))) # PF

        self.skip = True if not sw_start else False         # PF
        self.skip = True if not w_f else self.skip          # PF

    def inject_errors_wreg(self, x, weights, y):
        if self.skip:
            return y

        batch = x.size()[0]
        device = x.device

        x_cpu = x.to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(device='cpu', non_blocking=True)
        y_cpu = y.to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        py_start = self.sw_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_end = self.sw_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        self.inject_errors_wreg_cfn(px, pw, py, self.bit, self.bit_type, pw_cf, pw_if, pw_jf, len(self.w_cf), 
                               py_start, py_end, len(self.sw_start), self.kernel_size, self.in_size, self.out_w,
                               self.in_channels, batch, self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size,self.dmm_height-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.dmm_width-1)) if pe_row is None else pe_row
        
        self.bit = randint(0, self.bsize_psum-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        w_f = [(i * self.sa_size + self.pe_row + 1) for i in range(self.k_nsteps)]  # PF
        self.w_f = [x for x in w_f if not x >= self.dmm_width]                      # PF
        
        sw_f = [(i * self.sa_size + self.pe_col) for i in range(self.h_nsteps)]     # PF
        self.sw_f = [x for x in sw_f if not x >= self.dmm_height]                   # PF
        self.y_if = list(map(lambda x: floor(x/self.out_w), self.sw_f))     # PF
        self.y_jf = list(map(lambda x: x % self.out_w, self.sw_f))          # PF
        
        self.skip = True if not self.sw_f else False                     # PF
        self.skip = True if not self.w_f else self.skip                  # PF
        
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        # From [B,Cin,Hin,Win] to [B,Cin*K*K,Hout*Wout]
        x_unf = nn.functional.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        
        epsilon_shape = (batch, self.out_channels, len(self.sw_f))
        for w_f in self.w_f:
            # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
            psum = x_unf[:,w_f-self.pe_row:w_f+1,self.sw_f].transpose(1, 2).matmul(
                weights.view(weights.size(0), -1)[:,w_f-self.pe_row:w_f+1].t()).transpose(1, 2)
            
            epsilon_tensor = torch.full(epsilon_shape, self.epsilon, device=device)
            #p_int = floatToBitsTensor(psum, batch)
            p_int = psum.view(torch.int32)
            sign_coeff = (p_int >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, self.sign_value_one, self.sign_value_zero)
            if self.bit_type == 0:
                epsilon_tensor *= sign_coeff * psum
            elif self.bit_type == 1:
                epsilon_tensor = ((torch.pow(torch.full(epsilon_shape, 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * psum).to(torch.float)
            else:
                exponent = (p_int >> 23) & 0xFF
                epsilon_tensor *= (torch.pow(torch.full(epsilon_shape, 2.0, device=device), exponent - 127)) * sign_coeff
    
            y[:,:,self.y_if,self.y_jf] += epsilon_tensor

        return y

class QFaultSA_IS_Conv(FaultSA_IS_Conv):
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at):
        super(QFaultSA_IS_Conv, self).__init__(sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at)
        self.bsize_regs = 8
        self.bsize_psum = 32
        
        self.qinject_errors_ireg_cfn = self.lib.qinject_errors_ireg_is
        self.qinject_errors_wreg_cfn = self.lib.qinject_errors_wreg_is
    
    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_ireg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device

        x_cpu = x.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        y_cpu = y.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        
        py_if = self.y_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_jf = self.y_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        self.qinject_errors_ireg_cfn(px, pw, py, self.bit, self.epsilon, pw_cf, pw_if, pw_jf, len(self.w_cf), 
                               py_if, py_jf, len(self.y_if), self.kernel_size, self.in_size, self.out_w,
                               self.in_channels, batch, self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(dtype=torch.float32).to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_wreg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_wreg(self, x, weights, y):
        if self.skip:
            return y

        batch = x.size()[0]
        device = x.device

        x_cpu = x.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        y_cpu = y.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        
        py_start = self.sw_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_end = self.sw_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_cf = self.w_cf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_if = self.w_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pw_jf = self.w_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        self.qinject_errors_wreg_cfn(px, pw, py, self.bit, self.epsilon, pw_cf, pw_if, pw_jf, len(self.w_cf), 
                               py_start, py_end, len(self.sw_start), self.kernel_size, self.in_size, self.out_w,
                               self.in_channels, batch, self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(dtype=torch.float32).to(device=device, non_blocking=True)
        torch.cuda.synchronize()

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
        
        # for c_out in range(self.out_channels):
        #     for sw_f, y_if, y_jf in zip(self.sw_f, self.y_if, self.y_jf):
        epsilon_shape = (batch, self.out_channels, len(self.sw_f))
        for w_f in self.w_f:
            # w.view(w.size(0),-1) has shape [Cout,Cin*K*K]
            psum = x_unf[:,w_f-self.pe_row:w_f+1,self.sw_f].transpose(1, 2).matmul(
                weights.view(weights.size(0), -1)[:,w_f-self.pe_row:w_f+1].t()).transpose(1, 2)
            
            sign_coeff = (psum.to(torch.int32) >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, self.sign_value_one, self.sign_value_zero)
            
            epsilon_tensor = torch.full(epsilon_shape, self.epsilon, device=device)
            epsilon_tensor *= sign_coeff
    
            y[:,:,self.y_if,self.y_jf] += epsilon_tensor

        return y
    

class FaultSA_OS_Conv():
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at):
        super(FaultSA_OS_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sa_size = sa_size
        self.in_size = in_size
        self.stuck_at = stuck_at
        self.bsize_regs = 32
        self.bsize_psum = 32
        
        self.sign_value_one = 0 if stuck_at == 1 else -1
        self.sign_value_zero = 1 if stuck_at == 1 else 0

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
        
        self.lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "permanent_fault_lib.so"))
        self.inject_errors_ireg_cfn = self.lib.inject_errors_ireg_os
        self.inject_errors_wreg_cfn = self.lib.inject_errors_wreg_os
        self.inject_errors_oreg_cfn = self.lib.inject_errors_oreg_os
        #self.generate_fault = self.generate_fault_ireg

    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size-1,self.out_channels-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size-1,self.dmm_height-1)) if pe_row is None else pe_row
        
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        c_out_start = [(i * self.sa_size + self.pe_col + 1) for i in range(self.k_nsteps)]     # PF
        c_out_start = [x for x in c_out_start if not x >= self.out_channels]              # PF
        self.c_out_start = numpy.array(c_out_start)
        c_out_end = [((i + 1) * self.sa_size) for i in range(self.k_nsteps)]                        # PF
        self.c_out_end = numpy.array(list(map(lambda x: x if self.out_channels >= x else self.out_channels, c_out_end))) # PF
        
        sw_f = [(i * self.sa_size + self.pe_row) for i in range(self.h_nsteps)]     # PF
        sw_f = [x for x in sw_f if not x >= self.dmm_height]                        # PF
        self.y_if = numpy.array(list(map(lambda x: floor(x/self.out_w), sw_f)))                  # PF
        self.y_jf = numpy.array(list(map(lambda x: x % self.out_w, sw_f)))                       # PF
 
        self.skip = True if not sw_f else False                     # PF
        self.skip = True if not c_out_start else self.skip     # PF

    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(device='cpu', non_blocking=True)
        y_cpu = y.to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        pc_out_start = self.c_out_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pc_out_end = self.c_out_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_if = self.y_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_jf = self.y_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.inject_errors_ireg_cfn(px, pw, py, self.bit, self.bit_type, pc_out_start, pc_out_end, py_if, py_jf, len(self.y_if),
                                self.kernel_size, self.in_size, self.out_w, self.in_channels, batch,
                                len(self.c_out_start), self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size,self.out_channels-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.dmm_height-1)) if pe_row is None else pe_row
        
        self.bit = randint(0, self.bsize_regs-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        c_out = [(i * self.sa_size + self.pe_col) for i in range(self.k_nsteps)]    # PF
        c_out = [x for x in c_out if not x >= self.out_channels]                    # PF
        self.c_out = numpy.array(c_out)                                             # PF
        
        sw_start = [(i * self.sa_size + self.pe_row) for i in range(self.h_nsteps)]     # PF
        sw_start = [x for x in sw_start if not x >= self.dmm_height]                    # PF
        self.sw_start = numpy.array(sw_start)                                           # PF
        sw_end = [((i + 1) * self.sa_size) for i in range(self.h_nsteps)]               # PF
        self.sw_end = numpy.array(list(map(lambda x: x if self.dmm_height >= x else self.dmm_height-1, sw_end))) # PF

        self.skip = True if not sw_start else False        # PF
        self.skip = True if not c_out else self.skip       # PF
        
    def inject_errors_wreg(self, x, weights, y):
        if (self.skip < 0):
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(device='cpu', non_blocking=True)
        y_cpu = y.to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        pc_out = self.c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_start = self.sw_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_end = self.sw_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.inject_errors_wreg_cfn(px, pw, py, self.bit, self.bit_type, pc_out, py_start, py_end, len(self.sw_start),
                                self.kernel_size, self.in_size, self.out_w, self.in_channels, batch,
                                len(self.c_out), self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_psum(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        if skip_unused_pe == False:
            self.pe_col = randint(0, self.sa_size-1) if pe_col is None else pe_col
            self.pe_row = randint(0, self.sa_size-1) if pe_row is None else pe_row
        else:
            self.pe_col = randint(0, min(self.sa_size,self.out_channels-1)) if pe_col is None else pe_col
            self.pe_row = randint(0, min(self.sa_size,self.dmm_height-1)) if pe_row is None else pe_row
        
        self.bit = randint(0, self.bsize_psum-1) if bit is None else bit

        self.bit_type = 0 if self.bit == 31 else (1 if self.bit > 22 else 2)    # 0 sign, 1 exponent, 2 mantissa
        self.epsilon = -2.0 if self.bit_type == 0 else 2.0**(self.bit-23)
        
        c_out = [(i * self.sa_size + self.pe_col) for i in range(self.k_nsteps)]    # PF
        c_out = [x for x in c_out if not x >= self.out_channels]                    # PF
        self.c_out = numpy.array(c_out)                                             # PF
        
        self.skip = True if not c_out else False
        
    def inject_errors_psum(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        epsilon_shape = (batch,1,self.out_h,self.out_w)
        for c in self.c_out:
            epsilon_tensor = torch.full(epsilon_shape, self.epsilon, device=device)
            p_int = y[:,c,:,:].view(torch.int32)
            sign_coeff = (p_int >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, self.sign_value_one, self.sign_value_zero)
            if self.bit_type == 0:
                epsilon_tensor *= sign_coeff * y[:,c,:,:]
            elif self.bit_type == 1:
                epsilon_tensor = ((torch.pow(torch.full(epsilon_shape, 2.0, dtype=torch.double, device=device), sign_coeff*epsilon_tensor) - 1) * y[:,c,:,:]).to(torch.float)
            else:
                exponent = (p_int >> 23) & 0xFF
                epsilon_tensor *= (torch.pow(torch.full(epsilon_shape, 2.0, device=device), exponent - 127)) * sign_coeff
    
            y[:,c,:,:] += epsilon_tensor
        
        return y

class QFaultSA_OS_Conv(FaultSA_OS_Conv):
    def __init__(self, sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at):
        super(QFaultSA_OS_Conv, self).__init__(sa_size, in_size, in_channels, out_channels, kernel_size, stride, padding, stuck_at)
        self.bsize_regs = 8
        self.bsize_psum = 32
        
        self.qinject_errors_ireg_cfn = self.lib.qinject_errors_ireg_os
        self.qinject_errors_wreg_cfn = self.lib.qinject_errors_wreg_os
        self.qinject_errors_oreg_cfn = self.lib.qinject_errors_oreg_os
    
    def generate_fault_ireg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_ireg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_ireg(self, x, weights, y):
        if self.skip:
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        y_cpu = y.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        
        pc_out_start = self.c_out_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pc_out_end = self.c_out_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_if = self.y_if.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_jf = self.y_jf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.qinject_errors_ireg_cfn(px, pw, py, self.bit, self.epsilon, pc_out_start, pc_out_end, py_if, py_jf, 
                                   len(self.y_if), self.kernel_size, self.in_size, self.out_w, self.in_channels, batch, 
                                   len(self.c_out_start), self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(dtype=torch.float32).to(device=device, non_blocking=True)
        torch.cuda.synchronize()

        return y
    
    def generate_fault_wreg(self, timestamp=None, h_step=None, k_step=None, pe_col=None, pe_row=None, bit=None, skip_unused_pe=False):
        super().generate_fault_wreg(timestamp, h_step, k_step, pe_col, pe_row, bit, skip_unused_pe)
        self.epsilon = 2**self.bit
        self.epsilon = -self.epsilon if (self.bit == self.bsize_regs-1) else self.epsilon
    
    def inject_errors_wreg(self, x, weights, y):
        if (self.skip < 0):
            return y
        
        batch = x.size()[0]
        device = x.device
        
        x_cpu = x.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        weights_cpu = weights.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        y_cpu = y.to(dtype=torch.int32).to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()

        px = ctypes.cast(x_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        pw = ctypes.cast(weights_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        py = ctypes.cast(y_cpu.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        
        pc_out = self.c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_start = self.sw_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        py_end = self.sw_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.qinject_errors_wreg_cfn(px, pw, py, self.bit, self.epsilon, pc_out, py_start, py_end, len(self.sw_start),
                                self.kernel_size, self.in_size, self.out_w, self.in_channels, batch,
                                len(self.c_out), self.padding, self.stride, self.out_channels, self.stuck_at)
        y = y_cpu.to(dtype=torch.float32).to(device=device, non_blocking=True)
        torch.cuda.synchronize()

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
        
        epsilon_shape = (batch,self.out_h,self.out_w)
        for c in self.c_out:
            sign_coeff = (y[:,c,:,:].to(torch.int32) >> self.bit) & 1
            sign_coeff = torch.where(sign_coeff == 1, self.sign_value_one, self.sign_value_zero)
            
            epsilon_tensor = torch.full(epsilon_shape, self.epsilon, device=device)
            epsilon_tensor *= sign_coeff
    
            y[:,c,:,:] += epsilon_tensor
        
        return y