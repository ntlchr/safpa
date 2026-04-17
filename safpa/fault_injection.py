"""Run fault injection."""

import numpy as np
import logging
from datetime import datetime

import torch

from .fault_modeling import Conv2d_, QuantConv2d_, add_fi_layer, add_qfi_layer, substitute_qconv_layers, substitute_conv_layers

logger = logging.getLogger(__name__)
        
def eval_faulty_model(model, layer, dataloader, golden_run, golden_run_top5, fault_num, device, int_ops, skip_unused_pe):
    
    N = golden_run.shape[0]
    
    # For fully integer operations we don't use softmax and rescale the logits
    if int_ops:
        post_proc = torch.nn.Identity()
    else:
        post_proc = torch.nn.Softmax(dim=1)
    
    functions = [layer.fault.generate_fault_ireg, layer.fault.generate_fault_wreg,
                 layer.fault.generate_fault_psum]
    fnames = ['IREG', 'WREG', 'PSUM']
    
    for i, generate_fault in enumerate(functions):
        top1_class = 0
        top1_acc = 0
        top5_class = 0
        top5_acc = 0
        layer.fault.inject_errors = layer.fault.errors_func[i]
        for _ in range(fault_num):
            faulty_run = []
            generate_fault(skip_unused_pe=skip_unused_pe)
            with torch.no_grad():
                for step, batch in enumerate(dataloader):
                    batch = tuple(t.to(device) for t in batch)  # wouldn't need for ImageNet dataset
                    x, y = batch
                    logits = model.forward(x)
                    logits = post_proc(logits)
            
                    if len(faulty_run) == 0:
                        faulty_run.append(logits.detach().cpu().numpy())
                    else:
                        faulty_run[0] = np.append(faulty_run[0], logits.detach().cpu().numpy(), axis=0)
            
            faulty_run = faulty_run[0]
            # Rescale
            if int_ops:
                min_act = np.min(faulty_run, axis=1)
                max_act = np.max(faulty_run, axis=1)
                scaling_factor = ((max_act - min_act) / (2 ** 8 - 1)).reshape(N,1)
                faulty_run = np.round(faulty_run/scaling_factor).astype(np.int8)
            
            faulty_run_top5 = np.argsort(faulty_run, axis=1)[:,::-1][:,:5].astype(np.int32)
            faulty_run = np.sort(faulty_run, axis=1)[:,::-1][:,:5]
            
            y, y_bar = golden_run_top5[:,0], faulty_run_top5[:,0]
            p, p_bar = golden_run[:,0], faulty_run[:,0]
            
            top1_class += (y != y_bar).sum()
            top5_class += (golden_run_top5[:,:] != faulty_run_top5[:,:]).any(axis=1).sum()
            if int_ops:
                top1_acc += ((y != y_bar) | (p != p_bar)).sum()
                top5_acc += ((golden_run_top5[:,:] != faulty_run_top5[:,:]).any(axis=1) | (golden_run != faulty_run).any(axis=1)).sum()
            else:
                top1_acc += ((y != y_bar) | (~np.isclose(p, p_bar, rtol=0.05, atol=0.05))).sum()
                top5_acc += ((golden_run_top5[:,:] != faulty_run_top5[:,:]).any(axis=1) | (~np.isclose(golden_run, faulty_run, rtol=0.05, atol=0.05)).any(axis=1)).sum()
            
        top1_class_avf = top1_class / (fault_num*N)
        top1_acc_avf = top1_acc / (fault_num*N)
        top5_class_avf = top5_class / (fault_num*N)
        top5_acc_avf = top5_acc / (fault_num*N)
        
        logger.info(fnames[i])
        logger.info(f"Top1-class: {top1_class_avf*100:.4f}%")
        logger.info(f"Top1-acc: {top1_acc_avf*100:.4f}%")
        logger.info(f"Top5-class: {top5_class_avf*100:.4f}%")
        logger.info(f"Top5-acc: {top5_acc_avf*100:.4f}%")


def run_fault_injection(model, dataloader, device, sa_size=32, fault_num=100, dataflow='OS', 
                        fault_type='transient', quant=False, int_ops=False, stuck_at=0, skip_unused_pe=False):
    """Perform layer-wise fault injection into DNN model.
    
    Parameters:
      model: Pytorch DNN model for testing.
      dataloader: Test dataset loader.
      device: Torch device.
      sa_size: Size of the systolic array (default: 32).
      fault_num: Number of faults injected per each fault type, the total number
        of faults for each layer will be 4*fault_num (default: 100).
      dataflow: Systolic array dataflow: OS, WS or IS (default: OS).
      fault_type: Type of injected faults: transient or permanent (default: transient).
      quant: Whether model is quantized or not, int8 is assumed for quantized
        models (default: False).
      int_ops: Whether model is a fully quantized network that perform all operations
        with integer values (default: False).
      stuck_at: For permanent faults, the stuck-at value: 0 or 1 (default: 0).
      skip_unused_pe: Whether to skip unutilized PEs and only inject faults in
        the active ones (default: False).
    """
    
    if dataflow not in ['OS', 'WS', 'IS']:
        print("ERROR: Invalid systolic array dataflow. Supported values: 'OS', 'WS', 'IS'.")
        return
    
    if fault_type not in ['transient', 'permanent']:
        print("ERROR: Invalid fault type. Supported values: 'transient', 'permanent'.")
        return
    
    if fault_type == 'permanent' and stuck_at not in [0,1]:
        print("ERROR: Invalid stuck-at value. Supported values: 0, 1.")
        return
    
    ftype = f"{'Q' if quant else 'FP'}_{dataflow}_{fault_type[0].upper()}"
    
    # Set up log file
    log_file_name = f"log-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')
    
    logger.info("Fault injection parameters")
    logger.info(f"Systolic array: {dataflow} {sa_size}x{sa_size}")
    logger.info(f"Faults: {fault_type} {'(s@'+str(stuck_at)+')' if fault_type == 'permanent' else ''}")
    logger.info(f"Number of injected faults: {fault_num} per reg")
    
    if quant:
        run_fault_injection_quant(model, dataloader, device, sa_size, fault_num, ftype, stuck_at, int_ops, skip_unused_pe)
    else:
        run_fault_injection_fp32(model, dataloader, device, sa_size, fault_num, ftype, stuck_at, skip_unused_pe)
    
    
def run_fault_injection_fp32(model, dataloader, device, sa_size, fault_num, ftype, stuck_at, skip_unused_pe):
    """Perform layer-wise fault injection into full-precision DNN model."""
    
    # Evaluate the model   
    substitute_conv_layers(model)
    model.eval()
    all_preds, all_label = [], []
    golden_run = []
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            logits = model.forward(x)
            preds = torch.argmax(logits, dim=-1)
            logits = torch.nn.functional.softmax(logits, dim=1)
    
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
                golden_run.append(logits.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
                all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
                golden_run[0] = np.append(golden_run[0], logits.detach().cpu().numpy(), axis=0)
    
    all_preds, all_label = all_preds[0], all_label[0]
    golden_run = golden_run[0]
    
    golden_run_top5 = np.argsort(golden_run, axis=1)[:,::-1][:,:5].astype(np.int32)
    golden_run = np.sort(golden_run, axis=1)[:,::-1][:,:5]
    
    accuracy = (all_preds == all_label).mean() * 100
    logger.info("Validation accuracy of the model: %.2f" % accuracy)
    
    # Run fault injection
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_):
            logger.info(name)
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            fi_layer = add_fi_layer(model, module, layer, tokens[-1], sa_size, ftype, stuck_at)
            eval_faulty_model(model, fi_layer, dataloader, golden_run, golden_run_top5, fault_num, device, False, skip_unused_pe)
            setattr(layer, tokens[-1], module)    # Put the original module back
            break
    logger.info("Fault injection finished")


def run_fault_injection_quant(model, dataloader, device, sa_size, fault_num, ftype, stuck_at, int_ops, skip_unused_pe):
    """Perform layer-wise fault injection into quantized DNN model."""
    
    # For fully integer operations we don't use softmax and rescale the logits
    if int_ops:
        post_proc = torch.nn.Identity()
    else:
        post_proc = torch.nn.Softmax(dim=1)
    
    # Evaluate the model   
    substitute_qconv_layers(model)
    model.eval()
    all_preds, all_label = [], []
    golden_run = []
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            logits = model.forward(x)
            preds = torch.argmax(logits, dim=-1)
            logits = post_proc(logits)
    
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
                golden_run.append(logits.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
                all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
                golden_run[0] = np.append(golden_run[0], logits.detach().cpu().numpy(), axis=0)
    
    all_preds, all_label = all_preds[0], all_label[0]
    golden_run = golden_run[0]
    
    # Rescale
    if int_ops:
        min_act = np.min(golden_run, axis=1)
        max_act = np.max(golden_run, axis=1)
        scaling_factor = ((max_act - min_act) / (2 ** 8 - 1)).reshape(golden_run.shape[0],1)
        golden_run = np.round(golden_run/scaling_factor).astype(np.int8)
    
    golden_run_top5 = np.argsort(golden_run, axis=1)[:,::-1][:,:5].astype(np.int32)
    golden_run = np.sort(golden_run, axis=1)[:,::-1][:,:5]
    
    accuracy = (all_preds == all_label).mean() * 100
    logger.info("Validation accuracy of the model: %.2f" % accuracy)
    
    # Run fault injection
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_) or isinstance(module, QuantConv2d_):
            logger.info(name)
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            fi_layer = add_qfi_layer(model, module, layer, tokens[-1], sa_size, ftype, stuck_at)
            eval_faulty_model(model, fi_layer, dataloader, golden_run, golden_run_top5, fault_num, device, int_ops, skip_unused_pe)
            setattr(layer, tokens[-1], module)    # Put the original module back
            break
    logger.info("Fault injection finished")