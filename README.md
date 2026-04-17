# SAFPA: Reliability Assessment Method for DNN Inference on Systolic Arrays Based on Fault Propagation Analysis

The reliability assessment method combines fault injection with fault propagation analysis. Instead of modeling a systolic array on the microarchitecture level and performing time-consuming cycle-accurate simulations, fault propagation analysis is used to calculate the resulting error in the convolution layer output. This allows to noticeable speed up the reliability assessment of DNN inference on a systolic array. Method is described in the following papers: [FORTALESA: Fault-Tolerant Reconfigurable Systolic Array for DNN Inference](https://doi.org/10.1016/j.micpro.2025.105222) and [Special Session: Reliability Assessment of DNN Models and Inference on Systolic Arrays](https://) (Section II: SAFPA: Fast analytical reliability assessment for DNN Inference on Systolic Arrays).

The method allows to analyze transient and permanent faults in systolic arrays. Three systolic array dataflows are supported: output-stationary (OS), weight-stationary (WS), and input-stationary (IS).

Faults in registers holding intermediate values (IREG, WREG, PSUM) are considered for the analysis. Method supports both full-precision (FP32) and quantized models (INT8). For quantized models, input and weight registers are 8-bit long, and the output register holding partial sum is 32-bit long. Errors are injected layer-wise.

## Getting started

**Requirements**

- Pytorch >= 2.3 (<= 2.8 for Brevitas)
- Numpy
- Brevitas == 0.12.1

**Run experiments**

Clone this repo.
```
git clone https://github.com/ntlchr/safpa.git
cd safpa
```

For permanent faults, first, compile C library for permanent fault modeling using provided Makefile.
```
make all
```

Use one of the examples as a starting point. Examples present fault injection into quantized models. For quantized models, there are two ways to use the tool: with fully quantized networks that perform all operations with integer values (see `example_fully_quant.py`) and with networks quantized using [Brevitas](https://github.com/Xilinx/brevitas.git) that introduces quantizers but keeps floating point values (see `example_brevitas_quant.py`).

The tool provides two functions:

```python
def run_fault_injection(model, dataloader, device, sa_size, fault_num, dataflow, 
                        fault_type, quant, int_ops, stuck_at, skip_unused_pe):
  """Perform layer-wise fault injection into DNN model.
    
  Parameters:
    model: Pytorch DNN model for testing.
    dataloader: Test dataset loader.
    device: Torch device.
    sa_size: Size of the systolic array (default: 32).
    fault_num: Number of faults injected per each fault type, the total number
      of faults for each layer will be 3*fault_num (default: 100).
    dataflow: Systolic array dataflow: OS, WS or IS (default: 'OS').
    fault_type: Type of injected faults: transient or permanent (default: 'transient').
    quant: Whether model is quantized or not, int8 is assumed for quantized models (default: False).
    int_ops: Whether model is a fully quantized network that perform all operations with integer values (default: False).
    stuck_at: For permanent faults, the stuck-at value: 0 or 1 (default: 0).
    skip_unused_pe: Whether to skip unutilized PEs and only inject faults in the active ones (default: False).
  """
```

```python
def quantize(model, config, calib_loader, val_loader, device) -> quant_model:
  """
  Parameters:
	model: DNN model for quantization and subsequent testing.
	config: QuantConfig object holding configuration for quantization process.
	calib_loader: Calibration dataset loader.
	val_loader: Validation dataset loader.
	device: Torch device.
	
  Returns:
	quant_model: Quantized Brevitas model that can be then passed to run_fault_injection function.
  """
```

Results of fault injection are presented using Architectural vulnerability factor (AVF), the probability that a fault in the hardware structure causes an application output error. All results are saved in the log file.

## Citation

If you find this repo useful in your research, please consider citing the following papers:

```
@article{FORTALESA,
  title = {{FORTALESA:} Fault-Tolerant Reconfigurable Systolic Array for {DNN} Inference}, 
  author = {Natalia Cherezova and Artur Jutman and Maksim Jenihhin},
  journal = {Microprocessors and Microsystems},
  year = {2025},
  pages = {105222},
  issn = {0141-9331},
  doi = {https://doi.org/10.1016/j.micpro.2025.105222},
}
```

```
@inproceedings{SAFPA,
  title = {Special Session: Reliability Assessment of {DNN} Models and Inference on Systolic Arrays}, 
  author = {Natalia Cherezova and Salvatore Pappalardo and Bastien Deveautour and Lorenzo Fezza and Artur Jutman and Ernesto Sanchez and Alberto Bosio and Matteo Sonza Reorda and Maksim Jenihhin},
  booktitle = {EEE VLSI Test Symposium (VTS)},
  year = {2026},
  doi = {},
}
```