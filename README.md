# SAFPA: Reliability Assessment Method for DNN Inference on Systolic Arrays Based on Fault Propagation Analysis

The reliability assessment method combines fault injection with fault propagation analysis. Instead of modeling a systolic array on the microarchitecture level and performing time-consuming cycle-accurate simulations, fault propagation analysis is used to calculate the resulting error in the convolution layer output. This allows to noticeable speed up the reliability assessment of DNN inference on a systolic array. Method is described in detail in the following paper: [FORTALESA: Fault-Tolerant Reconfigurable Systolic Array for DNN Inference](https://doi.org/10.1016/j.micpro.2025.105222) ([arXiv](http://arxiv.org/abs/2503.04426)).

The method allows to analyze transient faults in the output-stationary systolic arrays. Other systolic array dataflows (i.e., weight- and input-stationary) might be added in the future. Support for permanent faults will be added soon.

Faults in registers holding intermediate values (IREG, WREG, PSUM) and multipliers are considered for the analysis. Systolic array supporting signed integer arithmetic is considered. Input and weight registers are 8-bit long, and the output register holding partial sum is 32-bit long. Errors are injected layer-wise.

## Getting started

###Requirements###

- Pytorch >= 2.3 (<= 2.8 for Brevitas)
- Numpy
- Brevitas == 0.12.1

###Run experiments###

Clone this repo.
```
git clone https://github.com/ntlchr/safpa.git
cd safpa
```

Use one of the examples as a starting point. There are two ways to use the tool: with fully quantized networks that perform all operations with integer values (see `example_fully_quant.py`) and with networks quantized using [Brevitas](https://github.com/Xilinx/brevitas.git) that introduces quantizers but keeps floating point values (see `example_brevitas_quant.py`).

The tool provides two functions:

```python
def run_fault_injection(model, dataloader, device, sa_size, fault_num, int_ops, skip_unused_pe):
  """
  Parameters:
	model: Pytorch DNN model for testing.
	dataloader: Test dataset loader.
	device: Torch device.
	sa_size: Size of the systolic array (default: 32).
	fault_num: Number of faults injected per each fault type, the total number of faults
	  for each layer will be 4*fault_num (default: 100).
	int_ops: Whether model is a fully quantized network that perform all operations with integer values (default: False).
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

If you find this repo useful in your research, please consider citing the following paper:

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
