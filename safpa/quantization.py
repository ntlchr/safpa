"""Quantization using Brevitas."""

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_act_equalization
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate_bn
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.utils import validate


class QuantConfig():
    def __init__(self, export_dir='quant_model/', act_bit_width=8, weight_bit_width=8, bias_bit_width=32):
        super(QuantConfig, self).__init__()
        self.export_dir = export_dir
        self.act_bit_width = act_bit_width
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width
        self.act_quant_type = 'sym'
        self.weight_quant_type = 'sym'
        self.act_quant_calibration_type = 'stats'
        self.act_quant_percentile = 99.999
        self.target_backend = 'fx'
        self.graph_eq_iterations = 20
        self.graph_eq_merge_bias = True
        self.merge_bn = True
        self.scale_factor_type = 'float_scale'
        self.weight_quant_granularity = 'per_tensor'
        self.weight_quant_calibration_type = 'stats'
        self.act_equalization = None
        self.act_quant_calibration_type = 'stats'
        self.learned_round_iters = 1000
        self.learned_round_lr = 1e-3
        self.weight_narrow_range = False
        self.quant_format = 'int'
        self.layerwise_first_last_bit_width = 8
        self.layerwise_first_last_mantissa_bit_width = 4
        self.layerwise_first_last_exponent_bit_width = 3
        self.weight_mantissa_bit_width = 4
        self.weight_exponent_bit_width = 3
        self.act_mantissa_bit_width = 4
        self.act_exponent_bit_width = 3
        self.channel_splitting_ratio = 0.0
        self.learned_round = False
        self.calibrate_bn = False
        self.channel_splitting_split_input = False
        self.bias_corr = True
        


def quantize(model, config, calib_loader, val_loader, device):
    """Quantize DNN model using Brevitas.
    
    Parameters:
      model: Pytorch DNN model for quantization and subsequent testing.
      config: QuantConfig object holding configuration parameters for
        quantization process.
      calib_loader: Calibration datatset loader.
      val_lodaer: Validation dataset loader.
      device: Torch device.
        
    Returns:
      model: Quantized Brevitas model.
    """
    
    dtype = getattr(torch, 'float')
    
    # Preprocess the model for quantization
    if config.target_backend == 'flexml':
        # flexml requires static shapes, pass a representative input in
        img, _ = next(iter(calib_loader))
        img_shape = img.size()[2]
        model = preprocess_for_flexml_quantize(
            model,
            torch.ones(1, 3, img_shape, img_shape, dtype=dtype),
            equalize_iters=config.graph_eq_iterations,
            equalize_merge_bias=config.graph_eq_merge_bias,
            merge_bn=config.merge_bn)
    elif config.target_backend == 'fx' or config.target_backend == 'layerwise':
        model = preprocess_for_quantize(
            model,
            equalize_iters=config.graph_eq_iterations,
            equalize_merge_bias=config.graph_eq_merge_bias,
            merge_bn=config.merge_bn,
            channel_splitting_ratio=config.channel_splitting_ratio,
            channel_splitting_split_input=config.channel_splitting_split_input)
    else:
        raise RuntimeError(f"{config.target_backend} backend not supported.")
    
    if config.act_equalization is not None:
        print("Applying activation equalization:")
        apply_act_equalization(model, calib_loader, layerwise=config.act_equalization == 'layerwise')
        
    # Define the quantized model
    quant_model = quantize_model(
        model,
        dtype=dtype,
        device=device,
        backend=config.target_backend,
        scale_factor_type=config.scale_factor_type,
        bias_bit_width=config.bias_bit_width,
        weight_bit_width=config.weight_bit_width,
        weight_narrow_range=config.weight_narrow_range,
        weight_param_method=config.weight_quant_calibration_type,
        weight_quant_granularity=config.weight_quant_granularity,
        weight_quant_type=config.weight_quant_type,
        layerwise_first_last_bit_width=config.layerwise_first_last_bit_width,
        act_bit_width=config.act_bit_width,
        act_param_method=config.act_quant_calibration_type,
        act_quant_percentile=config.act_quant_percentile,
        act_quant_type=config.act_quant_type,
        quant_format=config.quant_format,
        layerwise_first_last_mantissa_bit_width=config.layerwise_first_last_mantissa_bit_width,
        layerwise_first_last_exponent_bit_width=config.layerwise_first_last_exponent_bit_width,
        weight_mantissa_bit_width=config.weight_mantissa_bit_width,
        weight_exponent_bit_width=config.weight_exponent_bit_width,
        act_mantissa_bit_width=config.act_mantissa_bit_width,
        act_exponent_bit_width=config.act_exponent_bit_width)
    
    # Calibrate the quant_model on the calibration dataloader
    print("Starting activation calibration:")
    calibrate(calib_loader, quant_model)

    if config.calibrate_bn:
        print("Calibrate BN:")
        calibrate_bn(calib_loader, quant_model)

    if config.bias_corr:
        print("Applying bias correction:")
        apply_bias_correction(calib_loader, quant_model)
        
    # Validate the quant_model on the validation dataloader
    print("Starting validation:")
    validate(val_loader, quant_model, stable=True)
    
    return model