#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include "permanent_fault_lib.h"

// Define a union to access the same memory as both float and unsigned integer
union float_bits {
    float f;
    uint32_t u; // uint32_t ensures a 32-bit size, matching the typical float
};

// ----- FP -----
// WS
void inject_errors_ireg_ws(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_start, int *c_out_end,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	float weight, activation;
	float epsilon_error;
	
	int exponent;
	
	union float_bits fb;
	
	float epsilon = (bit_type == 0) ? -2.0 : pow(2.0, bit-23);
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int y_if = 0; y_if < out_size; y_if++) {
			for (int y_jf = 0; y_jf < out_size; y_jf++) {
				for (int c = 0; c < num_ch; c++) {
					for (int c_out = c_out_start[c]; c_out < c_out_end[c]; c_out++) {
						for (int i = 0; i < num_wf; i++) {
							int iw_cf = w_cf[i];
							int iw_if = w_if[i];
							int iw_jf = w_jf[i];
							int x_if = y_if * stride + iw_if - padding;
							int x_jf = y_jf * stride + iw_jf - padding;
							if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
								activation = 0.0;
								sign_coeff = stuck_at;
								if (bit_type == 0) {
									epsilon_error = 0.0;
								} else if (bit_type == 1) {
									epsilon_error = pow(2.0, epsilon-127) * sign_coeff;
								} else {
									epsilon_error = epsilon * pow(2.0, -126) * sign_coeff;
								}
							} else {
								activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
								// Convert float to int to find bit and sign_coeff
								fb.f = activation;
								uint32_t bits = fb.u;
								sign_coeff = (((bits >> bit) & 1) ^ stuck_at) * stuck_coeff;
								if (bit_type == 0) {
									epsilon_error = epsilon * sign_coeff * activation;
								} else if (bit_type == 1) {
									epsilon_error = (pow(2.0, sign_coeff * epsilon) - 1) * activation;
								} else {
									// Find exponent
									exponent = (bits >> 23) & 0xFF;
									epsilon_error = epsilon * pow(2.0, exponent - 127) * sign_coeff;
								}
							}
							weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
							y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * weight;
						}
					}
				}
			}
		}
	}
}

void inject_errors_wreg_ws(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_array,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	float weight, activation;
	float epsilon_error;
	
	int exponent;
	
	union float_bits fb;
	
	float epsilon = (bit_type == 0) ? -2.0 : pow(2.0, bit-23);
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int y_if = 0; y_if < out_size; y_if++) {
			for (int y_jf = 0; y_jf < out_size; y_jf++) {
				for (int c = 0; c < num_ch; c++) {
					int c_out = c_out_array[c];
					for (int i = 0; i < num_wf; i++) {
						int iw_cf = w_cf[i];
						int iw_if = w_if[i];
						int iw_jf = w_jf[i];
						int x_if = y_if * stride + iw_if - padding;
						int x_jf = y_jf * stride + iw_jf - padding;
						if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
							continue;
						} else {
							weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
							// Convert float to int to find bit and sign_coeff
							fb.f = weight;
							uint32_t bits = fb.u;
							sign_coeff = (((bits >> bit) & 1) ^ stuck_at) * stuck_coeff;
							if (bit_type == 0) {
								epsilon_error = epsilon * sign_coeff * weight;
							} else if (bit_type == 1) {
								epsilon_error = (pow(2.0, sign_coeff * epsilon) - 1) * weight;
							} else {
								// Find exponent
								exponent = (bits >> 23) & 0xFF;
								epsilon_error = epsilon * pow(2.0, exponent - 127) * sign_coeff;
							}
						}
						activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
						y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * activation;
					}
				}
			}
		}
	}

}

// OS
void inject_errors_ireg_os(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_start, int *c_out_end,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	float weight, activation;
	float epsilon_error;
	
	int exponent;
	
	union float_bits fb;
	
	float epsilon = (bit_type == 0) ? -2.0 : pow(2.0, bit-23);
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int w_cf = 0; w_cf < in_channels; w_cf++) {
			for (int w_if = 0; w_if < kernel_size; w_if++) {
				for (int w_jf = 0; w_jf < kernel_size; w_jf++) {
					for (int c = 0; c < num_ch; c++) {
						for (int c_out = c_out_start[c]; c_out < c_out_end[c]; c_out++) {
							for (int i = 0; i < num_y; i++) {
								int y_if = y_if_array[i];
								int y_jf = y_jf_array[i];
								int x_if = y_if * stride + w_if - padding;
								int x_jf = y_jf * stride + w_jf - padding;
								if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
									activation = 0.0;
									sign_coeff = stuck_at;
									if (bit_type == 0) {
										epsilon_error = 0.0;
									} else if (bit_type == 1) {
										epsilon_error = pow(2.0, epsilon-127) * sign_coeff;
									} else {
										epsilon_error = epsilon * pow(2.0, -126) * sign_coeff;
									}
								} else {
									activation = x[img*XBATCHSIZE + w_cf*XMATRIXSIZE + x_if*in_size + x_jf];
									// Convert float to int to find bit and sign_coeff
									fb.f = activation;
									uint32_t bits = fb.u;
									sign_coeff = (((bits >> bit) & 1) ^ stuck_at) * stuck_coeff;
									if (bit_type == 0) {
										epsilon_error = epsilon * sign_coeff * activation;
									} else if (bit_type == 1) {
										epsilon_error = (pow(2.0, sign_coeff * epsilon) - 1) * activation;
									} else {
										// Find exponent
										exponent = (bits >> 23) & 0xFF;
										epsilon_error = epsilon * pow(2.0, exponent - 127) * sign_coeff;
									}
								}
								weight = weights[c_out*WCHANNELSIZE + w_cf*WMATRIXSIZE + w_if*kernel_size + w_jf];
								y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * weight;
							}
						}
					}
				}
			}
		}
	}

}

void inject_errors_wreg_os(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_array,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	float weight, activation;
	float epsilon_error;
	
	int exponent;
	
	union float_bits fb;
	
	float epsilon = (bit_type == 0) ? -2.0 : pow(2.0, bit-23);
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int w_cf = 0; w_cf < in_channels; w_cf++) {
			for (int w_if = 0; w_if < kernel_size; w_if++) {
				for (int w_jf = 0; w_jf < kernel_size; w_jf++) {
					for (int c = 0; c < num_ch; c++) {
						int c_out = c_out_array[c];
						for (int i = 0; i < num_y; i++) {
							for (int sw_f = sw_start[i]; sw_f < sw_end[i]; sw_f++) {
								int y_if = sw_f / out_size;
								int y_jf = sw_f % out_size;
								int x_if = y_if * stride + w_if - padding;
								int x_jf = y_jf * stride + w_jf - padding;
								if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
									continue;
								} else {
									weight = weights[c_out*WCHANNELSIZE + w_cf*WMATRIXSIZE + w_if*kernel_size + w_jf];
									// Convert float to int to find bit and sign_coeff
									fb.f = weight;
									uint32_t bits = fb.u;
									sign_coeff = (((bits >> bit) & 1) ^ stuck_at) * stuck_coeff;
									if (bit_type == 0) {
										epsilon_error = epsilon * sign_coeff * weight;
									} else if (bit_type == 1) {
										epsilon_error = (pow(2.0, sign_coeff * epsilon) - 1) * weight;
									} else {
										// Find exponent
										exponent = (bits >> 23) & 0xFF;
										epsilon_error = epsilon * pow(2.0, exponent - 127) * sign_coeff;
									}
								}
								activation = x[img*XBATCHSIZE + w_cf*XMATRIXSIZE + x_if*in_size + x_jf];
								y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * activation;
							}
						}
					}
				}
			}
		}
	}

}

void inject_errors_oreg_os(float *y, int bit, int bit_type, int *c_out_array, int kernel_size,
						int in_size, int out_size, int in_channels, int batch, int num_ch, int padding,
						int stride, int out_channels, int stuck_at) {

	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	float orig_value;
	float epsilon_error;
	
	int exponent, sign_coeff;
	
	union float_bits fb;
	
	float epsilon = (bit_type == 0) ? -2.0 : pow(2.0, bit-23);
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int c = 0; c < num_ch; c++) {
			int c_out = c_out_array[c];
			for (int y_if = 0; y_if < out_size; y_if++) {
				for (int y_jf = 0; y_jf < out_size; y_jf++) {
					orig_value = y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf];
					// Convert float to int to find bit and sign_coeff
					fb.f = orig_value;
					uint32_t bits = fb.u;
					sign_coeff = (((bits >> bit) & 1) ^ stuck_at) * stuck_coeff;
					if (bit_type == 0) {
						epsilon_error = epsilon * sign_coeff * orig_value;
					} else if (bit_type == 1) {
						epsilon_error = (pow(2.0, sign_coeff * epsilon) - 1) * orig_value;
					} else {
						// Find exponent
						exponent = (bits >> 23) & 0xFF;
						epsilon_error = epsilon * pow(2.0, exponent - 127) * sign_coeff;
					}
					y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error;
				}
			}
		}
	}

}

// IS
void inject_errors_ireg_is(float *x, float *weights, float *y, int bit, int bit_type, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	float weight, activation;
	float epsilon_error;
	
	int exponent;
	
	union float_bits fb;
	
	float epsilon = (bit_type == 0) ? -2.0 : pow(2.0, bit-23);
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int c_out = 0; c_out < out_channels; c_out++) {
			for (int i = 0; i < num_y; i++) {
				int y_if = y_if_array[i];
				int y_jf = y_jf_array[i];
				for (int j = 0; j < num_wf; j++) {
					int iw_cf = w_cf[j];
					int iw_if = w_if[j];
					int iw_jf = w_jf[j];
					int x_if = y_if * stride + iw_if - padding;
					int x_jf = y_jf * stride + iw_jf - padding;
					if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
						activation = 0.0;
						sign_coeff = stuck_at;
						if (bit_type == 0) {
							epsilon_error = 0.0;
						} else if (bit_type == 1) {
							epsilon_error = pow(2.0, epsilon-127);
						} else {
							epsilon_error = epsilon * pow(2.0, -126);
						}
					} else {
						activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
						// Convert float to int to find bit and sign_coeff
						fb.f = activation;
						uint32_t bits = fb.u;
						sign_coeff = (((bits >> bit) & 1) ^ stuck_at) * stuck_coeff;
						if (bit_type == 0) {
							epsilon_error = epsilon * sign_coeff * activation;
						} else if (bit_type == 1) {
							epsilon_error = (pow(2.0, sign_coeff * epsilon) - 1) * activation;
						} else {
							// Find exponent
							exponent = (bits >> 23) & 0xFF;
							epsilon_error = epsilon * pow(2.0, exponent - 127) * sign_coeff;
						}
					}
					weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
					y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * weight;
				}
			}
		}
	}

}

void inject_errors_wreg_is(float *x, float *weights, float *y, int bit, int bit_type, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	float weight, activation;
	float epsilon_error;
	
	int exponent;
	
	union float_bits fb;
	
	float epsilon = (bit_type == 0) ? -2.0 : pow(2.0, bit-23);
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int c_out = 0; c_out < out_channels; c_out++) {
			for (int i = 0; i < num_y; i++) {
				for (int sw_f = sw_start[i]; sw_f < sw_end[i]; sw_f++) {
					int y_if = sw_f / out_size;
					int y_jf = sw_f % out_size;
					for (int j = 0; j < num_wf; j++) {
						int iw_cf = w_cf[j];
						int iw_if = w_if[j];
						int iw_jf = w_jf[j];
						int x_if = y_if * stride + iw_if - padding;
						int x_jf = y_jf * stride + iw_jf - padding;
						if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
							continue;
						} else {
							weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
							// Convert float to int to find bit and sign_coeff
							fb.f = weight;
							uint32_t bits = fb.u;
							sign_coeff = (((bits >> bit) & 1) ^ stuck_at) * stuck_coeff;
							if (bit_type == 0) {
								epsilon_error = epsilon * sign_coeff * weight;
							} else if (bit_type == 1) {
								epsilon_error = (pow(2.0, sign_coeff * epsilon) - 1) * weight;
							} else {
								// Find exponent
								exponent = (bits >> 23) & 0xFF;
								epsilon_error = epsilon * pow(2.0, exponent - 127) * sign_coeff;
							}
						}
						activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
						y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * activation;
					}
				}
			}
		}
	}

}

// ----- QUANT -----
// WS
void qinject_errors_ireg_ws(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_start, int *c_out_end,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	int weight, activation;
	int epsilon_error;
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int y_if = 0; y_if < out_size; y_if++) {
			for (int y_jf = 0; y_jf < out_size; y_jf++) {
				for (int c = 0; c < num_ch; c++) {
					for (int c_out = c_out_start[c]; c_out < c_out_end[c]; c_out++) {
						for (int i = 0; i < num_wf; i++) {
							int iw_cf = w_cf[i];
							int iw_if = w_if[i];
							int iw_jf = w_jf[i];
							int x_if = y_if * stride + iw_if - padding;
							int x_jf = y_jf * stride + iw_jf - padding;
							if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
								sign_coeff = stuck_at;
							} else {
								activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
								sign_coeff = (((activation >> bit) & 1) ^ stuck_at) * stuck_coeff;
							}
							epsilon_error = epsilon * sign_coeff;
							weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
							y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * weight;
						}
					}
				}
			}
		}
	}

}

void qinject_errors_wreg_ws(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_array,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	int weight, activation;
	int epsilon_error;
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int y_if = 0; y_if < out_size; y_if++) {
			for (int y_jf = 0; y_jf < out_size; y_jf++) {
				for (int c = 0; c < num_ch; c++) {
					int c_out = c_out_array[c];
					for (int i = 0; i < num_wf; i++) {
						int iw_cf = w_cf[i];
						int iw_if = w_if[i];
						int iw_jf = w_jf[i];
						int x_if = y_if * stride + iw_if - padding;
						int x_jf = y_jf * stride + iw_jf - padding;
						if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
							continue;
						} else {
							weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
							sign_coeff = (((weight >> bit) & 1) ^ stuck_at) * stuck_coeff;
							epsilon_error = epsilon * sign_coeff;
						}
						activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
						y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * activation;
					}
				}
			}
		}
	}

}

// OS
void qinject_errors_ireg_os(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_start, int *c_out_end,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	int weight, activation;
	int epsilon_error;
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int w_cf = 0; w_cf < in_channels; w_cf++) {
			for (int w_if = 0; w_if < kernel_size; w_if++) {
				for (int w_jf = 0; w_jf < kernel_size; w_jf++) {
					for (int c = 0; c < num_ch; c++) {
						for (int c_out = c_out_start[c]; c_out < c_out_end[c]; c_out++) {
							for (int i = 0; i < num_y; i++) {
								int y_if = y_if_array[i];
								int y_jf = y_jf_array[i];
								int x_if = y_if * stride + w_if - padding;
								int x_jf = y_jf * stride + w_jf - padding;
								if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
									activation = 0.0;
									sign_coeff = stuck_at;
								} else {
									activation = x[img*XBATCHSIZE + w_cf*XMATRIXSIZE + x_if*in_size + x_jf];
									sign_coeff = (((activation >> bit) & 1) ^ stuck_at) * stuck_coeff;
								}
								epsilon_error = epsilon * sign_coeff;
								weight = weights[c_out*WCHANNELSIZE + w_cf*WMATRIXSIZE + w_if*kernel_size + w_jf];
								y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * weight;
							}
						}
					}
				}
			}
		}
	}

}

void qinject_errors_wreg_os(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_array,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	int weight, activation;
	int epsilon_error;
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int w_cf = 0; w_cf < in_channels; w_cf++) {
			for (int w_if = 0; w_if < kernel_size; w_if++) {
				for (int w_jf = 0; w_jf < kernel_size; w_jf++) {
					for (int c = 0; c < num_ch; c++) {
						int c_out = c_out_array[c];
						for (int i = 0; i < num_y; i++) {
							for (int sw_f = sw_start[i]; sw_f < sw_end[i]; sw_f++) {
								int y_if = sw_f / out_size;
								int y_jf = sw_f % out_size;
								int x_if = y_if * stride + w_if - padding;
								int x_jf = y_jf * stride + w_jf - padding;
								if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
									continue;
								} else {
									weight = weights[c_out*WCHANNELSIZE + w_cf*WMATRIXSIZE + w_if*kernel_size + w_jf];
									sign_coeff = (((weight >> bit) & 1) ^ stuck_at) * stuck_coeff;
									epsilon_error = epsilon * sign_coeff;
								}
								activation = x[img*XBATCHSIZE + w_cf*XMATRIXSIZE + x_if*in_size + x_jf];
								y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * activation;
							}
						}
					}
				}
			}
		}
	}

}

void qinject_errors_oreg_os(int *y, int bit, int epsilon, int *c_out_array, int kernel_size,
						int in_size, int out_size, int in_channels, int batch, int num_ch, int padding,
						int stride, int out_channels, int stuck_at) {
		
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	int orig_value;
	int epsilon_error;
	int sign_coeff;
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int c = 0; c < num_ch; c++) {
			int c_out = c_out_array[c];
			for (int y_if = 0; y_if < out_size; y_if++) {
				for (int y_jf = 0; y_jf < out_size; y_jf++) {
					orig_value = y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf];
					sign_coeff = (((orig_value >> bit) & 1) ^ stuck_at) * stuck_coeff;
					epsilon_error = epsilon * sign_coeff;
					y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error;
				}
			}
		}
	}

}

// IS
void qinject_errors_ireg_is(int *x, int *weights, int *y, int bit, int epsilon, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	int weight, activation;
	int epsilon_error;
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int c_out = 0; c_out < out_channels; c_out++) {
			for (int i = 0; i < num_y; i++) {
				int y_if = y_if_array[i];
				int y_jf = y_jf_array[i];
				for (int j = 0; j < num_wf; j++) {
					int iw_cf = w_cf[j];
					int iw_if = w_if[j];
					int iw_jf = w_jf[j];
					int x_if = y_if * stride + iw_if - padding;
					int x_jf = y_jf * stride + iw_jf - padding;
					if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
						activation = 0.0;
						sign_coeff = stuck_at;
					} else {
						activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
						sign_coeff = (((activation >> bit) & 1) ^ stuck_at) * stuck_coeff;
					}
					epsilon_error = epsilon * sign_coeff;
					weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
					y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * weight;
				}
			}
		}
	}

}

void qinject_errors_wreg_is(int *x, int *weights, int *y, int bit, int epsilon, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at) {
	int sign_coeff;
	int XBATCHSIZE = in_channels * in_size * in_size;
	int XMATRIXSIZE = in_size * in_size;
	int WCHANNELSIZE = in_channels * kernel_size * kernel_size;
	int WMATRIXSIZE = kernel_size * kernel_size;	
	int YBATCHSIZE = out_channels * out_size * out_size;
	int YMATRIXSIZE = out_size * out_size;
	
	int weight, activation;
	int epsilon_error;
	
	int stuck_coeff = (stuck_at == 0) ? -1 : 1;
	
	for (int img = 0; img < batch; img++) {
		for (int c_out = 0; c_out < out_channels; c_out++) {
			for (int i = 0; i < num_y; i++) {
				for (int sw_f = sw_start[i]; sw_f < sw_end[i]; sw_f++) {
					int y_if = sw_f / out_size;
					int y_jf = sw_f % out_size;
					for (int j = 0; j < num_wf; j++) {
						int iw_cf = w_cf[j];
						int iw_if = w_if[j];
						int iw_jf = w_jf[j];
						int x_if = y_if * stride + iw_if - padding;
						int x_jf = y_jf * stride + iw_jf - padding;
						if (x_if < 0 || x_if >= in_size || x_jf < 0 || x_jf >= in_size) {
							continue;
						} else {
							weight = weights[c_out*WCHANNELSIZE + iw_cf*WMATRIXSIZE + iw_if*kernel_size + iw_jf];
							sign_coeff = (((weight >> bit) & 1) ^ stuck_at) * stuck_coeff;
							epsilon_error = epsilon * sign_coeff;
						}
						activation = x[img*XBATCHSIZE + iw_cf*XMATRIXSIZE + x_if*in_size + x_jf];
						y[img*YBATCHSIZE + c_out*YMATRIXSIZE + y_if*out_size + y_jf] += epsilon_error * activation;
					}
				}
			}
		}
	}

}