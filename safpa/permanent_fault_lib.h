#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

void inject_errors_ireg_ws(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_start, int *c_out_end,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at);
void inject_errors_wreg_ws(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_array,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at);

void inject_errors_ireg_os(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_start, int *c_out_end,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at);
void inject_errors_wreg_os(float *x, float *weights, float *y, int bit, int bit_type, int *c_out_array,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at);
void inject_errors_oreg_os(float *y, int bit, int bit_type, int *c_out_array, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at);

void inject_errors_ireg_is(float *x, float *weights, float *y, int bit, int bit_type, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at);
void inject_errors_wreg_is(float *x, float *weights, float *y, int bit, int bit_type, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at);

void qinject_errors_ireg_ws(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_start, int *c_out_end,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at);
void qinject_errors_wreg_ws(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_array,
					   int *w_cf, int *w_if, int *w_jf, int kernel_size, int in_size, int out_size, int in_channels,
					   int batch, int num_wf, int num_ch, int padding, int stride, int out_channels, int stuck_at);

void qinject_errors_ireg_os(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_start, int *c_out_end,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at);
void qinject_errors_wreg_os(int *x, int *weights, int *y, int bit, int epsilon, int *c_out_array,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at);
void qinject_errors_oreg_os(int *y, int bit, int epsilon, int *c_out_array, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int num_ch, int padding, int stride, int out_channels, int stuck_at);

void qinject_errors_ireg_is(int *x, int *weights, int *y, int bit, int epsilon, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *y_if_array, int *y_jf_array, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at);
void qinject_errors_wreg_is(int *x, int *weights, int *y, int bit, int epsilon, int *w_cf, int *w_if, int *w_jf, int num_wf,
					   int *sw_start, int *sw_end, int num_y, int kernel_size, int in_size, int out_size,
					   int in_channels, int batch, int padding, int stride, int out_channels, int stuck_at);

#endif