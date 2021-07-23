/* -*- c++ -*- */
/*
 * Copyright 2021 gr-testmult author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_mult_impl.h"
#include <gnuradio/io_signature.h>

#include <cuda_buffer/cuda_buffer.h>

void exec_multiply_kernel_ccc(const cuFloatComplex *in1, const cuFloatComplex *in2,
                              cuFloatComplex *out, int n, int grid_size,
                              int block_size, cudaStream_t stream);

void get_block_and_grid_multiply(int* minGrid, int* minBlock);

namespace gr {
namespace blnxngsched {

using input_type = gr_complex;
using output_type = gr_complex;
cuda_mult::sptr cuda_mult::make() { return gnuradio::make_block_sptr<cuda_mult_impl>(); }


/*
 * The private constructor
 */
cuda_mult_impl::cuda_mult_impl()
    : gr::sync_block("mult",
                     gr::io_signature::make(2 /* min inputs */, 2 /* max inputs */, 
                                            sizeof(input_type), cuda_buffer::type),
                     gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */, 
                                            sizeof(output_type), cuda_buffer::type))
{
    cudaStreamCreate(&d_stream);
    get_block_and_grid_multiply(&d_min_grid_size, &d_block_size);
    
    std::cout << "Min grid: " << d_min_grid_size 
              << " -- Block size: " << d_block_size << std::endl;
}

/*
 * Our virtual destructor.
 */
cuda_mult_impl::~cuda_mult_impl() {}

int cuda_mult_impl::work(int noutput_items,
                         gr_vector_const_void_star& input_items,
                         gr_vector_void_star& output_items)
{
    auto in1 = reinterpret_cast<const cuFloatComplex*>(input_items[0]);
    auto in2 = reinterpret_cast<const cuFloatComplex*>(input_items[1]);
    auto out = reinterpret_cast<cuFloatComplex*>(output_items[0]);

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    exec_multiply_kernel_ccc(in1, in2, out, noutput_items, gridSize, d_block_size, d_stream);
    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace blnxngsched */
} /* namespace gr */
