/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_loopback_impl.h"
#include <gnuradio/io_signature.h>

extern void apply_copy(const cuFloatComplex* in,
                       cuFloatComplex* out,
                       int grid_size,
                       int block_size,
                       int load,
                       cudaStream_t stream);
                       
extern void get_block_and_grid(int* minGrid, int* minBlock);

namespace gr {
namespace blnxngsched {

using input_type = gr_complex;
using output_type = gr_complex;


cuda_loopback::sptr cuda_loopback::make(int batch_size, int load)
{
    return gnuradio::make_block_sptr<cuda_loopback_impl>(batch_size, load);
}


/*
 * The private constructor
 */
cuda_loopback_impl::cuda_loopback_impl(int batch_size, int load)
    : gr::block("cuda_loopback",
                gr::io_signature::make(
                    1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                gr::io_signature::make(
                    1 /* min outputs */, 1 /*max outputs */, sizeof(output_type))),
      d_batch_size(batch_size),
      d_load(load),
      d_min_grid_size(0),
      d_block_size(0)
{
    get_block_and_grid(&d_min_grid_size, &d_block_size);
    cudaDeviceSynchronize();
    
    set_output_multiple(d_batch_size);
    
    cudaError_t rc = cudaStreamCreate(&d_stream);
    if (rc)
    {
        std::cerr << "Error creating stream: " << cudaGetErrorName(rc) 
                 << " -- " << cudaGetErrorString(rc);
    }
}

/*
 * Our virtual destructor.
 */
cuda_loopback_impl::~cuda_loopback_impl() {}

void cuda_loopback_impl::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
}

int cuda_loopback_impl::general_work(int noutput_items,
                                     gr_vector_int& ninput_items,
                                     gr_vector_const_void_star& input_items,
                                     gr_vector_void_star& output_items)
{
    const cuFloatComplex* in = reinterpret_cast<const cuFloatComplex*>(input_items[0]);
    cuFloatComplex* out = reinterpret_cast<cuFloatComplex*>(output_items[0]);
    
    cudaError_t rc = cudaSuccess;
    
    auto num_iters = noutput_items / d_batch_size;
    for (uint32_t iter_idx = 0; iter_idx < num_iters; ++iter_idx)
    {
#if 1
        apply_copy(in + (iter_idx * d_batch_size),
                   out + (iter_idx * d_batch_size),
                   d_batch_size / d_block_size,
                   d_block_size,
                   d_load,
                   d_stream);
                   

#else
        // For sanity check
        rc = cudaMemcpy((out + (iter_idx * d_batch_size)), 
                        (in + (iter_idx * d_batch_size)), 
                        d_batch_size * sizeof(gr_complex), 
                        cudaMemcpyHostToDevice);
        if (rc)
        {
            std::cerr << "Error performing cudaMemcpy: " << cudaGetErrorName(rc) 
                     << " -- " << cudaGetErrorString(rc);
        }
#endif   
    }
    
    // Can this go outside the loop?
    cudaStreamSynchronize(d_stream);
    
    // Tell runtime system how many input items we consumed on
    // each input stream.
    consume_each(num_iters * d_batch_size);

    // Tell runtime system how many output items we produced.
    return (num_iters * d_batch_size);
}

} /* namespace blnxngsched */
} /* namespace gr */
