/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstring>

#include "mixed_2_port_loopback_impl.h"
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

mixed_2_port_loopback::sptr mixed_2_port_loopback::make(int batch_size, int load) 
{
    return gnuradio::make_block_sptr<mixed_2_port_loopback_impl>(batch_size, load);
}

/*
 * The private constructor
 */
mixed_2_port_loopback_impl::mixed_2_port_loopback_impl(int batch_size, int load)
    : gr::block("mixed_2_port_loopback",
                gr::io_signature::make2(2 /* min inputs */, 2 /* max inputs */,
                                       sizeof(input_type), sizeof(input_type),
                                       cuda_buffer::type),
                gr::io_signature::make2(2 /* min outputs */, 2 /*max outputs */,
                                        sizeof(output_type), sizeof(output_type),
                                        cuda_buffer::type)),
      d_batch_size(batch_size),
      d_load(load),
      d_min_grid_size(0),
      d_block_size(0)
{
    get_block_and_grid(&d_min_grid_size, &d_block_size);
    //~ std::cout << "Min grid: " << d_min_grid_size << " -- Block size: " << d_block_size << std::endl;
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
mixed_2_port_loopback_impl::~mixed_2_port_loopback_impl() {}

void mixed_2_port_loopback_impl::forecast(int noutput_items, gr_vector_int &ninput_items_required) 
{
    ninput_items_required[0] = noutput_items;
    ninput_items_required[1] = noutput_items;
}

int mixed_2_port_loopback_impl::general_work(
    int noutput_items, 
    gr_vector_int &ninput_items,
    gr_vector_const_void_star &input_items, 
    gr_vector_void_star &output_items) 
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

        cudaStreamSynchronize(d_stream);
    }
    
    // Do the host side copy for grins
    const char* in_host = reinterpret_cast<const char*>(input_items[1]);
    char* out_host = reinterpret_cast<char*>(output_items[1]);
    
    for (uint32_t iter_idx = 0; iter_idx < num_iters; ++iter_idx)
    {
        std::memcpy((out_host + (iter_idx * d_batch_size)), 
                    (in_host + (iter_idx * d_batch_size)), 
                    d_batch_size * sizeof(gr_complex));
    }
    
    
    // Tell runtime system how many input items we consumed on
    // each input stream.
    consume(0, num_iters * d_batch_size);
    consume(1, num_iters * d_batch_size);
    
    produce(0, num_iters * d_batch_size);
    produce(1, num_iters * d_batch_size);

    // Tell runtime system how many output items we produced.
    return WORK_CALLED_PRODUCE;
}

} /* namespace blnxngsched */
} /* namespace gr */
