/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_CUDA_LOOPBACK_IMPL_H
#define INCLUDED_BLNXNGSCHED_CUDA_LOOPBACK_IMPL_H

#include <cuComplex.h>
#include <blnxngsched/cuda_loopback.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_buffer/cuda_buffer.h>

namespace gr {
namespace blnxngsched {

class cuda_loopback_impl : public cuda_loopback
{
private:
    int d_batch_size;
    int d_load;
    int d_min_grid_size;
    int d_block_size;
    cudaStream_t d_stream;

public:
    cuda_loopback_impl(int batch_size, int load);
    ~cuda_loopback_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
                     
    buffer_type get_buffer_type()
    {        
        return buftype_DEFAULT_CUDA::get();
    }
                     
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_CUDA_LOOPBACK_IMPL_H */
