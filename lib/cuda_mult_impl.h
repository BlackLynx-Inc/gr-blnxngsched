/* -*- c++ -*- */
/*
 * Copyright 2021 gr-testmult author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_CUDA_MULT_IMPL_H
#define INCLUDED_BLNXNGSCHED_CUDA_MULT_IMPL_H

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <blnxngsched/cuda_mult.h>

namespace gr {
namespace blnxngsched {

class cuda_mult_impl : public cuda_mult
{
private:
    cudaStream_t d_stream;
    int d_min_grid_size;
    int d_block_size;

public:
    cuda_mult_impl();
    ~cuda_mult_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_CUDA_MULT_IMPL_H */
