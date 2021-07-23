/* -*- c++ -*- */
/*
 * Copyright 2021 gr-testmult author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_CUDA_FANOUT_IMPL_H
#define INCLUDED_BLNXNGSCHED_CUDA_FANOUT_IMPL_H

#include <blnxngsched/cuda_fanout.h>

namespace gr {
namespace blnxngsched {

class cuda_fanout_impl : public cuda_fanout
{
private:
    // Nothing to declare in this block.

public:
    cuda_fanout_impl();
    ~cuda_fanout_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_CUDA_FANOUT_IMPL_H */
