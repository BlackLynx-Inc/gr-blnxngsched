/* -*- c++ -*- */
/*
 * Copyright 2021 gr-testmult author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_fanout_impl.h"
#include <gnuradio/io_signature.h>

#include <cuda_buffer/cuda_buffer.h>

namespace gr {
namespace blnxngsched {

cuda_fanout::sptr cuda_fanout::make() { return gnuradio::make_block_sptr<cuda_fanout_impl>(); }


/*
 * The private constructor
 */
cuda_fanout_impl::cuda_fanout_impl()
    : gr::block("fanout",
                gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */, 
                                       sizeof(gr_complex), cuda_buffer::type),
                gr::io_signature::make3(3 /* min outputs */, 3 /*max outputs */, 
                                        sizeof(gr_complex), sizeof(gr_complex), sizeof(float),
                                        cuda_buffer::type, cuda_buffer::type, cuda_buffer::type))
{
    set_history(64);
}

/*
 * Our virtual destructor.
 */
cuda_fanout_impl::~cuda_fanout_impl() {}

void cuda_fanout_impl::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
}

int cuda_fanout_impl::general_work(int noutput_items,
                              gr_vector_int& ninput_items,
                              gr_vector_const_void_star& input_items,
                              gr_vector_void_star& output_items)
{
    auto in = reinterpret_cast<const gr_complex*>(input_items[0]);
    auto out1 = reinterpret_cast<gr_complex*>(output_items[0]);
    auto out2 = reinterpret_cast<gr_complex*>(output_items[1]);
    auto out3 = reinterpret_cast<float*>(output_items[2]);

    cudaMemcpy(out1, in + 47, noutput_items * sizeof(gr_complex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out2, in, noutput_items * sizeof(gr_complex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out3, in + 63, noutput_items * sizeof(float),cudaMemcpyDeviceToDevice);

    // Do <+signal processing+>
    // Tell runtime system how many input items we consumed on
    // each input stream.
    consume_each(noutput_items);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace blnxngsched */
} /* namespace gr */
