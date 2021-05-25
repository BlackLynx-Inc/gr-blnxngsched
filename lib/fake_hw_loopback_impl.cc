/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstring>

#include "fake_hw_loopback_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace blnxngsched {

using input_type = gr_complex;
using output_type = gr_complex;

fake_hw_loopback::sptr fake_hw_loopback::make(int batch_size)
{
    return gnuradio::make_block_sptr<fake_hw_loopback_impl>(batch_size);
}


/*
 * The private constructor
 */
fake_hw_loopback_impl::fake_hw_loopback_impl(int batch_size)
    : gr::block("fake_hw_loopback",
                gr::io_signature::make(
                    1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                gr::io_signature::make(
                    1 /* min outputs */, 1 /*max outputs */, sizeof(output_type))),
      d_batch_size(batch_size)
                
{
    set_output_multiple(d_batch_size);
}

/*
 * Our virtual destructor.
 */
fake_hw_loopback_impl::~fake_hw_loopback_impl() {}

void fake_hw_loopback_impl::forecast(int noutput_items,
                                     gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
}

int fake_hw_loopback_impl::general_work(int noutput_items,
                                        gr_vector_int& ninput_items,
                                        gr_vector_const_void_star& input_items,
                                        gr_vector_void_star& output_items)
{
    const input_type* in = reinterpret_cast<const input_type*>(input_items[0]);
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);


    std::memcpy(out, in, noutput_items * sizeof(input_type));

    // Do <+signal processing+>
    // Tell runtime system how many input items we consumed on
    // each input stream.
    consume_each(noutput_items);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace blnxngsched */
} /* namespace gr */
