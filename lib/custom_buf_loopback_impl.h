/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_CUSTOM_BUF_LOOPBACK_IMPL_H
#define INCLUDED_BLNXNGSCHED_CUSTOM_BUF_LOOPBACK_IMPL_H

#include <blnxngsched/custom_buf_loopback.h>
#include "custom_buffer.h"

namespace gr {
namespace blnxngsched {

class custom_buf_loopback_impl : public custom_buf_loopback
{
private:
    int d_batch_size;
    

public:
    custom_buf_loopback_impl(int batch_size);
    ~custom_buf_loopback_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
                     
    buffer_type get_buffer_type()
    {
        // This will cause the host_buffer subclass to be used for this class
        return buftype_CUSTOM_HOST::get();
    }
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_CUSTOM_BUF_LOOPBACK_IMPL_H */
