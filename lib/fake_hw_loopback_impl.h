/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_FAKE_HW_LOOPBACK_IMPL_H
#define INCLUDED_BLNXNGSCHED_FAKE_HW_LOOPBACK_IMPL_H

#include <blnxngsched/fake_hw_loopback.h>

namespace gr {
namespace blnxngsched {

class fake_hw_loopback_impl : public fake_hw_loopback
{
private:
    int d_batch_size;
    

public:
    fake_hw_loopback_impl(int batch_size);
    ~fake_hw_loopback_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
                     
    buffer_type get_buffer_type()
    {
        // This will cause the host_buffer subclass to be used for this class
        return buftype_DEFAULT_HOST::get();
    }
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_FAKE_HW_LOOPBACK_IMPL_H */
