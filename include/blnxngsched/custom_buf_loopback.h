/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_CUSTOM_BUF_LOOPBACK_H
#define INCLUDED_BLNXNGSCHED_CUSTOM_BUF_LOOPBACK_H

#include <gnuradio/block.h>
#include <blnxngsched/api.h>

namespace gr {
namespace blnxngsched {

/*!
 * \brief <+description of block+>
 * \ingroup blnxngsched
 *
 */
class BLNXNGSCHED_API custom_buf_loopback : virtual public gr::block
{
public:
    typedef std::shared_ptr<custom_buf_loopback> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of blnxngsched::custom_buf_loopback.
     *
     * To avoid accidental use of raw pointers, blnxngsched::custom_buf_loopback's
     * constructor is in a private implementation
     * class. blnxngsched::custom_buf_loopback::make is the public interface for
     * creating new instances.
     */
    static sptr make(int batch_size);
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_FAKE_HW_LOOPBACK_H */
