/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_CUDA_LOOPBACK_H
#define INCLUDED_BLNXNGSCHED_CUDA_LOOPBACK_H

#include <gnuradio/block.h>
#include <blnxngsched/api.h>

namespace gr {
namespace blnxngsched {

/*!
 * \brief <+description of block+>
 * \ingroup blnxngsched
 *
 */
class BLNXNGSCHED_API cuda_loopback : virtual public gr::block
{
public:
    typedef std::shared_ptr<cuda_loopback> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of blnxngsched::cuda_loopback.
     *
     * To avoid accidental use of raw pointers, blnxngsched::cuda_loopback's
     * constructor is in a private implementation
     * class. blnxngsched::cuda_loopback::make is the public interface for
     * creating new instances.
     */
    static sptr make(int batch_size, int load);
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_CUDA_LOOPBACK_H */
