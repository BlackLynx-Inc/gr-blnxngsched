/* -*- c++ -*- */
/*
 * Copyright 2021 gr-testmult author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_FANOUT_H
#define INCLUDED_BLNXNGSCHED_FANOUT_H

#include <gnuradio/block.h>
#include <blnxngsched/api.h>

namespace gr {
namespace blnxngsched {

/*!
 * \brief <+description of block+>
 * \ingroup blnxngsched
 *
 */
class BLNXNGSCHED_API cuda_fanout : virtual public gr::block
{
public:
    typedef std::shared_ptr<cuda_fanout> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of blnxngsched::fanout.
     *
     * To avoid accidental use of raw pointers, blnxngsched::fanout's
     * constructor is in a private implementation
     * class. blnxngsched::fanout::make is the public interface for
     * creating new instances.
     */
    static sptr make();
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_FANOUT_H */
