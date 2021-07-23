/* -*- c++ -*- */
/*
 * Copyright 2021 gr-testmult author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLNXNGSCHED_MULT_H
#define INCLUDED_BLNXNGSCHED_MULT_H

#include <gnuradio/sync_block.h>
#include <blnxngsched/api.h>

namespace gr {
namespace blnxngsched {

/*!
 * \brief <+description of block+>
 * \ingroup blnxngsched
 *
 */
class BLNXNGSCHED_API cuda_mult : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<cuda_mult> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of blnxngsched::mult.
     *
     * To avoid accidental use of raw pointers, blnxngsched::mult's
     * constructor is in a private implementation
     * class. blnxngsched::mult::make is the public interface for
     * creating new instances.
     */
    static sptr make();
};

} // namespace blnxngsched
} // namespace gr

#endif /* INCLUDED_BLNXNGSCHED_MULT_H */
