/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <cstring>
#include <sstream>
#include <stdexcept>

#include <gnuradio/block.h>
#include "custom_buffer.h"

namespace gr {
namespace blnxngsched {
    
buffer_type custom_buffer::type(buftype_CUSTOM_HOST{});

void* custom_buffer::device_memcpy(void* dest, const void* src, std::size_t count)
{
    // There is no spoon...er... device so fake it out using regular memcpy
    return std::memcpy(dest, src, count);
}

void* custom_buffer::device_memmove(void* dest, const void* src, std::size_t count)
{
    // There is no spoon...er... device so fake it out using regular memmmove
    return std::memmove(dest, src, count);
}


custom_buffer::custom_buffer(int nitems,
                             size_t sizeof_item,
                             uint64_t downstream_lcm_nitems,
                             uint32_t downstream_max_out_mult,
                             block_sptr link,
                             block_sptr buf_owner)
    : buffer_single_mapped(nitems, sizeof_item, downstream_lcm_nitems, 
                           downstream_max_out_mult, link, buf_owner),
      d_device_base(nullptr)
{
    gr::configure_default_loggers(d_logger, d_debug_logger, "custom_buffer");
    if (!allocate_buffer(nitems))
        throw std::bad_alloc();
}

custom_buffer::~custom_buffer() {}

void custom_buffer::post_work(int nitems)
{
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "custom_buffer [" << d_context << "] -- post_work: " << nitems;
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    if (nitems <= 0) {
        return;
    }

    // NOTE: when this function is called the write pointer has not yet been
    // advanced so it can be used directly as the source ptr
    switch (d_context) {
    case buffer_context::HOST_TO_DEVICE: {
        // Copy data from host buffer to device buffer
        void* dest_ptr = &d_device_base[d_write_index * d_sizeof_item];
        device_memcpy(dest_ptr, write_pointer(), nitems * d_sizeof_item);
    } break;

    case buffer_context::DEVICE_TO_HOST: {
        // Copy data from device buffer to host buffer
        void* dest_ptr = &d_base[d_write_index * d_sizeof_item];
        device_memcpy(dest_ptr, write_pointer(), nitems * d_sizeof_item);
    } break;

    case buffer_context::DEVICE_TO_DEVICE:
        // No op
        break;

    default:
        std::ostringstream msg;
        msg << "Unexpected context for custom_buffer: " << d_context;
        GR_LOG_ERROR(d_logger, msg.str());
        throw std::runtime_error(msg.str());
    }
}

bool custom_buffer::do_allocate_buffer(size_t final_nitems, size_t sizeof_item)
{
#ifdef BUFFER_DEBUG
    {
        std::ostringstream msg;
        msg << "[" << this << "] "
            << "custom_buffer constructor -- nitems: " << final_nitems;
        GR_LOG_DEBUG(d_logger, msg.str());
    }
#endif

    // This is the host buffer
    d_buffer.reset(new char[final_nitems * sizeof_item]());
    d_base = d_buffer.get();

    // This is the simulated device buffer
    d_device_buf.reset(new char[final_nitems * sizeof_item]());
    d_device_base = d_device_buf.get();

    return true;
}

void* custom_buffer::write_pointer()
{
    void* ptr = nullptr;
    switch (d_context) {
    case buffer_context::HOST_TO_DEVICE:
        // Write into host buffer
        ptr = &d_base[d_write_index * d_sizeof_item];
        break;

    case buffer_context::DEVICE_TO_HOST:
    case buffer_context::DEVICE_TO_DEVICE:
        // Write into "device" buffer
        ptr = &d_device_base[d_write_index * d_sizeof_item];
        break;

    default:
        std::ostringstream msg;
        msg << "Unexpected context for custom_buffer: " << d_context;
        GR_LOG_ERROR(d_logger, msg.str());
        throw std::runtime_error(msg.str());
    }

    return ptr;
}

const void* custom_buffer::_read_pointer(unsigned int read_index)
{
    void* ptr = nullptr;
    switch (d_context) {
    case buffer_context::HOST_TO_DEVICE:
    case buffer_context::DEVICE_TO_DEVICE:
        // Read from "device" buffer
        ptr = &d_device_base[read_index * d_sizeof_item];
        break;

    case buffer_context::DEVICE_TO_HOST:
        // Read from host buffer
        ptr = &d_base[read_index * d_sizeof_item];
        break;

    default:
        std::ostringstream msg;
        msg << "Unexpected context for custom_buffer: " << d_context;
        GR_LOG_ERROR(d_logger, msg.str());
        throw std::runtime_error(msg.str());
    }

    return ptr;
}

bool custom_buffer::input_blocked_callback(int items_required,
                                         int items_avail,
                                         unsigned read_index)
{
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "custom_buffer [" << d_context << "] -- input_blocked_callback";
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    bool rc = false;
    switch (d_context) {
    case buffer_context::HOST_TO_DEVICE:
    case buffer_context::DEVICE_TO_DEVICE:
        // Adjust "device" buffer
        rc = input_blocked_callback_logic(items_required,
                                          items_avail,
                                          read_index,
                                          d_device_base,
                                          custom_buffer::device_memcpy,
                                          custom_buffer::device_memmove);
        break;

    case buffer_context::DEVICE_TO_HOST:
        // Adjust host buffer
        rc = input_blocked_callback_logic(
            items_required, items_avail, read_index, d_base, std::memcpy, std::memmove);
        break;

    default:
        std::ostringstream msg;
        msg << "Unexpected context for custom_buffer: " << d_context;
        GR_LOG_ERROR(d_logger, msg.str());
        throw std::runtime_error(msg.str());
    }

    return rc;
}

bool custom_buffer::output_blocked_callback(int output_multiple, bool force)
{
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "custom_buffer [" << d_context << "] -- output_blocked_callback";
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    bool rc = false;
    switch (d_context) {
    case buffer_context::HOST_TO_DEVICE:
        // Adjust host buffer
        rc = output_blocked_callback_logic(output_multiple, force, d_base, std::memmove);
        break;

    case buffer_context::DEVICE_TO_HOST:
    case buffer_context::DEVICE_TO_DEVICE:
        // Adjust "device" buffer
        rc = output_blocked_callback_logic(
            output_multiple, force, d_device_base, custom_buffer::device_memmove);
        break;

    default:
        std::ostringstream msg;
        msg << "Unexpected context for custom_buffer: " << d_context;
        GR_LOG_ERROR(d_logger, msg.str());
        throw std::runtime_error(msg.str());
    }

    return rc;
}

buffer_sptr custom_buffer::make_custom_buffer(int nitems,
                                              size_t sizeof_item,
                                              uint64_t downstream_lcm_nitems,
                                              uint32_t downstream_max_out_mult,
                                              block_sptr link,
                                              block_sptr buf_owner)
{
    return buffer_sptr(new custom_buffer(nitems, sizeof_item, downstream_lcm_nitems, 
                                         downstream_max_out_mult, link, buf_owner));
}


} /* namespace blnxngsched */
} /* namespace gr */
