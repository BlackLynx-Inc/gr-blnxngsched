/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(custom_buf_loopback.h)                                     */
/* BINDTOOL_HEADER_FILE_HASH(9d2c26793b9602052de355f6a3a6db18)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <blnxngsched/custom_buf_loopback.h>
// pydoc.h is automatically generated in the build directory
#include <custom_buf_loopback_pydoc.h>

void bind_custom_buf_loopback(py::module& m)
{

    using custom_buf_loopback    = gr::blnxngsched::custom_buf_loopback;


    py::class_<custom_buf_loopback, gr::block, gr::basic_block,
        std::shared_ptr<custom_buf_loopback>>(m, "custom_buf_loopback", D(custom_buf_loopback))

        .def(py::init(&custom_buf_loopback::make),
           py::arg("batch_size"),
           D(custom_buf_loopback,make)
        )
        



        ;




}








