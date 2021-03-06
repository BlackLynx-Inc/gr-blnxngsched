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
/* BINDTOOL_HEADER_FILE(cuda_loopback.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(3112d000ab3550d573de6523e77f1900)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <blnxngsched/cuda_loopback.h>
// pydoc.h is automatically generated in the build directory
#include <cuda_loopback_pydoc.h>

void bind_cuda_loopback(py::module& m)
{

    using cuda_loopback    = gr::blnxngsched::cuda_loopback;


    py::class_<cuda_loopback, gr::block, gr::basic_block,
        std::shared_ptr<cuda_loopback>>(m, "cuda_loopback", D(cuda_loopback))

        .def(py::init(&cuda_loopback::make),
           py::arg("batch_size"),
           py::arg("load"),
           D(cuda_loopback,make)
        )
        



        ;




}








