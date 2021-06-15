# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT blnxngsched_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT blnxngsched_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND blnxngsched_python_files
    cuda_loopback_python.cc 
    custom_buf_loopback_python.cc
    python_bindings.cc)

GR_PYBIND_MAKE_OOT(blnxngsched 
   ../..
   gr::blnxngsched
   "${blnxngsched_python_files}")

install(TARGETS blnxngsched_python DESTINATION ${GR_PYTHON_DIR}/blnxngsched COMPONENT pythonapi)
