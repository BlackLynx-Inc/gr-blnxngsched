#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-testmult author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
try:
    from blnxngsched import cuda_fanout
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from blnxngsched import cuda_fanout

class qa_cuda_fanout(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_descriptive_test_name(self):
        # set up fg
        nitems = 100000
        data = [complex(x,-x) for x in range(nitems)]
        vsrc = blocks.vector_source_c(data, repeat=False)
        vsnk_1 = blocks.vector_sink_c()
        vsnk_2 = blocks.vector_sink_c()
        vsnk_3 = blocks.vector_sink_f()

        op = cuda_fanout()

        self.tb.connect(vsrc, op)
        self.tb.connect((op, 0), vsnk_1)
        self.tb.connect((op, 1), vsnk_2)
        self.tb.connect((op, 2), vsnk_3)

        self.tb.run()
        
        # check data
        self.assertEqual(len(vsnk_1.data()), nitems + 63)
        self.assertEqual(len(vsnk_2.data()), nitems + 63)
        self.assertEqual(len(vsnk_3.data()), nitems + 63)


if __name__ == '__main__':
    gr_unittest.run(qa_cuda_fanout)
