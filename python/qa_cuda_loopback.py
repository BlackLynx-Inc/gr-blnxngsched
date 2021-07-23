#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2021 BlackLynx Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import filecmp
import os
import tempfile

from gnuradio import gr, gr_unittest
from gnuradio import blocks
try:
    from blnxngsched import cuda_loopback
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from blnxngsched import cuda_loopback

class qa_cuda_loopback(gr_unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_file_1MB = tempfile.NamedTemporaryFile(suffix='.bin')
        cls.test_file_16MB = tempfile.NamedTemporaryFile(suffix='.bin')
        cls.test_file_64MB = tempfile.NamedTemporaryFile(suffix='.bin')
        
        one_MB = 2**20
        cls.test_file_1MB.write(os.urandom(one_MB))
        cls.test_file_16MB.write(os.urandom(16 * one_MB))
        cls.test_file_64MB.write(os.urandom(64 * one_MB))

    @classmethod
    def tearDownClass(cls):
        cls.test_file_1MB = None
        cls.test_file_16MB = None
        cls.test_file_64MB = None

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None
        
    def _simple_loopback_helper(self, input_file, batch_size, num_loopbacks=1):
        """ Helper method to run simple loopback cases on the passed input file
        and batch size combination.
        """
        output_file = tempfile.NamedTemporaryFile(suffix='.bin')
        
        source = blocks.file_source(gr.sizeof_gr_complex, input_file, False, 0, 0)
        sink = blocks.file_sink(gr.sizeof_gr_complex, output_file.name, False)
        sink.set_unbuffered(False)
        
        # Create the loopbacks
        loopbacks = []
        for idx in range(num_loopbacks):
            loopbacks.append(cuda_loopback(batch_size, 1))
        last_idx = num_loopbacks - 1

        # Hook everything together
        self.tb.connect(source, loopbacks[0])
        for idx in range(1, num_loopbacks):
            self.tb.connect(loopbacks[idx - 1], loopbacks[idx])
        self.tb.connect(loopbacks[last_idx], sink)

        # Run the flowgraph then verify that the utput file match
        self.tb.run()
        self.assertTrue(filecmp.cmp(input_file, output_file.name, shallow=False))
        
    def _multi_loopback_helper(self, input_file, batch_sizes, num_loopbacks):
        """ Helper method to run multi loopback cases on the passed input file
        and using the passed batch sizes for the corresponding loopback blocks.
        """
        self.assertEqual(len(batch_sizes), num_loopbacks) 
        
        output_file = tempfile.NamedTemporaryFile(suffix='.bin')
        
        source = blocks.file_source(gr.sizeof_gr_complex, input_file, False, 0, 0)
        sink = blocks.file_sink(gr.sizeof_gr_complex, output_file.name, False)
        sink.set_unbuffered(False)
        
        # Create the loopbacks
        loopbacks = []
        for idx in range(num_loopbacks):
            loopbacks.append(cuda_loopback(batch_sizes[idx], 1))
        last_idx = num_loopbacks - 1

        # Hook everything together
        self.tb.connect(source, loopbacks[0])
        for idx in range(1, num_loopbacks):
            self.tb.connect(loopbacks[idx - 1], loopbacks[idx])
        self.tb.connect(loopbacks[last_idx], sink)

        # Run the flowgraph then verify that the utput file match
        self.tb.run()
        self.assertTrue(filecmp.cmp(input_file, output_file.name, shallow=False))

    def test_001_loopback_small_bs4096_lb1(self):
        """ Run loopback test with a small (1MB) input file, 4096 batch size, 
        and 1 loopback block.
        """
        self._simple_loopback_helper(self.test_file_1MB.name, 4096, 1)
    
    def test_002_loopback_small_bs65536_lb1(self):
        """ Run loopback test with a small (1MB) input file, 65536 batch size, 
        and 1 loopback block.
        """
        self._simple_loopback_helper(self.test_file_1MB.name, 65536, 1)
    
    def test_003_loopback_medium_bs4096_lb1(self):
        """ Run simple loopback test with a medium (16MB) input file, 4096 
        batch size, and 1 loopback block.
        """
        self._simple_loopback_helper(self.test_file_16MB.name, 4096, 1)
    
    def test_004_loopback_medium_bs65536_lb1(self):
        """ Run simple loopback test with a medium (16MB) input file, 65536 
        batch size, and 1 loopback block.
        """
        self._simple_loopback_helper(self.test_file_16MB.name, 65536, 1)
        
    def test_005_loopback_large_bs4096_lb1(self):
        """ Run simple loopback test with a large (64MB) input file, 4096 
        batch size, and 1 loopback block.
        """
        self._simple_loopback_helper(self.test_file_64MB.name, 4096, 1)
    
    def test_006_loopback_large_bs65536_lb1(self):
        """ Run simple loopback test with a large (64MB) input file, 65536
        batch size, and 1 loopback block.
        """
        self._simple_loopback_helper(self.test_file_64MB.name, 65536, 1)
        
    def test_007_loopback_small_bs4096_lb2(self):
        """ Run loopback test with a small (1MB) input file, 4096 batch size, 
        and 2 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_1MB.name, 4096, 2)
    
    def test_008_loopback_small_bs65536_lb2(self):
        """ Run loopback test with a small (1MB) input file, 65536 batch size, 
        and 2 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_1MB.name, 65536, 2)
    
    def test_009_loopback_medium_bs4096_lb2(self):
        """ Run simple loopback test with a medium (16MB) input file, 4096 
        batch size, and 2 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_16MB.name, 4096, 2)
    
    def test_010_loopback_medium_bs65536_lb2(self):
        """ Run simple loopback test with a medium (16MB) input file, 65536 
        batch size, and 2 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_16MB.name, 65536, 2)
        
    def test_011_loopback_large_bs4096_lb2(self):
        """ Run simple loopback test with a large (64MB) input file, 4096 
        batch size, and 2 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_64MB.name, 4096, 2)
    
    def test_012_loopback_large_bs65536_lb2(self):
        """ Run simple loopback test with a large (64MB) input file, 65536
        batch size, and 2 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_64MB.name, 65536, 2)
        
    def test_013_loopback_small_bs4096_lb16(self):
        """ Run loopback test with a small (1MB) input file, 4096 batch size, 
        and 16 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_1MB.name, 4096, 16)
    
    def test_014_loopback_small_bs65536_lb16(self):
        """ Run loopback test with a small (1MB) input file, 65536 batch size, 
        and 16 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_1MB.name, 65536, 16)
    
    def test_015_loopback_medium_bs4096_lb16(self):
        """ Run simple loopback test with a medium (16MB) input file, 4096 
        batch size, and 16 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_16MB.name, 4096, 16)
    
    def test_016_loopback_medium_bs65536_lb16(self):
        """ Run simple loopback test with a medium (16MB) input file, 65536 
        batch size, and 16 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_16MB.name, 65536, 16)
        
    def test_017_loopback_large_bs4096_lb16(self):
        """ Run simple loopback test with a large (64MB) input file, 4096 
        batch size, and 16 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_64MB.name, 4096, 16)
    
    def test_018_loopback_large_bs65536_lb16(self):
        """ Run simple loopback test with a large (64MB) input file, 65536
        batch size, and 16 loopback blocks.
        """
        self._simple_loopback_helper(self.test_file_64MB.name, 65536, 16)
        
    def test_019_loopback_multi_batch_size(self):
        """ Run loopback test with a small (1MB) input file and 4 loopback 
        blocks with various batch sizes.
        """
        batch_sizes = [1024, 2048, 4096, 16384]
        self._multi_loopback_helper(self.test_file_1MB.name, batch_sizes, 4)
        
    def test_020_loopback_multi_batch_size(self):
        """ Run loopback test with a large (64MB) input file and 4 loopback 
        blocks with various batch sizes.
        """
        batch_sizes = [1024, 2048, 4096, 16384]
        self._multi_loopback_helper(self.test_file_64MB.name, batch_sizes, 4)

    def test_021_loopback_multi_batch_size(self):
        """ Run loopback test with a small (1MB) input file and 4 loopback 
        blocks with various batch sizes.
        """
        batch_sizes = [16384, 4096, 2048, 1024]
        self._multi_loopback_helper(self.test_file_1MB.name, batch_sizes, 4)
        
    def test_022_loopback_multi_batch_size(self):
        """ Run loopback test with a large (64MB) input file and 4 loopback 
        blocks with various batch sizes.
        """
        batch_sizes = [16384, 4096, 2048, 1024]
        self._multi_loopback_helper(self.test_file_64MB.name, batch_sizes, 4)
        
    def test_023_loopback_multi_batch_size(self):
        """ Run loopback test with a small (1MB) input file and 4 loopback 
        blocks with various batch sizes.
        """
        batch_sizes = [1024, 16384, 2048, 4096]
        self._multi_loopback_helper(self.test_file_1MB.name, batch_sizes, 4)
        
    def test_024_loopback_multi_batch_size(self):
        """ Run loopback test with a large (64MB) input file and 4 loopback 
        blocks with various batch sizes.
        """
        batch_sizes = [1024, 16384, 2048, 4096]
        self._multi_loopback_helper(self.test_file_64MB.name, batch_sizes, 4)

        
if __name__ == '__main__':
    gr_unittest.run(qa_cuda_loopback)
