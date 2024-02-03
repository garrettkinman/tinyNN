# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import tinyNN

var
    a = Tensor.new([2, 3], @[float32 -3, -2, -1, 1, 2, 3])
    b = Tensor.new([3, 2], @[uint8 0, 1, 127, 128, 254, 255])
    c = Tensor.new([2, 2], @[int8 -128, -1, 1, 127])

test "sigmoid":
    a.sigmoid()
    check a == Tensor.new([2, 3], @[float32 0.04742587357759476, 0.1192029193043709, 0.2689414322376251, 0.7310585975646973, 0.8807970285415649, 0.9525741338729858])
    b.sigmoid()
    check b == Tensor.new([3, 2], @[uint8 69, 69, 127, 128, 186, 186])
    c.sigmoid()
    check c == Tensor.new([2, 2], @[int8 34, 64, 64, 93])