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
    d = Tensor.new([2, 2], @[int 1, 2, 3, 4])
    e = Tensor.new([1, 2, 3], @[int 1, 2, 3, 4, 5, 6])

test "sigmoid":
    a.sigmoid()
    check a == Tensor.new([2, 3], @[float32 0.04742587357759476, 0.1192029193043709, 0.2689414322376251, 0.7310585975646973, 0.8807970285415649, 0.9525741338729858])
    b.sigmoid()
    check b == Tensor.new([3, 2], @[uint8 69, 69, 127, 128, 186, 186])
    # check a == a
    # check a != b
    # check a != b.transpose()
    # check c == c
    # check c != a
    # check c != b
    # check c != d
    # check c.shape == d.shape
    # check c.shape != a.shape
    # check a.shape != b.shape
    # check a.shape == b.transpose().shape