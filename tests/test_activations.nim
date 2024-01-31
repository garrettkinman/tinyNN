# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import tinyNN

var
    a = Tensor.new([2, 3], @[float32 1, 2, 3, 4, 5, 6])
    b = Tensor.new([3, 2], @[int 1, 2, 3, 4, 5, 6])
    c = Tensor.new([2, 2], @[int 22, 28, 49, 64])
    d = Tensor.new([2, 2], @[int 1, 2, 3, 4])
    e = Tensor.new([1, 2, 3], @[int 1, 2, 3, 4, 5, 6])

test "sigmoid":
    a.sigmoid();
    echo a;
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