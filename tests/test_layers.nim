# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import tinyNN

let
    a = Dense[int8].new([2, 3], (1.int8)..(1.int8), relu)
    b = Dense[int8].new(
        Tensor.new([2, 2], @[int8 1, 0, 0, 1]),
        Tensor.new([2, 1], @[int8 0, 0]),
        relu
    )
    c = Dense[int8].new(
        [2, 2],
        @[int8 1, 0, 0, 1],
        @[int8 0, 0],
        relu
    )

test "dense":
    check a == Dense[int8].new(
        [2, 3],
        (1.int8)..(1.int8),
        relu
    )
    check a.forward(Tensor.new([3, 1], @[int8 -1, 0, 1])) == Tensor.new([2, 1], @[int8 1, 1])
    check b == Dense[int8].new(
        Tensor.new([2, 2], @[int8 1, 0, 0, 1]),
        Tensor.new([2, 1], @[int8 0, 0]),
        relu
    )
    check b.forward(Tensor.new([2, 1], @[int8 1, 1])) == Tensor.new([2, 1], @[int8 1, 1])
    check c == Dense[int8].new(
        [2, 2],
        @[int8 1, 0, 0, 1],
        @[int8 0, 0],
        relu
    )
    check c.forward(Tensor.new([2, 1], @[int8 1, 1])) == Tensor.new([2, 1], @[int8 1, 1])
