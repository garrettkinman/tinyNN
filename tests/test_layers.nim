# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import tinyNN

let a = Dense[int8].new([2, 3], (1.int8)..(1.int8), relu)

test "dense":
    # TODO: add tests
    check a == Dense[int8].new([2, 3], (1.int8)..(1.int8), relu)