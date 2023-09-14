# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
import ../tensors

# TODO: add common activation functions for sequential operation on the cpu
# 1. sigmoid
# 2. relu
# 3. tanh
# and more

func sigmoid(x: uint8): uint8 =
    # TODO: look-up table
    result = x

func sigmoid(x: int8): int8 =
    # TODO: look-up table
    result = x

func sigmoid(x: float32): float32 =
    result = 1 / (1 + exp(-x))

# TODO: test to make sure this is correct
proc sigmoid*[T](tensor: var Tensor[T]): void =
    ## A generic sequential implementation of the sigmoid activation function
    for it in 0..<tensor.len:
        let x: T = tensor.data[it]
        let y: T = sigmoid(x)
        tensor.data[it] = y
    
    