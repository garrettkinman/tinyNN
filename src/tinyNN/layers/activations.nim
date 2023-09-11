# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import ../tensors

# TODO: add common activation functions for sequential operation on the cpu
# 1. sigmoid
# 2. relu
# 3. tanh
# and more

# TODO: test to make sure this is correct
proc sigmoid*[T](tensor: Tensor[T]): void =
    ## A generic sequential implementation of the sigmoid activation function
    for it in 0..<tensor.len:
        let x: T = tensor.data[it]
        let y: T = 1 / (1 + exp(-x))
        tensor.data[it] = y
    
    