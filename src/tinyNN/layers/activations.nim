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

import .. / constants / [lookup_tables]

func sigmoid(x: int8): int8 =
    result = sigmoid_i8[x]

func sigmoid(x: float32): float32 =
    result = 1 / (1 + exp(-x))

proc sigmoid*[T](tensor: var Tensor[T]): void =
    ## A generic sequential implementation of the sigmoid activation function
    for it in 0..<tensor.len:
        let x: T = tensor.data[it]
        let y: T = sigmoid(x)
        tensor.data[it] = y

func relu(x: int8): int8 =
    if x > 0:
        result = x
    else:
        result = 0

func relu(x: float32): float32 =
    if x > 0:
        result = x
    else:
        result = 0

proc relu*[T](tensor: var Tensor[T]): void =
    ## A generic sequential implementation of the ReLU activation function
    for it in 0..<tensor.len:
        let x: T = tensor.data[it]
        let y: T = relu(x)
        tensor.data[it] = y