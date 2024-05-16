# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
import ../tensors

func exp(x: int8): int8 =
    # TODO: Implement quantized exp
    result = 0

func exp(x: float32): float32 =
    result = math.exp(x)

proc exp(tensor: Tensor[int8]): Tensor[int8] =
    result = Tensor[int8].new(tensor.shape)
    for it in 0..<tensor.len:
        let x: int8 = tensor.data[it]
        let y: int8 = exp(x)
        result.data[it] = y

proc exp(tensor: Tensor[float32]): Tensor[float32] =
    result = Tensor[float32].new(tensor.shape)
    for it in 0..<tensor.len:
        let x: float32 = tensor.data[it]
        let y: float32 = exp(x)
        result.data[it] = y

proc softmax*(tensor: Tensor[int8]): Tensor[int8] =
    # TODO: Implement quantized softmax
    result = Tensor[int8].new(tensor.shape)

proc softmax*(tensor: Tensor[float32]): Tensor[float32] =
    result = exp(tensor)
    for it in 0..<tensor.len:
        result.data[it] = result.data[it] / result.sum()