# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sugar
import ../tensors

type
    Conv2D*[T] = object
        weights*: Tensor[T]
        bias*: Tensor[T]
        activation*: (Tensor[T]) -> Tensor[T]
        stride*: int
        padding*: int

proc newConv2D*[T](kernel_size, in_channels, out_channels: int, slice: HSlice[T, T], stride, padding: int, activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] =
    result = Conv2D[T](
        weights: Tensor[T].rand([out_channels, in_channels, kernel_size, kernel_size], slice),
        bias: Tensor[T].rand([out_channels, 1], slice),
        activation: activation,
        stride: stride,
        padding: padding
    )

proc new*[T](_: typedesc[Conv2D[T]], kernel_size, in_channels, out_channels: int, slice: HSlice[T, T], stride, padding: int, activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] = newConv2D[T](kernel_size, in_channels, out_channels, slice, stride, padding, activation)

proc forward*[T](self: Conv2D, x: Tensor[T]): Tensor[T] =
    # Perform 2D convolution
    # This is a simplified version and does not handle padding or stride
    # TODO: handle padding + stride
    result = Tensor[T].zeros([self.weights.shape[0], x.shape[1], x.shape[2]])
    for out_channel in 0..<self.weights.shape[0]:
        for in_channel in 0..<self.weights.shape[1]:
            for i in 0..<x.shape[1]:
                for j in 0..<x.shape[2]:
                    for m in 0..<self.weights.shape[2]:
                        for n in 0..<self.weights.shape[3]:
                            result[out_channel, i, j] += self.weights[out_channel, in_channel, m, n] * x[in_channel, i + m, j + n]
    result += self.bias
    result = self.activation(result)
