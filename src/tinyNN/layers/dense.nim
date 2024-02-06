# Copyright (c) 2023 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sugar
import ../tensors

type
    Dense*[T] = object
        # TODO: make these statically allocated on the stack for performance + memory safety
        weights*: Tensor[T]
        bias*: Tensor[T]
        activation*: (Tensor[T]) -> Tensor[T]

# TODO: fix this to work and look nice
proc newDense*[T](shape: openArray[int], activation: (Tensor[T]) -> Tensor[T]): Dense[T] =
    Dense[T](
        weights: rand[T](shape),
        bias: rand[T](shape[0]), # make sure this is the right shape
        activation: activation
    )

# TODO: fix this to work and look nice
proc newDense*[T](shape: openArray[int], weights: seq[T], bias: seq[T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] =
    Dense[T](
        weights: rand[T](shape),
        bias: rand[T](shape[0]), # make sure this is the right shape
        activation: activation
    )

proc new*[T](_: typedesc[Dense[T]], shape: openArray[int], activation: (Tensor[T]) -> void): Dense[T] = newDense[T](shape)
proc new*[T](_: typedesc[Tensor[T]], shape: openArray[int], weights: seq[T], bias:seq[T], activation: (Tensor[T]) -> Tensor[T]): Dense[T] = newDense(shape, weights, bias, activation)
# proc new*[T](_: typedesc[Tensor[T]], shape: openArray[int], value: T): Tensor[T] = newTensor(shape, value)

proc forward*[T](self: Dense, x: Tensor[T]): Tensor[T] =
    result = ((self.weights * x) + self.bias).activation()