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
        activation*: (Tensor[T]) -> void

proc forward*[T](layer: Dense, x: Tensor[T]): Tensor[T] =
    result = ((layer.weights * x) + layer.bias).activation()