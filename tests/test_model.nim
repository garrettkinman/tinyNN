# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import tinyNN

type
    SingleLayerPerceptronModel[T] = object
        dense: Dense[int8]

    MultiLayerPerceptronModel[T] = object
        dense1: Dense[int8]
        dense2: Dense[int8]

proc forward[T](model: SingleLayerPerceptronModel[T], x: Tensor[T]): Tensor[T] =
    result = model.dense.forward(x)

proc forward[T](model: MultiLayerPerceptronModel[T], x: Tensor[T]): Tensor[T] =
    result = model.dense2.forward(model.dense1.forward(x))

let
    slp = SingleLayerPerceptronModel[int8](
        dense: Dense[int8].new(
            [1, 3],
            (1.int8)..(1.int8),
            relu
        )
    )
    mlp = MultiLayerPerceptronModel[int8](
        dense1: Dense[int8].new(
            [3, 3],
            (1.int8)..(1.int8),
            relu
        ),
        dense2: Dense[int8].new(
            [1, 3],
            (1.int8)..(1.int8),
            relu
        )
    )

test "singleLayerPerceptron":
    echo slp.forward(Tensor.new([3, 1], @[int8 1, 2, 3]))
    echo mlp.forward(Tensor.new([3, 1], @[int8 1, 2, 3]))