# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

using Flux.NNlib

function σ(x::UInt8)
    # scale to [-1, 1]
    x_scaled = ((float(x) / 255.0) * 2.0) - 1.0
    y = NNlib.σ(x_scaled)
    # y = 1.0 / (1.0 + exp(-x_scaled))
    return round(y * 255) |> UInt8
end

lookup_table = σ.(UInt8.(0:255))