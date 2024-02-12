# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

using Flux.NNlib

function σ(x::Int8)
    # scale to [-1, 1]
    x_scaled = (float(x) / 255.0) * 2.0
    y = NNlib.σ(x_scaled)
    return round(y * 255 / 2) |> Int8
end

lookup_table_uint8 = σ.(UInt8.(0:255))
lookup_table_int8 = σ.(Int8.(-128:127))

