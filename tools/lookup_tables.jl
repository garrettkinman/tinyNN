# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

using Flux.NNlib

function get_quant_params(α, β)
    scale = (β - α) / 255
    zero_point = round(((-128 * 3.0) - (127 * 0.0)) / 255)
    return scale, zero_point

end

function generate_lut(f::Function, scale_in::Float32, zero_point_in::Int8; scale_out::Float32=Float32(1/255), zero_point_out::Int8=Int8(-128))
    x_quant = Int8.(-128:127)
    x_dequant = Float32.(Int32.(x_quant) .- Int32(zero_point_in)) .* scale_in
    y = f.(x_dequant)
    y_quant = Int32.(round.(y ./ scale_out)) .+ Int32(zero_point_out)
    lut = Int8.(clamp.((y_quant), -128, 127))
    return lut
end

lut_σ₁ = generate_lut(NNlib.σ, Float32(0.00392156885968563), Int8(-128))
lut_σ₂ = generate_lut(NNlib.σ, Float32(1/255), Int8(-128))

lut_exp₁ = generate_lut(NNlib.exp, Float32(0.00392156885968563), Int8(-128))
lut_exp₂ = generate_lut(NNlib.exp, Float32(3/255), Int8(-2))
