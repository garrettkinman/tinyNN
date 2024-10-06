<!--
 Copyright (c) 2023 Garrett Kinman
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# tinyNN
Lightweight, performant, and dependency-free neural network inference engine for TinyML written in pure Nim. Portable to custom accelerators and WASM.

# Planned Features
- Data types
  - `int8`
  - `float32`
  - More? (TBD)
- Tensor types
  - Regular (dynamically allocated)
  - Sparse
  - Statically allocated? (TBD)
- Standard layers
  - Dense
  - Conv2D, DepthwiseConv
  - Pooling layers
  - Output layers, e.g., softmax
  - Recurrent layers? (TBD)
  - Attention layers? (TBD)
  - More? (TBD)
- Standard activation functions
  - sigmoid
  - relu
  - tanh
  - More? (TBD)
- Statically-allocated model parameters
  - Maybe have the entire library itself have no dynamic allocations?
- Optimized CPU operations (so no BLAS dependencies)
- Built-in hardware acceleration support
  - RISC-V V extension (vector)
  - RISC-V P extension (packed SIMD)
  - More? (TBD)
- Ability to (relatively) easily accelerate on other hardware
  - Expose layers as some combination of tensor operations so that accelerating becomes a matter of accelerating the tensor operations
  - Treat activation functions as tensor operations so that they, too, can be accelerated
- Ability to easily port to WASM
- Ability to load pre-trained parameters from a standard `.tflite` file
- On-device learning? (TBD)
- Forward-only learning? (TBD)
  - Forward-Forward
  - PEPITA
  - MEMPITA
