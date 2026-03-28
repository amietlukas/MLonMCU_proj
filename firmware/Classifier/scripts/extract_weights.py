#!/usr/bin/env python3
"""
Extract INT8 weights from ONNX QDQ model and generate C source files
for CMSIS-NN inference on STM32.

Usage:
    python extract_weights.py <onnx_model_path> <output_dir>
"""

import sys
import os
import numpy as np
import onnx
from onnx import numpy_helper


def get_initializer(model, name):
    """Get a named initializer as numpy array."""
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"Initializer '{name}' not found")


def quantize_multiplier(real_multiplier):
    """Convert float multiplier to (int32 multiplier, int shift) for CMSIS-NN.

    CMSIS-NN convention: real_multiplier = multiplier * 2^(shift - 31)
    where multiplier is in [2^30, 2^31 - 1] (i.e., normalized to [0.5, 1.0) in Q31).
    """
    if real_multiplier == 0.0:
        return 0, 0

    shift = 0
    q = real_multiplier

    # Normalize q into [0.5, 1.0)
    while q < 0.5:
        q *= 2.0
        shift -= 1
    while q >= 1.0:
        q /= 2.0
        shift += 1

    # Convert to Q31 fixed-point
    q_fixed = int(round(q * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1

    return q_fixed, shift


def format_int8_array(arr, name, per_line=16):
    """Format int8 array as C source."""
    flat = arr.flatten()
    lines = [f"const int8_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        vals = ", ".join(f"{v}" for v in chunk)
        lines.append(f"    {vals},")
    lines.append("};")
    return "\n".join(lines)


def format_int32_array(arr, name, per_line=8):
    """Format int32 array as C source."""
    flat = arr.flatten()
    lines = [f"const int32_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        vals = ", ".join(f"{v}" for v in chunk)
        lines.append(f"    {vals},")
    lines.append("};")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <onnx_model> <output_dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    model = onnx.load(model_path)
    print(f"Loaded model: {model_path}")

    # =========================================================================
    # Extract quantization parameters for activations
    # =========================================================================
    input_scale = float(get_initializer(model, "input_scale"))
    input_zp = int(get_initializer(model, "input_zero_point"))

    conv0_out_scale = float(get_initializer(model, "/features/features.0/features.0.2/Relu_output_0_scale"))
    conv0_out_zp = int(get_initializer(model, "/features/features.0/features.0.2/Relu_output_0_zero_point"))

    conv1_out_scale = float(get_initializer(model, "/features/features.1/features.1.2/Relu_output_0_scale"))
    conv1_out_zp = int(get_initializer(model, "/features/features.1/features.1.2/Relu_output_0_zero_point"))

    conv2_out_scale = float(get_initializer(model, "/features/features.2/features.2.2/Relu_output_0_scale"))
    conv2_out_zp = int(get_initializer(model, "/features/features.2/features.2.2/Relu_output_0_zero_point"))

    conv3_out_scale = float(get_initializer(model, "/features/features.3/features.3.2/Relu_output_0_scale"))
    conv3_out_zp = int(get_initializer(model, "/features/features.3/features.3.2/Relu_output_0_zero_point"))

    reducemean_out_scale = float(get_initializer(model, "/ReduceMean_output_0_scale"))
    reducemean_out_zp = int(get_initializer(model, "/ReduceMean_output_0_zero_point"))

    logits_scale = float(get_initializer(model, "logits_scale"))
    logits_zp = int(get_initializer(model, "logits_zero_point"))

    print(f"\n=== Activation quantization ===")
    print(f"  Input:       scale={input_scale:.10f}, zp={input_zp}")
    print(f"  Conv0 out:   scale={conv0_out_scale:.10f}, zp={conv0_out_zp}")
    print(f"  Conv1 out:   scale={conv1_out_scale:.10f}, zp={conv1_out_zp}")
    print(f"  Conv2 out:   scale={conv2_out_scale:.10f}, zp={conv2_out_zp}")
    print(f"  Conv3 out:   scale={conv3_out_scale:.10f}, zp={conv3_out_zp}")
    print(f"  ReduceMean:  scale={reducemean_out_scale:.10f}, zp={reducemean_out_zp}")
    print(f"  Logits out:  scale={logits_scale:.10f}, zp={logits_zp}")

    # =========================================================================
    # Extract weights, biases, and compute requantization params
    # =========================================================================
    layers = [
        {
            "name": "conv0",
            "weight_key": "onnx::Conv_46_quantized",
            "weight_scale_key": "onnx::Conv_46_scale",
            "bias_key": "onnx::Conv_47_quantized",
            "input_scale": input_scale,
            "output_scale": conv0_out_scale,
            "is_conv": True,
        },
        {
            "name": "conv1",
            "weight_key": "onnx::Conv_49_quantized",
            "weight_scale_key": "onnx::Conv_49_scale",
            "bias_key": "onnx::Conv_50_quantized",
            "input_scale": conv0_out_scale,  # MaxPool preserves scale
            "output_scale": conv1_out_scale,
            "is_conv": True,
        },
        {
            "name": "conv2",
            "weight_key": "onnx::Conv_52_quantized",
            "weight_scale_key": "onnx::Conv_52_scale",
            "bias_key": "onnx::Conv_53_quantized",
            "input_scale": conv1_out_scale,
            "output_scale": conv2_out_scale,
            "is_conv": True,
        },
        {
            "name": "conv3",
            "weight_key": "onnx::Conv_55_quantized",
            "weight_scale_key": "onnx::Conv_55_scale",
            "bias_key": "onnx::Conv_56_quantized",
            "input_scale": conv2_out_scale,
            "output_scale": conv3_out_scale,
            "is_conv": True,
        },
        {
            "name": "fc",
            "weight_key": "classifier.weight_quantized",
            "weight_scale_key": "classifier.weight_scale",
            "bias_key": "classifier.bias_quantized",
            # AvgPool output uses ReduceMean quantization from ONNX
            "input_scale": reducemean_out_scale,
            "output_scale": logits_scale,
            "is_conv": False,
        },
    ]

    c_arrays = []
    h_externs = []

    for layer in layers:
        name = layer["name"]
        print(f"\n--- {name} ---")

        # Get weights
        weights = get_initializer(model, layer["weight_key"])
        weight_scales = get_initializer(model, layer["weight_scale_key"])
        bias = get_initializer(model, layer["bias_key"]).copy()

        print(f"  Weights: shape={weights.shape}, dtype={weights.dtype}")
        print(f"  Weight scales: shape={weight_scales.shape}")
        print(f"  Bias: shape={bias.shape}, dtype={bias.dtype}")

        if layer["is_conv"]:
            # ONNX conv weights: (out_ch, in_ch, kH, kW) = OIHW
            # CMSIS-NN expects: (out_ch, kH, kW, in_ch) = OHWI
            weights_ohwi = weights.transpose(0, 2, 3, 1).copy()
            print(f"  Transposed OIHW→OHWI: {weights.shape} → {weights_ohwi.shape}")
            w_flat = weights_ohwi.flatten().astype(np.int8)
        else:
            # FC weights: ONNX Gemm stores (out_features, in_features)
            # CMSIS-NN fully_connected_s8 expects same: (out_features, in_features)
            w_flat = weights.flatten().astype(np.int8)

        # Compute per-channel requantization multiplier and shift
        in_scale = layer["input_scale"]
        out_scale = layer["output_scale"]
        num_channels = len(weight_scales)

        multipliers = np.zeros(num_channels, dtype=np.int32)
        shifts = np.zeros(num_channels, dtype=np.int32)

        # Detect dead channels (all-zero weights) to avoid overflow
        dead_channels = 0
        for ch in range(num_channels):
            ch_weights = weights[ch]  # per output channel
            if np.all(ch_weights == 0):
                # Dead channel: force mult=0, shift=0
                # Output will be 0 + output_zp = -128 (ReLU zero)
                multipliers[ch] = 0
                shifts[ch] = 0
                bias[ch] = 0  # Zero out garbage bias for dead channels
                dead_channels += 1
            else:
                eff_scale = (in_scale * float(weight_scales[ch])) / out_scale
                m, s = quantize_multiplier(eff_scale)
                multipliers[ch] = m
                shifts[ch] = s

        if dead_channels > 0:
            print(f"  WARNING: {dead_channels}/{num_channels} dead channels (zeroed)")

        active_scales = [(in_scale * float(weight_scales[ch])) / out_scale
                         for ch in range(num_channels) if not np.all(weights[ch] == 0)]
        if active_scales:
            print(f"  Active effective scales: [{min(active_scales):.8f}, {max(active_scales):.8f}]")

        # Generate C arrays
        c_arrays.append(format_int8_array(w_flat, f"{name}_weights"))
        c_arrays.append(format_int32_array(bias, f"{name}_bias"))
        c_arrays.append(format_int32_array(multipliers, f"{name}_output_mult"))
        c_arrays.append(format_int32_array(shifts, f"{name}_output_shift"))

        h_externs.append(f"extern const int8_t {name}_weights[{len(w_flat)}];")
        h_externs.append(f"extern const int32_t {name}_bias[{num_channels}];")
        h_externs.append(f"extern const int32_t {name}_output_mult[{num_channels}];")
        h_externs.append(f"extern const int32_t {name}_output_shift[{num_channels}];")
        h_externs.append("")

    # =========================================================================
    # Write header file
    # =========================================================================
    header = f"""#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

/* ======================================================================
 * Auto-generated from ONNX model by extract_weights.py
 * Model: {os.path.basename(model_path)}
 * ====================================================================== */

/* Input quantization */
#define MODEL_INPUT_SCALE    {input_scale:.10f}f
#define MODEL_INPUT_ZP       ({input_zp})

/* Output quantization */
#define MODEL_OUTPUT_SCALE   {logits_scale:.10f}f
#define MODEL_OUTPUT_ZP      ({logits_zp})

/* Per-layer activation quantization (scale, zero_point) */
#define CONV0_OUT_SCALE      {conv0_out_scale:.10f}f
#define CONV0_OUT_ZP         ({conv0_out_zp})
#define CONV1_OUT_SCALE      {conv1_out_scale:.10f}f
#define CONV1_OUT_ZP         ({conv1_out_zp})
#define CONV2_OUT_SCALE      {conv2_out_scale:.10f}f
#define CONV2_OUT_ZP         ({conv2_out_zp})
#define CONV3_OUT_SCALE      {conv3_out_scale:.10f}f
#define CONV3_OUT_ZP         ({conv3_out_zp})
#define AVGPOOL_OUT_SCALE    {reducemean_out_scale:.10f}f
#define AVGPOOL_OUT_ZP       ({reducemean_out_zp})

/* Weight and quantization parameter arrays */
{chr(10).join(h_externs)}
#endif /* MODEL_WEIGHTS_H */
"""

    header_path = os.path.join(output_dir, "model_weights.h")
    with open(header_path, "w") as f:
        f.write(header)
    print(f"\nWrote header: {header_path}")

    # =========================================================================
    # Write source file
    # =========================================================================
    source = f"""/* Auto-generated from ONNX model by extract_weights.py
 * Model: {os.path.basename(model_path)}
 * DO NOT EDIT - regenerate with extract_weights.py
 */

#include "model_weights.h"

{chr(10).join(c_arrays)}
"""

    source_path = os.path.join(output_dir, "model_weights.c")
    with open(source_path, "w") as f:
        f.write(source)
    print(f"Wrote source: {source_path}")

    # Print summary
    total_bytes = sum(
        get_initializer(model, l["weight_key"]).nbytes +
        get_initializer(model, l["bias_key"]).nbytes
        for l in layers
    )
    print(f"\nTotal weight+bias data: {total_bytes} bytes ({total_bytes/1024:.1f} KB)")


if __name__ == "__main__":
    main()
