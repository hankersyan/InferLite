#!/usr/bin/env python
"""Generate a small OpenVINO IR sample model for testing InferLite.

Creates models/sample_model/1/{model.xml, model.bin} and config.pbtxt.
The model computes y = W * x + b for input shape [1, 4] and output shape [1, 2].
"""
import argparse
import os
import sys

import numpy as np
import openvino as ov
from openvino.runtime import opset8 as ops


def build_model():
    # y = x * 2 + 1, applied element-wise over a [1, 4] input.
    x = ops.parameter([1, 4], ov.Type.f32, name="input")
    const = ops.constant(np.array([2.0], dtype=np.float32), name="scale")
    mul = ops.multiply(x, const, name="mul")
    bias = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    add = ops.add(mul, bias, name="add")
    model = ov.Model([add], [x], "sample_model")
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate a sample OpenVINO model.")
    parser.add_argument("--out", default="models/sample_model", help="Output model dir")
    args = parser.parse_args()

    version_dir = os.path.join(args.out, "1")
    os.makedirs(version_dir, exist_ok=True)

    model = build_model()
    # Serialize to IR in the version directory.
    ov.save_model(model, os.path.join(version_dir, "model.xml"))

    # Write config.pbtxt for InferLite.
    cfg = """name: "sample_model"
backend: "openvino"
max_batch_size: 0
input {
  name: "input"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}
output {
  name: "add"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}
instance_group {
  count: 2
  kind: KIND_CPU
}
"""
    with open(os.path.join(args.out, "config.pbtxt"), "w") as f:
        f.write(cfg)

    print("Sample model written to", os.path.abspath(args.out))
    print("  version dir :", os.path.abspath(version_dir))


if __name__ == "__main__":
    main()
