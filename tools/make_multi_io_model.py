#!/usr/bin/env python
"""Generate a multi-input / multi-output OpenVINO IR sample model for testing
InferLite's Triton array-of-message input/output config syntax.

Creates models/multi_io_model/1/{model.xml, model.bin} and a config.pbtxt
that uses the unified array form with TWO inputs and TWO outputs:

  input:  [ { name:"input_a" ... }, { name:"input_b" ... } ]
  output: [ { name:"out_a" ... },   { name:"out_b" ... } ]

The model computes:
  out_a = input_a * 2 + 1   (element-wise over [1,4])
  out_b = input_b - 3       (element-wise over [1,4])
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
import openvino.opset8 as ops


def build_model():
    a = ops.parameter([1, 4], ov.Type.f32, name="input_a")
    b = ops.parameter([1, 4], ov.Type.f32, name="input_b")

    scale = ops.constant(np.array([2.0], dtype=np.float32), name="scale")
    bias = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    out_a = ops.add(ops.multiply(a, scale, name="mul_a"), bias, name="out_a")

    offset = ops.constant(np.array([3.0], dtype=np.float32), name="offset")
    out_b = ops.subtract(b, offset, name="out_b")

    return ov.Model([out_a, out_b], [a, b], "multi_io_model")


def main():
    parser = argparse.ArgumentParser(description="Generate a multi-IO OpenVINO model.")
    parser.add_argument("--out", default="models/multi_io_model", help="Output model dir")
    args = parser.parse_args()

    version_dir = os.path.join(args.out, "1")
    os.makedirs(version_dir, exist_ok=True)

    model = build_model()
    ov.save_model(model, os.path.join(version_dir, "model.xml"))

    cfg = """name: "multi_io_model"
backend: "openvino"
max_batch_size: 0
input: [
  {
    name: "input_a"
    data_type: TYPE_FP32
    dims: [ 1, 4 ]
  },
  {
    name: "input_b"
    data_type: TYPE_FP32
    dims: [ 1, 4 ]
  }
]
output: [
  {
    name: "out_a"
    data_type: TYPE_FP32
    dims: [ 1, 4 ]
  },
  {
    name: "out_b"
    data_type: TYPE_FP32
    dims: [ 1, 4 ]
  }
]
instance_group {
  count: 1
  kind: KIND_CPU
}
"""
    with open(os.path.join(args.out, "config.pbtxt"), "w") as f:
        f.write(cfg)

    meta = {
        "model_id": "multi_io_model",
        "version": "1.0.0",
        "intended_use": "Test multi-input/multi-output Triton array config syntax",
        "training_dataset_id": "N/A",
        "approval_status": "approved",
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Multi-IO model written to", os.path.abspath(args.out))
    print("  version dir :", os.path.abspath(version_dir))


if __name__ == "__main__":
    main()
