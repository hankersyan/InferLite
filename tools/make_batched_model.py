#!/usr/bin/env python
"""Generate a Triton-style batched OpenVINO IR sample model for InferLite.

Demonstrates the Triton batching convention implemented by InferLite:

  max_batch_size: 1   (batching enabled, at most 1 request per inference)
  input/output dims   (per-request, NO batch dimension)
      input: dims: [ 4 ]      -> client sends shape [1, 4]
      output: dims: [ 4 ]     -> server returns shape [1, 4]

Per Triton, when `max_batch_size > 0` the `dims` in config.pbtxt describe the
per-request shape WITHOUT the leading batch dimension; the actual tensors a
client sends / receives carry that batch dimension as dim 0
(1 <= batch <= max_batch_size). Here max_batch_size=1 so the batch dim is 1.

Creates models/batched_model/1/{model.xml, model.bin} and config.pbtxt.
The model computes y = 2*x + 1 element-wise over a [1, 4] tensor.
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
import openvino.opset8 as ops


def build_model():
    # IR input shape is the FULL shape a client will send: [batch, ...dims].
    x = ops.parameter([1, 4], ov.Type.f32, name="input")
    const = ops.constant(np.array([2.0], dtype=np.float32), name="scale")
    mul = ops.multiply(x, const, name="mul")
    bias = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    add = ops.add(mul, bias, name="add")
    return ov.Model([add], [x], "batched_model")


def main():
    parser = argparse.ArgumentParser(description="Generate a Triton-batched OpenVINO model.")
    parser.add_argument("--out", default="models/batched_model", help="Output model dir")
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Triton max_batch_size (default 1)")
    args = parser.parse_args()

    version_dir = os.path.join(args.out, "1")
    os.makedirs(version_dir, exist_ok=True)

    model = build_model()
    ov.save_model(model, os.path.join(version_dir, "model.xml"))

    # Triton batching: config dims are per-request (no batch dimension). The
    # client must send shapes that prepend the batch dimension: [B, ...dims].
    cfg = """name: "batched_model"
backend: "openvino"
max_batch_size: %d
input {
  name: "input"
  data_type: TYPE_FP32
  dims: [ 4 ]
}
output {
  name: "add"
  data_type: TYPE_FP32
  dims: [ 4 ]
}
instance_group {
  count: 1
  kind: KIND_CPU
}
""" % args.max_batch_size
    with open(os.path.join(args.out, "config.pbtxt"), "w") as f:
        f.write(cfg)

    meta = {
        "model_id": "batched_model",
        "version": "1.0.0",
        "intended_use": "Test Triton-style batching (max_batch_size=%d): y = 2x + 1" %
                        args.max_batch_size,
        "training_dataset_id": "N/A",
        "approval_status": "approved",
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Batched model written to", os.path.abspath(args.out))
    print("  version dir     :", os.path.abspath(version_dir))
    print("  max_batch_size  :", args.max_batch_size)
    print("  config dims     : [ 4 ]  (per-request, no batch dim)")
    print("  client shape    : [ 1, 4 ]  (batch dim prepended)")


if __name__ == "__main__":
    main()
