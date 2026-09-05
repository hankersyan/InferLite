#!/usr/bin/env python
"""Generate a Triton dynamic-batching OpenVINO IR model for InferLite.

Implements the batching mode that follows NVIDIA Triton (Phase 7):

  max_batch_size: 8      (batching enabled, at most 8 samples per inference)
  dynamic_batching {     (scheduler coalesces concurrent requests into batches)
    preferred_batch_size: [ 8 ]
    max_queue_delay_microseconds: 150000   # 150 ms
  }
  input/output dims (per-request, NO batch dimension)
      input:  dims: [ 4 ]   -> each request sends shape [B, 4] with 1<=B<=8
      output: dims: [ 4 ]   -> the server returns shape [B, 4]

The IR input is a *dynamic* PartialShape([-1, 4]) so a single backend call can
run any batch in [1, max_batch_size] -- this is what lets the scheduler merge
several concurrent requests (B=1 each, say) into one [N, 4] execution and slice
the output back per request. A model IR compiled with a static batch dimension
(e.g. [1, 4]) cannot serve merged batches and fails at runtime.

Creates models/dynamic_batch_model/1/{model.xml, model.bin}, config.pbtxt and
metadata.json. The model computes y = 2*x + 1 element-wise.
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
import openvino.opset8 as ops


def build_model():
    # Dynamic batch dimension on axis 0: any batch 1..max_batch_size at runtime.
    x = ops.parameter(ov.PartialShape([-1, 4]), ov.Type.f32, name="input")
    const = ops.constant(np.array([2.0], dtype=np.float32), name="scale")
    mul = ops.multiply(x, const, name="mul")
    bias = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    add = ops.add(mul, bias, name="add")
    return ov.Model([add], [x], "dynamic_batch_model")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Triton dynamic-batching OpenVINO model.")
    parser.add_argument("--out", default="models/dynamic_batch_model",
                        help="Output model dir")
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Triton max_batch_size (default 8)")
    parser.add_argument("--preferred", type=str, default="8",
                        help="Comma-separated preferred_batch_size list (default '8')")
    parser.add_argument("--delay-us", type=int, default=150000,
                        help="max_queue_delay_microseconds (default 150000 = 150ms)")
    args = parser.parse_args()

    if args.max_batch_size <= 0:
        raise SystemExit("--max-batch-size must be > 0 for dynamic batching")
    preferred = [int(x) for x in args.preferred.split(",") if x.strip() != ""]
    for p in preferred:
        if not (0 < p <= args.max_batch_size):
            raise SystemExit("preferred_batch_size must be in [1, max_batch_size]")

    version_dir = os.path.join(args.out, "1")
    os.makedirs(version_dir, exist_ok=True)

    model = build_model()
    ov.save_model(model, os.path.join(version_dir, "model.xml"))

    preferred_text = ", ".join(str(p) for p in preferred)
    cfg = """name: "dynamic_batch_model"
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
dynamic_batching {
  preferred_batch_size: [ %s ]
  max_queue_delay_microseconds: %d
}
""" % (args.max_batch_size, preferred_text, args.delay_us)
    with open(os.path.join(args.out, "config.pbtxt"), "w") as f:
        f.write(cfg)

    meta = {
        "model_id": "dynamic_batch_model",
        "version": "1.0.0",
        "intended_use":
            "Test Triton dynamic batching: y = 2x + 1; the scheduler merges "
            "concurrent requests into batches up to max_batch_size=%d" %
            args.max_batch_size,
        "training_dataset_id": "N/A",
        "approval_status": "approved",
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Dynamic-batch model written to", os.path.abspath(args.out))
    print("  max_batch_size            :", args.max_batch_size)
    print("  preferred_batch_size      :", preferred)
    print("  max_queue_delay_us        :", args.delay_us)
    print("  IR input shape            : [-1, 4] (dynamic batch)")
    print("  config dims               : [ 4 ]  (per-request, no batch dim)")
    print("  client request shape      : [B, 4] with 1 <= B <= max_batch_size")


if __name__ == "__main__":
    main()
