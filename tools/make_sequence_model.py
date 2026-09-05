#!/usr/bin/env python
"""Generate a Triton sequence-batching OpenVINO IR model for InferLite.

Creates models/sequence_model/1/{model.xml, model.bin}, config.pbtxt and
metadata.json. The model is stateful through a hidden state tensor kept by the
sequence scheduler (Triton `state`):

  inputs : input [1]  (per-step value), state [1]  (scheduler-owned)
  outputs: output [1] = input + state
           state_out [1] = state + 1     (next hidden state, scheduler-owned)

Clients never send/receive `state`/`state_out`. Each request also carries the
sequence control tensors declared in `sequence_batching.control_input`
(START/END/CORRID) which the scheduler strips before inference:
  START  INT32 0/1   first request of a sequence
  END    INT32 0/1   final request of a sequence
  CORRID INT32       correlation id identifying the sequence
A sequence keeps a single slot on the single instance until it sends END or
becomes idle for max_sequence_idle_microseconds.
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
import openvino.opset8 as ops


def build_model():
    x = ops.parameter([1], ov.Type.f32, name="input")
    s = ops.parameter([1], ov.Type.f32, name="state")
    y = ops.add(x, s, name="output")
    one = ops.constant(np.array([1.0], dtype=np.float32), name="one")
    nxt = ops.add(s, one, name="state_out")
    return ov.Model([y, nxt], [x, s], "sequence_model")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Triton sequence-batching OpenVINO model.")
    parser.add_argument("--out", default="models/sequence_model",
                        help="Output model dir")
    parser.add_argument("--idle-us", type=int, default=300000,
                        help="max_sequence_idle_microseconds (default 300000 = 300ms)")
    args = parser.parse_args()

    version_dir = os.path.join(args.out, "1")
    os.makedirs(version_dir, exist_ok=True)

    model = build_model()
    ov.save_model(model, os.path.join(version_dir, "model.xml"))

    cfg = """name: "sequence_model"
backend: "openvino"
max_batch_size: 0
input {
  name: "input"
  data_type: TYPE_FP32
  dims: [ 1 ]
}
input {
  name: "state"
  data_type: TYPE_FP32
  dims: [ 1 ]
}
output {
  name: "output"
  data_type: TYPE_FP32
  dims: [ 1 ]
}
output {
  name: "state_out"
  data_type: TYPE_FP32
  dims: [ 1 ]
}
instance_group {
  count: 1
  kind: KIND_CPU
}
sequence_batching {
  max_sequence_idle_microseconds: %d
  control_input {
    name: "START"
    control {
      kind: CONTROL_SEQUENCE_START
      int32_false_true: [ 0, 1 ]
    }
  }
  control_input {
    name: "END"
    control {
      kind: CONTROL_SEQUENCE_END
      int32_false_true: [ 0, 1 ]
    }
  }
  control_input {
    name: "CORRID"
    control {
      kind: CONTROL_SEQUENCE_CORRID
      int32_false_true: [ 0, 0 ]
    }
  }
  state {
    input_name: "state"
    output_name: "state_out"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
}
""" % args.idle_us
    with open(os.path.join(args.out, "config.pbtxt"), "w") as f:
        f.write(cfg)

    meta = {
        "model_id": "sequence_model",
        "version": "1.0.0",
        "intended_use":
            "Test Triton sequence batching: stateful y = input + state; "
            "state advances by 1 per step; idle timeout %dus" % args.idle_us,
        "training_dataset_id": "N/A",
        "approval_status": "approved",
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Sequence model written to", os.path.abspath(args.out))
    print("  idle timeout us :", args.idle_us)
    print("  inputs          : input[1], state[1] (state scheduler-owned)")
    print("  outputs         : output[1], state_out[1] (state scheduler-owned)")
    print("  controls        : START/END INT32 0|1, CORRID INT32")


if __name__ == "__main__":
    main()
