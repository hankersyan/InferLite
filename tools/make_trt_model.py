#!/usr/bin/env python
"""Build a minimal TensorRT engine (model.plan) for the InferLite GPU backend.

The engine computes  output = input + 1.0  for a fixed [1,4] FP32 tensor, using
TensorRT's elementwise layer. It is saved as:
    models/sample_trt_model/1/model.plan
with a matching config.pbtxt (backend: "tensorrt", instance_group KIND_GPU).

Usage:
    python tools/make_trt_model.py
Requires the tensorrt Python wheel matching this Python (pip install the
cpXXX wheel from the TensorRT package's python/ directory) and numpy.
"""
import os

import numpy as np
import tensorrt as trt


def build(plan_path: str) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    inp = network.add_input("input", trt.float32, (1, 4))
    profile.set_shape(inp.name, (1, 4), (1, 4), (1, 4))

    # output = input + 1.0  (elementwise)
    const = network.add_constant((1, 4), trt.Weights(np.ones((1, 4), dtype=np.float32)))
    ew = network.add_elementwise(inp, const.get_output(0), trt.ElementWiseOperation.SUM)
    out = ew.get_output(0)
    out.name = "output"
    network.mark_output(out)

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(engine)
    print("Wrote", os.path.abspath(plan_path))


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    build(os.path.join(base, "sample_trt_model", "1", "model.plan"))
