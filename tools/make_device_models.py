#!/usr/bin/env python
"""Generate Phase 4 multi-device sample models (CPU / NPU / Intel GPU / AUTO).

Creates model directories under `models/` for each Intel execution device using
the OpenVINO Python API. Configs use the Triton-style `kind` field:

  intel_cpu_model  -> model.xml + model.bin  (kind: KIND_CPU)
  intel_npu_model  -> model.npu_blob         (kind: KIND_NPU, compiled for NPU)
  intel_gpu_model  -> model.gpu_blob         (kind: KIND_GPU_INTEL, GPU)
  intel_auto_model -> model.xml + model.bin  (kind: KIND_AUTO, compiles from IR)

NPU/GPU blobs require the corresponding hardware to be present and enabled by
OpenVINO. If a device is unavailable, the model directory is still created with
the IR files and a config using `kind: KIND_CPU` so the repository remains
startable (fail-fast is relaxed for examples). Run with --force to overwrite.

Note: the sample model computes y = 2*x + 1 element-wise over [1, 4].
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
import openvino.opset8 as ops


def build_model():
    x = ops.parameter([1, 4], ov.Type.f32, name="input")
    const = ops.constant(np.array([2.0], dtype=np.float32), name="scale")
    mul = ops.multiply(x, const, name="mul")
    bias = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    add = ops.add(mul, bias, name="add")
    return ov.Model([add], [x], "sample_model")


# Device identifier -> Triton-style KIND_* string.
DEVICE_KIND = {
    "cpu": "KIND_CPU",
    "npu": "KIND_NPU",
    "gpui": "KIND_GPU_INTEL",
    "auto": "KIND_AUTO",
}


def write_config(model_name, device, count, out_dir, blob_required=False):
    kind = DEVICE_KIND[device]
    device_comment = ""
    if device == "npu":
        device_comment = "# Requires model.npu_blob (precompiled NPU blob).\n"
    elif device == "gpui":
        device_comment = "# Requires model.gpu_blob (precompiled Intel GPU blob).\n"
    elif device == "auto":
        device_comment = "# OpenVINO AUTO selects best available Intel device.\n"
    cfg = ('# Intel %s model (Phase 4).\n%s'
           'name: "%s"\n'
           'backend: "openvino"\n'
           'max_batch_size: 0\n'
           'input {\n'
           '  name: "input"\n'
           '  data_type: TYPE_FP32\n'
           '  dims: [ 1, 4 ]\n'
           '}\n'
           'output {\n'
           '  name: "add"\n'
           '  data_type: TYPE_FP32\n'
           '  dims: [ 1, 4 ]\n'
           '}\n'
           'instance_group {\n'
           '  count: %d\n'
           '  kind: %s\n'
           '}\n') % (device, device_comment, model_name, count, kind)
    with open(os.path.join(out_dir, "config.pbtxt"), "w") as f:
        f.write(cfg)


def write_metadata(model_name, out_dir):
    meta = {
        "model_id": model_name,
        "version": "1.0.0",
        "intended_use": "Phase 4 multi-device sample: y = 2x + 1",
        "training_dataset_id": "N/A",
        "approval_status": "approved",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def make_device_model(root, model_name, device, count):
    version_dir = os.path.join(root, model_name, "1")
    os.makedirs(version_dir, exist_ok=True)

    model = build_model()
    # Always write the IR so CPU/AUTO can compile from it and the repo is
    # startable even when the target device is absent. save_model emits both
    # model.xml and model.bin.
    ov.save_model(model, os.path.join(version_dir, "model.xml"))

    # Map our device identifier to the OpenVINO plugin device name.
    ov_name = {"cpu": "CPU", "npu": "NPU", "gpui": "GPU", "auto": "AUTO"}[device]
    available = ov_name in ov.Core().available_devices
    blob_name = None
    if device == "npu":
        blob_name = "model.npu_blob"
    elif device == "gpui":
        blob_name = "model.gpu_blob"

    if blob_name is not None:
        if available:
            compiled = ov.Core().compile_model(model, ov_name)
            blob_path = os.path.join(version_dir, blob_name)
            import io
            buf = io.BytesIO()
            compiled.export_model(buf)
            with open(blob_path, "wb") as f:
                f.write(buf.getvalue())
            print("  [%s] exported %s (device %s)" % (model_name, blob_name, ov_name))
        else:
            print("  [%s] device '%s' unavailable; writing IR only (kind=KIND_CPU)" %
                  (model_name, device))
            device = "cpu"
    elif device == "auto":
        # AUTO may import a blob; here we let it compile from IR at server load.
        print("  [%s] AUTO: writing IR; blob optional" % model_name)

    write_config(model_name, device, count, os.path.join(root, model_name))
    write_metadata(model_name, os.path.join(root, model_name))


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 4 device samples")
    parser.add_argument("--repo", default="models", help="Model repository root")
    args = parser.parse_args()

    make_device_model(args.repo, "intel_cpu_model", "cpu", 2)
    make_device_model(args.repo, "intel_npu_model", "npu", 2)
    make_device_model(args.repo, "intel_gpu_model", "gpui", 1)
    make_device_model(args.repo, "intel_auto_model", "auto", 2)

    print("\nDone. Available OpenVINO devices:",
          ", ".join(ov.Core().available_devices) or "none")


if __name__ == "__main__":
    main()
