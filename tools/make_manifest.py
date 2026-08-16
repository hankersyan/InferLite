#!/usr/bin/env python
"""Generate manifest.json for the InferLite model repository.

Computes SHA-256 hashes of each model's artifact files and each plugin library
using the same rule as the server:
  model_hash = SHA256( concat( SHA256_hex(each artifact) ) )
               where artifacts are model.xml + model.bin (OpenVINO),
               model.plan (TensorRT), and the precompiled blobs
               model.npu_blob / model.gpu_blob (Intel NPU / GPU) if present.
  plugin_sha256 = SHA256_hex(plugin library file)

Writes models/manifest.json. The manifest hash is reported by the server.
"""
import argparse
import hashlib
import json
import os
import re


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def model_hash(version_dir):
    # Artifact files in a fixed, deterministic order (matches the server).
    parts = []
    # Order must match the server's config_store.hashModelFiles().
    for name in ("model.xml", "model.bin", "model.plan",
                 "model.npu_blob", "model.gpu_blob"):
        p = os.path.join(version_dir, name)
        if os.path.exists(p):
            parts.append(sha256_file(p))
    if not parts:
        return ""
    return hashlib.sha256("".join(parts).encode("ascii")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate manifest.json")
    parser.add_argument("--repo", default="models", help="Model repository root")
    args = parser.parse_args()

    models = []
    for model_name in sorted(os.listdir(args.repo)):
        model_dir = os.path.join(args.repo, model_name)
        if not os.path.isdir(model_dir):
            continue
        cfg = os.path.join(model_dir, "config.pbtxt")
        if not os.path.exists(cfg):
            continue
        # Find highest numeric version dir.
        versions = []
        for v in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, v)) and v.isdigit():
                versions.append(int(v))
        version = str(max(versions)) if versions else ""
        entry = {"model_id": model_name, "version": version}
        if version:
            entry["sha256"] = model_hash(os.path.join(model_dir, version))
        # Plugin library in the model dir. The filename is taken from the
        # config.pbtxt `plugin_library` field (each plugin model may use its
        # own library); fall back to sample_plugin.dll for backwards
        # compatibility with repositories written before plugin_library.
        plugin_name = None
        with open(cfg, "r", encoding="utf-8", errors="replace") as f:
            m = re.search(r'plugin_library\s*:\s*"([^"]+)"', f.read())
            if m:
                plugin_name = m.group(1)
        if plugin_name is None:
            plugin_name = "sample_plugin.dll"
        plugin = os.path.join(model_dir, plugin_name)
        if os.path.exists(plugin):
            entry["plugin_sha256"] = sha256_file(plugin)
        models.append(entry)

    manifest = {"version": "1", "models": models}
    out = os.path.join(args.repo, "manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote", os.path.abspath(out))
    for e in models:
        print("  ", e["model_id"], e.get("sha256", "")[:16], e.get("plugin_sha256", "")[:16])


if __name__ == "__main__":
    main()
