#!/usr/bin/env python
"""Generate manifest.json for the InferLite model repository.

Computes SHA-256 hashes of each OpenVINO model's artifacts and each plugin
library using the same rule as the server (Phase 4 adds the precompiled blobs):
  model_hash = SHA256( concat( SHA256_hex(model.xml),
                               SHA256_hex(model.bin),
                               SHA256_hex(model.npu_blob),   # if present
                               SHA256_hex(model.gpu_blob) ) )# if present
  plugin_sha256 = SHA256_hex(plugin library file)

Writes models/manifest.json. The manifest hash is reported by the server.
"""
import argparse
import hashlib
import json
import os


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def model_hash(version_dir):
    parts = []
    # Order must match the server's config_store.hashModelFiles().
    for name in ("model.xml", "model.bin", "model.npu_blob", "model.gpu_blob"):
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
        # Plugin library in the model dir.
        plugin = os.path.join(model_dir, "sample_plugin.dll")
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
