# Cross-model rate limiter (Triton-style)

InferLite can coordinate backend executions **across all loaded models** so that
models sharing a scarce resource (the one GPU, its memory, a device copy engine)
never execute concurrently in a way that over-subscribes it. This is the
Triton *rate limiter* model, adapted to InferLite's single-node design and to
the FDA resource-limit story already provided by `MAX_GPU_MEMORY_MB`.

When the rate limiter is disabled (the default) every execution starts as soon
as an instance is free — the historical behavior. Enabling it adds one
reservation step before each backend execution.

## 1. Enable

```
inferlite --model-repository=... --rate-limit ...
```

`--rate-limit` turns on admission control. It is off by default, mirroring
Triton (`--rate-limit=off`).

## 2. Declare resources in config.pbtxt

A model declares what one **execution** consumes inside its instance group:

```proto
instance_group {
  count: 1
  kind: KIND_GPU
  rate_limiter {
    priority: 2                      # optional; higher wins when capacity is contended
    resources {
      name: "GPU_MEMORY"             # any name; shared across models that use it
      count: 1                       # units one execution reserves
      global: true                   # true = one system-wide pool
    }
  }
}
```

Semantics (NVIDIA Triton `ModelInstanceGroup.RateLimiter`):

| Field     | Meaning                                                        |
|-----------|----------------------------------------------------------------|
| `name`    | Resource identifier. Models that list the same name share it.   |
| `count`   | Units reserved for the whole duration of one backend execution. |
| `global`  | `true` → a single system-wide pool; `false` → one pool per execution device (the model's resolved device label). |

Every GPU model that could collide should declare the same global resource with
a `count` proportional to its working set. Example for one small GPU:

- two models each needing the whole device declare
  `GPU_MEMORY { count: 1, global: true }` → capacity defaults to `max(1,1) = 1`,
  so only **one** of them executes at a time (the other waits);
- a half-size model declares `count: 2` against the same pool while the two
  full-size models stay at `count: 2` and the capacity is raised to
  `--rate-limit-resource=GPU_MEMORY:4`: the half-size model can run alongside
  one full-size execution (2 + 2 = 4), but two full-size executions cannot.

In general, choose units so that the sum of the counts of the executions you
want to allow concurrently is at most the pool capacity.

Rules:

- A model with **no** `rate_limiter` block is unrestricted: it never waits and
  never counts against a pool.
- `count` must be ≥ 1 and resource names within one instance group unique
  (validated at load; a bad config fails fast).
- Batching interacts naturally: a **merged dynamic-batch execution** is one
  admission, so a batched model reserves once per backend call, not once per
  request.
- An **ensemble** is gated as a single admission (its steps execute inline), so
  an ensemble config should declare the union of the shared resources its GPU
  steps consume; members that are also served directly declare their own needs
  against the same pools, which keeps both paths safe.
- If a model has `count: N` GPU instances, every instance reserves the same
  `count` per execution; with the default capacity the instances of one model
  can be serialized too. Raise the capacity (next section) to allow more
  concurrency.

## 3. Pool capacity

The available units of a pool default to the **largest single requirement any
loaded model declares** for it — so any one model can always run. Raise it with
the server flag (repeatable):

```
--rate-limit-resource=GPU_MEMORY:2
```

Capacity applies to the resource by name for every pool backing it (global and
per-device). Overriding below the largest declared requirement is allowed but
will serialize models that declare more than the pool can serve.

## 4. When a model waits

A worker that cannot reserve its units is **postponed**: the request stays
queued on the model's scheduler (existing queue semantics apply — timeouts,
`max_queue_size`, priority scheduling all still work). When an execution
finishes, its units are returned and the waiting model with the highest
`rate_limiter.priority` (oldest first among equals) is admitted. During
shutdown / unload a blocked worker is woken and exits without executing, so a
model cannot deadlock a server stop.

## 5. Observability

`GET /v2/health/detailed` reports the limiter state:

```json
"rate_limiter": {
  "enabled": true,
  "pools": [ { "resource": "GPU_MEMORY", "scope": "global",
               "capacity": 1, "used": 0 } ]
}
```

`used` is the number of units currently reserved by running executions. `GET
/v2/models/<name>/config` echoes the parsed `instance_group.rate_limiter` block
back.

## 6. Relation to `MAX_GPU_MEMORY_MB`

`MAX_GPU_MEMORY_MB` caps how much device memory **one** TensorRT model may
allocate (`--max-gpu-memory-mb`, default 2048). The rate limiter is the
*concurrency* companion: it keeps the memory-heavy executions of **several**
models from overlapping on the same GPU in the first place. For a validated
deployment, declare conservative resource counts and pin the pool capacity with
`--rate-limit-resource` so the approved configuration is deterministic.

## 7. Test

`scripts/test_rate_limiter.ps1` builds two "heavy" plugin models that share one
global resource unit and verifies with wall-clock timing that `--rate-limit`
serializes them (≈2× latency) while models that declare no resources keep
running concurrently; then re-runs the heavy pair without the flag and verifies
they overlap (≈1× latency).
