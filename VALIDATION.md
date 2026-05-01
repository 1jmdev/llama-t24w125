# Validation performed in this environment

Environment validation was limited to repository-local unit tests. The hosted environment does not include Llama 3.2 weights, Ultra-FineWeb data, or a strong CUDA GPU for the requested 10 GB post-training run.

Validated:

```text
5-bit code bitpack/unpack roundtrip
T24 dense fake quantization shape/value consistency
exactly 2 nonzeros per 4 weights
T24LinearSTE forward/backward path
Muon optimizer step on a T24 layer
```

Result:

```text
5 passed
```
