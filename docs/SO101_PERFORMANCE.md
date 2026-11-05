# SO101 performance

**Goal:** Choose a memory-safe local collection/learning configuration while preserving source PPO, SimBa, SGS, SDF equations and contact settings.

Measured 2026-11-05 on an i9-9900K (8 cores / 16 threads), approximately 15 GiB host RAM, RTX 5080 16 GB, MuJoCo 3.17 and PyTorch 2.11/CUDA 12.8. These are short throughput measurements, not learned assembly results. The fixture predates the passive tool sleeve and final gripper contact revisions; rerun after integration.

## Native collection

Five settled task states, cyclic task distribution, zero actions, one warmup control and four measured controls. Control period 40 ms, physics timestep 0.5 ms. Peak RSS sums worker processes for remote runs; shared pages may be counted more than once.

| SDF lookup | Environments | Collection | Transitions/s | Process RSS, decimal GB |
| --- | ---: | --- | ---: | ---: |
| Original mutex | 10 | 1 thread | 5.28 | 2.09* |
| Original mutex | 10 | 4 threads | 4.23 | 1.69 |
| Thread-local pointer cache | 10 | 1 thread | 5.21 | 1.67 |
| Thread-local pointer cache | 10 | 4 threads | 13.89 | 1.68 |
| Thread-local pointer cache | 40 | 8 threads | 25.76 | 1.84 |
| Thread-local pointer cache | 40 | 16 threads | **32.68** | **1.87** |
| Thread-local pointer cache | 40 | 4 worker processes | 17.66 | 6.80 |

*First invocation also generated the settled fixture, leaving additional model allocation resident. The measured interval excludes fixture creation and initialization.

The initial one-thread profile spent 7.446 of 7.552 profiled seconds in native rollout; observation forwarding used 0.092 seconds. Optimizing Python observation construction would have little effect here. Sixteen threads consumed approximately 8.88 CPU core-seconds per wall second; four processes consumed 3.99.

`get_grid` now caches immutable, globally owned grid pointers per thread. Only initial lookup/loading uses the shared mutex. Grid loading, trilinear interpolation, gradients and analytic thread equations are unchanged. The four-thread result improves 3.28 times. Ten-environment observation SHA-256 is identical for both original configurations and both patched configurations:

`18c4f2f7eac3b485a326e9ac6058322f147aab594851cec9298b93a41142a8c3`

Forty-environment observations also match exactly between 8 threads, 16 threads and 4 TCP workers:

`9dbc60b808a3ba386c2b94a72a55dc03f668b95508b83b9b1804af0a58b4c62c`

All measured transitions were valid, finite, without safety termination or instability. Remote shape/history parity here covers ordinary steps; terminal/reset protocol checks belong to preflight. Scene signature: `ef8785c8891949caf14a95635633cb2cd38d4abe2645af523aa249d35b1fc286`.

## CUDA learner

The actual unchanged `Trainer.update` was timed after one warmup update on repeated real observations with synthetic advantages; no policy checkpoint was saved. Both runs used source defaults: five epochs and four minibatches.

| Batch | Update seconds | Peak Torch allocated | Peak Torch reserved | Host RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 (32 × 32) | 0.129 | 80.9 MB | 107.0 MB | 3.14 GB |
| 4,096 (128 × 32) | 0.125 | 133.8 MB | 195.0 MB | 3.16 GB |

All losses were finite. The few milliseconds between timings are not evidence that the larger batch is faster. CUDA allocator measurements exclude driver/context memory. Both use 320 points, 265 proprioception inputs and 12 actions: actor 1,435,532 + critic 1,432,705 + 12 learned standard deviations = **2,868,249 trainable parameters**.

Start with **32–64 environments, 16 simulation threads, 32 steps, CUDA learner**, retaining source optimizer settings. At the measured 32.68 transitions/s, 1,024 transitions take approximately 31.3 seconds to collect versus 0.13 seconds to update. PPO occupies approximately 0.4% of this estimated iteration time; this is not a measured GPU utilization percentage. Raising batch size cannot remove the CPU physics bottleneck. Two hours at that short-test rate is approximately 235,000 transitions before evaluation, checkpointing or changing contact costs.

## Reproduce and limits

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/so101_benchmark.py \
  --envs 40 --threads 16 --steps 4 --warmup 1 \
  --out artifacts_local/performance/current.json
```

Reuse `--fixture PATH` for matched states (the script rejects a changed scene signature). Add `--backend remote --workers 4` for source TCP workers. Use `--learner-observations PATH.observations.npz --envs 32 --steps 32` for the CUDA update benchmark. Raw JSON, observations, fixtures and profiles are under `artifacts_local/performance/`, excluded from Git.

Sleeping was not enabled: installed MuJoCo 3.17 state definitions omit `tree_asleep` from both `mjSTATE_FULLPHYSICS` and `mjSTATE_INTEGRATION`, while rollout exchanges FULLPHYSICS states. That requires a separate continuation/wake audit.

[Official MJWarp documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#sdf-plugins) supports custom SDF functions written in Warp, not automatic execution of this C++ library on the GPU. Installed MJWarp 3.17 requires `user_sdf`/`user_sdf_grad` implementations and filters model plugins to geom plugins; the passive driver callback needs an explicit port too. No GPU physics port, solver reduction, timestep change or contact simplification was made for these measurements.

## Isolated GPU prototype, 07:27 UTC

**Decision: do not use for overnight training.** `assembly_policy/so101/warp_physics.py` is an experimental backend; production environment and learner dispatch do not import it. A bounded follow-up port was authorized after the native benchmarks above. No training used this prototype.

The prototype compiles the original C++ grid/thread/head equation bodies inside Warp, retaining double arithmetic until float32 contact output. GPU arrays contain the exact decoded half-grid samples. Pointers and float64 grid headers are encoded in exact 16-bit attribute pieces, checked on import, and kept alive by the backend. All fourteen final-scene plugin instances are recognized, including shared jaw grids 201/202. An explicit per-step passive callback applies the original button-dependent torque to the same spindle DOF, preserving its reaction through the joint.

MJWarp 3.17 required two explicit compatibility corrections: its filtered geometry-plugin arrays were allocated with the unfiltered native count; its importer also silently raises solver tolerance to 1e-6. A narrowly scoped allocation shim restores the library helper immediately, and the prototype restores native tolerance to 1e-8. Forty-seven mechanical parameter arrays, contact exclusions, relevant options and initial state/control/applied-force arrays are checked after float32 conversion. These checks do not establish engine-level contact equivalence.

| Probe | Result |
| --- | --- |
| 24,000 SDF queries, twelve contact geoms including both jaws | Maximum distance error 1.86 nm; maximum gradient error 4.49e-6 versus original C++/original configuration values |
| Contact-free gravity and zero/nonzero position actuator, 50 ms | Maximum qpos error below 15 nm/rad; no overflow |
| Guided screw rotation, 2 s, native 0.1 ms bench timestep | Pitch error: native +1.68 µm, GPU −2.02 µm; GPU line-search limit bit 1024 remains |
| Guided axial push, 2 s | Advance: native 7.37 µm, GPU 7.36 µm; no final overflow |
| Isolated physical driver button press then release | Final spindle: native 1.3278817 rad/s, GPU 1.3278801 rad/s; driver position difference 88 nm; no overflow |
| Final full workcell, HOME pose, 16 worlds | Interrupted after approximately 235 s without completing the first zero-action probe; no valid final-scene throughput |

An earlier scene without the new jaw SDFs reached 116.9 controls/s at sixteen worlds, but used the importer's relaxed tolerance and showed contact differences. That number is not a result for the final scene. Inspection found a full triangle loop in MJWarp's mesh–SDF ray query, a possible slowdown source requiring profiling. No collision pair, geometry, solver iteration limit or timestep was removed to improve performance.

Raw probes are in `artifacts_local/warp_prototype/` and `artifacts_local/warp_final/`; the latter uses restored native tolerance. `scripts/so101_warp_check.py` reproduces SDF, basic mechanics, thread/axial and full/isolated-tool probes. Native and GPU traces are retained. Full-scene contact trajectories, grasp/release reliability, continuation semantics (including float32 clock drift) and final-scene performance remain unqualified. All prototype processes stopped and GPU utilization returned to 0% / 34 MiB before handing the GPU back to the native training run.
