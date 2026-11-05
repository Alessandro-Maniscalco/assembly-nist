# SO101 overnight status

Goal: record verified behavior, remaining failures and the next physical setup without conflating scripted motion, prepared-state learning and loose-part assembly.

**Full assembly from loose parts remains unlearned.** The fresh fork, native simulation, optional GPU learner, checkpoints and independent evaluation are operational. No physical robot was operated.

## Verified results

| Check | Outcome and limit |
| --- | --- |
| Learning recipe | Upstream PPO update, SimBa networks and SGS equations remain unchanged. SO101 actions/observations, task scoring and the small-bank sampling floor are explicit adaptations. |
| GPU run | Passing five-update pilot followed by 68 total updates / 278,528 transitions. It stopped cleanly at 14:09 UTC after the independent behavioral regression below; no numerical failures, forced kill or restart. |
| Separate corrective trial | Starts from frozen review iteration 60 with fresh optimizer/SGS and safety penalty 1.0. Three pilot updates / 12,288 transitions pass at 80.6–82.2 transitions/s with no numerical errors. The timed continuation and independent evaluation are pending completion; safety improvement is not established. |
| GPU physics | Whole-helper SDF geometry, physical tool replay and environment/reset/sensor checks pass. GPU tolerance is explicitly 1e-6; canonical native tolerance remains 1e-8. Exact GPU reproducibility is unqualified. |
| Native transfer of GPU checkpoint | Iteration 20: 12/12 mean and 11/12 sampled successes on twelve held-out prepared starts. One sampled motor episode exceeds the base–table contact-depth guard: 0.439483 mm versus 0.400 mm. No replacement trials. |
| Release audit | All 23 current holdout successes have no hand/tool contacts over their final 800 native physics steps. The earlier native reference was 24/24 under the control-sampled rule, with two brief contacts in one fine-step window. |
| Loose evaluation | Frozen GPU iteration 60: 0/10 successes, all 10 safety terminations. The earlier native check was 0/10 with 2 safety terminations and 8 timeouts. Backend causality is unresolved; these small checks are diagnostics. |
| Curriculum meaning | At native iteration 105, 4,001 of 4,004 TRAIN successes started within the 1 mm placement tolerance. Only three started beyond it. A declared 1.796 mm motor-start probe failed in both mean and sampled first episodes. |
| Motor manipulation | One nominal scripted loose-to-seated motor assembly passes 140,000 uninterrupted native steps / 70 s, zero state reloads/warnings, all motion limits and ≤0.158 mm native contact depth. Final 0.4 s: ≤0.631 mm error, no motor/base hand/tool contact, 62 µm translation drift. This uses the experimental prototype below. |
| Screwdriver handling | Native pickup, lift, physical ON/OFF, coast-down, return and release pass over 40,601 steps. This is scripted, one nominal start, and does not establish screw tightening. |
| Further tool design | A fixed-finger locator plus passive rocking trigger pad improves contact behavior, but narrowly fails its rotation-retention gate. It remains isolated; reaction torque and full screw driving are unqualified. |
| Screw coupling | A prepared native trace turns the screw 2.794 revolutions and advances it 1.414 mm relative to the motor. It fails the 30 rad/s screw-speed guard and remains in tool contact at the end. Full screw-task success is false. |

Horn and retaining-frame release from prepared holds are demonstrated. Their tabletop acquisition/insertion, yoke assembly and complete five-stage sequencing remain unverified. Screw and yoke currently have only loose training starts.

## Selected physical prototype

Use opposing mounts along the insertion direction:

- Left base: **(-0.011, +0.35, +0.0024) m**, yaw **−90°**.
- Right base: **(0, −0.20, +0.0024) m**, yaw **+90°**.

Coordinates use the simulation table surface as Z = 0. This arrangement has prepared insertion/table-placement evidence and reduces the unfavorable shoulder-pan lever arm of the old central insertion direction. The tested prototype uses 1 N m gripper and 2.94 N m arm caps; they require hardware identification. The active training model retains its separately qualified ±0.25 m X mounts and 0.35 N m gripper caps. Existing banks/checkpoints are not silently migrated to the prototype.

The [fit audit](SO101_FIT.md) finds the same 31.8 mm base cavity in both vendor 3DM and printable 3MF, around a roughly 32.0 mm motor. Actual printing tolerances, deformation and insertion force remain unmeasured. Native contact depth is not maximum triangle overlap.

## Review and next gates

Use [setup/run instructions](SO101.md), [tool evidence](SO101_TOOL.md) and [GPU qualification](SO101_GPU_CANDIDATE.md). Local evidence is deliberately excluded from Git:

- `runs/so101_gpu_tol1e6_20261105/`: live health, supervisor, checkpoints and scheduled reviews.
- `runs/so101_safety_penalty_20261105/`: separate corrective trial; its parent is the frozen review, not the later iteration-68 final checkpoint.
- `artifacts_local/overnight_progress.{png,json}`: phase-separated learning curves.
- `artifacts_local/gpu_native_holdout/handoff.json`: fixed native-transfer results and failure replay.
- `artifacts_local/curriculum_progress/handoff.json`: actual start distances, sampler coverage and failed insertion probes.
- `artifacts_local/prepared_screw/handoff.json`: physical coupling and unresolved failures.

The motor replay's side-push load peaks at 11.964 N. Pickup produces opposing jaw-normal loads of about 57 N per jaw, with only 0.50 N net motor load at the scalar peak; the compression itself remains uncalibrated. The 70 s demonstration also exceeds the current 15 s learning horizon. A curriculum and time budget for the selected layout need qualification; the demonstration does not prove that this policy can complete the task within 15 s.

Next gates are randomized motor robustness, stable tool reaction-torque/bit alignment, screw release, and task-specific pickup/insertion curricula. Simulator observations use exact body poses and CAD point samples; real perception and calibration are not implemented. No hardware-transfer claim follows from these simulation results.
