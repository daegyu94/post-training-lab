# Spark Repeated Measurements

This record reports the short Spark measurement plan, with two measured repeats per variant.
Read the timing ranges as observations of this fixed small-model workload, not general speedup claims.

Commit: `862624156b0f55504de64d66a7e1a0cebcf9a3b3`.
The retained overlap-mbs-1 cell ran at ecd1e50; all remaining cells ran at the commit above.

Actual budget: 16 optimizer steps, two measured repeats per variant, first four within-run steps excluded.
Each variant also has a separate four-step warmup.
The measured order is A/B followed by B/A for each cell.

| Cell | Variant | Passed measured repeats | Step median range (ms) | Peak CUDA allocated (GiB) | Whole run range (s) | Save-call total range (s) | Blocking finalization total range (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap-mbs-1 | off | 2 | 397.350–398.450 | 6.584 | 132.698–136.696 | 63.895–63.995 | 0.000–0.000 |
| overlap-mbs-1 | on | 2 | 359.850–365.300 | 6.566 | 90.102–122.375 | 63.576–63.693 | 0.000–0.000 |
| overlap-mbs-2 | off | 2 | 367.900–369.700 | 7.598 | 120.141–135.276 | 63.789–63.857 | 0.000–0.000 |
| overlap-mbs-2 | on | 2 | 328.650–328.700 | 7.581 | 119.628–124.127 | 63.594–64.095 | 0.000–0.000 |
| recompute-length-2048 | full | 2 | 1715.500–1718.050 | 7.492 | 142.037–145.326 | 63.399–63.843 | 0.000–0.000 |
| recompute-length-2048 | selective | 0 | unavailable | unavailable | unavailable | unavailable | unavailable |
| recompute-length-4096 | full | 2 | 5136.150–5140.200 | 9.648 | 198.624–200.088 | 63.536–64.150 | 0.000–0.000 |
| recompute-length-4096 | selective | 0 | unavailable | unavailable | unavailable | unavailable | unavailable |
| sequence-parallel-length-2048 | off | 2 | 892.250–896.100 | 5.161 | 127.085–141.247 | 63.566–63.968 | 0.000–0.000 |
| sequence-parallel-length-2048 | on | 2 | 926.250–929.050 | 5.121 | 98.137–136.540 | 64.017–64.292 | 0.000–0.000 |
| sequence-parallel-length-4096 | off | 2 | 1677.700–1683.650 | 6.118 | 110.545–151.256 | 63.955–64.328 | 0.000–0.000 |
| sequence-parallel-length-4096 | on | 2 | 1741.500–1742.650 | 6.036 | 141.155–146.677 | 62.735–63.775 | 0.000–0.000 |
| checkpoint-interval-16 | sync | 2 | 398.600–402.100 | 6.584 | 304.891–313.383 | 247.626–248.255 | 0.000–0.000 |
| checkpoint-interval-16 | async | 2 | 403.800–409.900 | 6.584 | 266.752–305.863 | 177.934–181.176 | 60.830–60.926 |
| checkpoint-interval-32 | sync | 2 | 396.350–404.450 | 6.584 | 152.914–187.618 | 124.426–125.020 | 0.000–0.000 |
| checkpoint-interval-32 | async | 2 | 403.250–403.800 | 6.584 | 149.839–188.121 | 60.630–61.114 | 61.149–62.099 |

## Reading the measurements

Checkpoint cell names retain the default intervals 16/32; this execution used effective intervals 4/8, recorded in the per-run configs.
All other comparisons save once at their final step.
Whole-run time includes remote launch/claim waiting, model setup, training, checkpoint work, evaluation and teardown.

Memory is the maximum CUDA allocator peak across ranks over the full train stage, including model/optimizer setup and evaluation; it is not total device memory.

Save and blocking-finalization events are accumulated per rank and then maximized across ranks.
The installed Bridge pauses its native interval timer around checkpoint save calls; background async contention can still affect training steps.
These checkpoint measurements are host call times, including possible synchronization and filesystem waiting, not isolated disk bandwidth or crash durability.

## Evidence

[Manifest and summaries](manifest.json), [incremental records](records.jsonl), [remaining plan](remaining/plan.json) and [initial plan](initial/plan.json), [setup used](setup-used.json), and per-run configs/logs/measurement JSONL are retained here.
Raw provenance paths refer to the controller checkout and Spark mounts used during execution.
Large checkpoints remain in the shared results directory and are not stored in Git.

## Failures and Interpretation

Selective recompute at both lengths was rejected before training because the current backend passes recompute_num_layers=1 and Bridge requires None.
This is a current configuration failure, not evidence of a Spark hardware limitation.

- recompute-length-2048 / selective / run 7: failed; see rank logs and run manifest
- recompute-length-2048 / selective / run 9: skipped; variant failed earlier; remaining repeats skipped
- recompute-length-2048 / selective / run 10: skipped; variant failed earlier; remaining repeats skipped
- recompute-length-4096 / selective / run 13: failed; see rank logs and run manifest
- recompute-length-4096 / selective / run 15: skipped; variant failed earlier; remaining repeats skipped
- recompute-length-4096 / selective / run 16: skipped; variant failed earlier; remaining repeats skipped

The initial matrix was stopped before any length comparison to enable fixed padding; its completed overlap-mbs-1 cell is retained, and all remaining cells were rerun.
These are descriptive measurements of a pinned small-model workload with only two repeats per variant.
Finite loss and gradient norms and zero skipped/NaN steps establish bounded execution evidence, not model quality.
No uninterrupted-versus-resumed numerical equivalence or failure-injection durability experiment is included.
