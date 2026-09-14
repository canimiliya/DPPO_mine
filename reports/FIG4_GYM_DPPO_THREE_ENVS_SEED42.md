# Gym Figure 4 DPPO three-environment seed=42 overview

## Status

**COMPLETE as a single-seed three-environment DPPO overview**

This summary combines the three formal Gym DPPO runs for Hopper, Walker2D,
and HalfCheetah. It is a single-seed overview, not a five-seed statistical
reproduction and not a claim of exact recovery of the paper's aggregate
Figure 4 statistics.

## Results

| Environment | Initial eval reward | Final eval reward | Best eval reward | Best iteration | Final environment steps | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|
| Hopper | 1436.617177925794749 | 3092.054468005484068 | 3092.054468005484068 | 990 | 71,280,000 | 4.1509 h |
| Walker2D | 2772.202863878287189 | 3943.777933631889937 | 3979.244093043152134 | 970 | 71,280,000 | 4.2122 h |
| HalfCheetah | 4170.441033292450811 | 5133.426464062357809 | 5164.536209644036717 | 980 | 71,280,000 | 4.1061 h |

All plotted points are the real evaluation points from the three formal CSV
files; no paper-curve digitization, interpolation, or cross-environment reward
normalization was used. Each panel retains its own y-axis scale.

## Scope boundary

Gym Walker2D/HalfCheetah IDQL and DIPO, new seeds, five-seed statistics,
mechanism/ablation experiments, and Figure 18 remain outside this completed
scope.
