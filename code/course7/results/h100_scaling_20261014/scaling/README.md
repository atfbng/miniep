# H100 EP scaling measurement

2 workloads × 3 EP sizes × 3 independent trials = 18 runs.
42 rank reports passed Reference and pairwise split checks.
Each trial uses 50 warmups and 100 repetitions.
Whole-layer time is the median of repetition-wise max-rank CUDA Event samples.
Summary time is the median of three trial medians; whiskers show min/max trial medians.
Stage times are measured independently and must not be summed into total latency.

- `summary.json`: table/plot data
- `reports.json`: per-trial Rank 0 reports
- `raw/`: every rank's samples, layouts, environment, and logs
- `figures/`: exported PNG and SVG figures
- `hardware_topology.txt`: NVLink/NUMA topology
- `gpu_inventory_before.csv`: devices before the experiment

Lesson: https://tvle9mq8jh.feishu.cn/wiki/UullwSRDtillhlkI714cOrbJnNd
