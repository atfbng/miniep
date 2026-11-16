#!/usr/bin/env bash
# 规模扫描：DeepEP 与 RouteEP 在不同输入规模下的 prefill / decode 耗时（4×H100，EP4 = TP2/DP2）。
#
#   prefill：每个 DP 副本 4 条请求，单条长度 128..4096；三组都是 eager
#            ht = DeepEP HT，peer = RouteEP Peer，nccl = RouteEP NCCL
#   decode ：每个 DP 副本 batch 1..64，输入 128、生成 64
#            llg64 = DeepEP LL + FULL_DECODE_ONLY Graph（容量 64），ncclg = RouteEP NCCL + Graph，
#            另带 ht / peer / nccl 的 eager 参照
# 每组 3 个 trial，trial 间轮换顺序；每个 job 预热 2 次、测量 3 次；任何 job 失败立即停止。
#
# 用法（在 routeep 目录、4 卡容器内）：
#   MODEL_ROOT=/models MANIFEST_DIR=results/manifests OUT=results/sweep bash benchmarks/scaling_sweep.sh
#   PHASES=decode MODELS=Qwen3-30B-A3B bash benchmarks/scaling_sweep.sh   # 只跑一部分
# 清单文件用 tools/checkpoint_manifest.py 生成，命名为 <模型目录名>-checkpoint.json。
set -u
MODEL_ROOT=${MODEL_ROOT:-/models}
MANIFEST_DIR=${MANIFEST_DIR:-results/manifests}
OUT=${OUT:-results/sweep}
MODELS=${MODELS:-"Qwen3-30B-A3B Qwen3.5-35B-A3B"}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
COMMON="--warmup 2 --repeats 3 --clock-warmup-seconds 3 --safetensors-load-strategy lazy --kv-cache-mib 4096 --max-model-len 4608 --max-num-seqs 64"
LLENV="NVSHMEM_QP_DEPTH=32768 NVSHMEM_IBGDA_NIC_HANDLER=gpu NVSHMEM_REMOTE_TRANSPORT=none"
DECODE_CASES="decode_b1_s128 decode_b2_s128 decode_b4_s128 decode_b8_s128 decode_b16_s128 decode_b32_s128 decode_b64_s128"
for phase in ${PHASES:-prefill decode}; do
  for m in $MODELS; do
    for t in 0 1 2; do
      if [ "$phase" = prefill ]; then L=(ht peer nccl); else L=(llg64 ncclg ht peer nccl); fi
      n=${#L[@]}; labels=""
      for i in $(seq 0 $((n - 1))); do labels="$labels ${L[$(( (i + t) % n ))]}"; done
      for l in $labels; do
        out=$OUT/$phase-$m-trial$t-$l; rm -rf "$out"; extra=""
        cases=""; [ "$phase" = decode ] && cases="--cases $DECODE_CASES"
        budget="--max-num-batched-tokens 16384"
        case $l in
          ht)    args="--backend native" ;;
          peer)  args="--backend miniep --transport peer" ;;
          nccl)  args="--backend miniep --transport nccl" ;;
          ncclg) args="--backend miniep --transport nccl --execution-mode full-decode-only" ;;
          llg64) args="--backend native --all2all-backend deepep_low_latency --execution-mode full-decode-only --enable-chunked-prefill"
                 budget="--max-num-batched-tokens 64"; extra=$LLENV ;;
        esac
        env $extra timeout 3000 torchrun --standalone --nproc-per-node=4 --monitor_interval=1 "$HERE/deepep_latency_run.py" \
          --model "$MODEL_ROOT/$m" --checkpoint-manifest "$MANIFEST_DIR/$m-checkpoint.json" $args $budget \
          --workload-spec "$HERE/workloads/sweep-$phase.json" $cases --label "sweep-$l" --output "$out" $COMMON > "$out.log" 2>&1
        st=$?
        echo "$(date +%H:%M) $phase $m trial$t $l exit=$st complete=$(ls "$out"/rank*/completed.json 2>/dev/null | wc -l)/4"
        [ $st = 0 ] || { echo "FAILED: see $out.log"; exit 1; }
      done
    done
  done
done
echo ALLDONE
