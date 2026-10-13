#!/bin/bash
#SBATCH --job-name=gw_diag
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=diag_%j.out
#SBATCH --error=diag_%j.err

# Same fp16 probe as micro_bench.sh, plus a host sampler every second, to
# identify what slows the CPU-side launch path in daytime. Submit once in
# daytime and once at night and compare host_<job>.csv.

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
source ~/greenweight_env.sh

CPUS=$(python -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0)))))")
HOST="host_${SLURM_JOB_ID}.csv"
echo "t,load1,running_procs,mhz_our_cpus,mhz_all_cpus,our_cpus_busy_by_others" > "$HOST"
(
  while true; do
    python - "$CPUS" >> "$HOST" <<'EOF'
import os, sys, time
ours = set(int(c) for c in sys.argv[1].split(","))
mhz, cur = {}, None
for line in open("/proc/cpuinfo"):
    if line.startswith("processor"): cur = int(line.split(":")[1])
    elif line.startswith("cpu MHz") and cur is not None: mhz[cur] = float(line.split(":")[1])
mo = [mhz[c] for c in ours if c in mhz]
la = open("/proc/loadavg").read().split()
# processes of other users last scheduled on one of our CPUs (field 39 = processor)
me = os.getuid(); others = 0
for d in os.listdir("/proc"):
    if not d.isdigit(): continue
    try:
        st = os.stat(f"/proc/{d}")
        f = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
        if st.st_uid != me and f[0] == "R" and int(f[36]) in ours: others += 1
    except (OSError, IndexError, ValueError):
        pass
print(f"{time.time():.1f},{la[0]},{la[3].split('/')[0]},{sum(mo)/len(mo):.0f},"
      f"{sum(mhz.values())/len(mhz):.0f},{others}")
EOF
    sleep 1
  done
) &
SAMPLER=$!

nvidia-smi dmon -s u -d 1 -c 300 > "dmon_${SLURM_JOB_ID}.log" 2>&1 &
DMON=$!
python -u ~/greenAI/major-project/training/scripts/micro_bench.py
kill "$SAMPLER" "$DMON" 2>/dev/null || true

echo "=== host during run (median of samples) ==="
python - "$HOST" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for k in ["load1", "running_procs", "mhz_our_cpus", "mhz_all_cpus", "our_cpus_busy_by_others"]:
    v = [float(r[k]) for r in rows]
    print(f"{k:24s} median={statistics.median(v):.1f} min={min(v):.1f} max={max(v):.1f}")
EOF
echo "=== other jobs on this node ==="
squeue -w "$(hostname)" -o "%.8i %.18j %.12u %.2t %.10M %C %b"
