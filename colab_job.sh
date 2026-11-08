#!/usr/bin/env bash
# Run a long experiment script detached on a Colab GPU VM and keep pulling results.
#
#   ./colab_job.sh start  <job> <script.py> [GPU]   new VM (default A100; CPU: a CPU-only high-RAM runtime), upload, launch, follow
#   ./colab_job.sh resume <job>                     reattach to the job's VM and keep following
#   ./colab_job.sh fetch  <job>                     one-off download of results so far
#   ./colab_job.sh stop   <job>                     download results, then release the VM
#
# The script runs under nohup on the VM, but a free VM with no client attached is
# reclaimed after a while, so start/resume keep this Mac awake (caffeinate -i;
# a closed lid on battery still sleeps); after a sleep, rerun `resume`. Results are synced to
# results/<job>/ after every poll. Scripts should write one file per finished
# experiment and skip experiments whose file already exists: `start` uploads
# results/<job>/ first, so a fresh VM resumes where the last one stopped.
# Code comes from this folder (*.py) and data from ~/data_repo/caiso.
set -euo pipefail
if [[ "${1:-}" =~ ^(start|resume)$ && -z "${COLAB_JOB_AWAKE:-}" ]] && command -v caffeinate >/dev/null; then
  COLAB_JOB_AWAKE=1 exec caffeinate -i "$0" "$@"
fi

CMD="${1:?usage: colab_job.sh start|resume|fetch|stop <job> ...}"
JOB="${2:?job name required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${BLML_DATA_SRC:-$HOME/data_repo/caiso}"
RES="$HERE/results/$JOB"
STATE="$HERE/results/.$JOB.session"
REMOTE=/content/blml
POLL=20
mkdir -p "$RES"

# Hard wall-clock limit on every remote call: a dropped connection can leave
# `colab exec` hanging forever despite its own --timeout (macOS has no timeout(1)).
limit() { perl -e 'alarm shift; exec @ARGV' "$@"; }
rexec() { limit 300 colab exec -s "$SESSION" --timeout 240 2>/dev/null; }
# Status checks are quick; a short limit turns a stalled connection into one 20 s retry.
rstat() { limit 30 colab exec -s "$SESSION" --timeout 25 2>/dev/null; }
# Colab CLI calls fail transiently (a hung exec hits the alarm, an upload returns non-zero): retry setup steps.
retry() { local i; for i in 1 2 3; do "$@" && return 0; echo "[retry $i] $*" >&2; sleep 10; done; return 1; }
upload() { retry limit 300 colab upload -s "$SESSION" "$1" "$2" >/dev/null; }

fetch() {
  echo "import shutil; shutil.make_archive('$REMOTE/out_sync', 'zip', '$REMOTE/outputs')" | rexec >/dev/null
  limit 600 colab download -s "$SESSION" "$REMOTE/out_sync.zip" "$RES/.sync.zip" >/dev/null 2>&1 || return 1
  unzip -o -q "$RES/.sync.zip" -d "$RES" && rm -f "$RES/.sync.zip"
}

follow() {
  local last="" n=0
  while true; do
    status=$(cat <<EOF | rstat || echo "UNREACHABLE"
import os
pid = open('$REMOTE/job.pid').read().strip()
alive = os.path.exists(f'/proc/{pid}') and 'zombie' not in open(f'/proc/{pid}/status').read().lower()
log = open('$REMOTE/outputs/job.log').read().splitlines()
print('ALIVE' if alive else 'FINISHED', sum(len(f) for _, _, f in os.walk('$REMOTE/outputs')))
print('\n'.join(log[-3:]))
EOF
)
    if [[ "$status" == UNREACHABLE* ]]; then
      echo "[$(date +%H:%M)] VM unreachable; retrying in ${POLL}s (rerun 'resume $JOB' if this persists)"
      sleep "$POLL"; continue
    fi
    head=$(echo "$status" | grep -E '^(ALIVE|FINISHED)' | head -1 || true)
    n=$((n + 1))                       # also every 15 polls (~5 min): a VM ended by Colab loses little
    if [[ "$head" != "$last" ]] || (( n % 15 == 0 )); then fetch || true; last="$head"; fi
    echo "[$(date +%H:%M)] $(echo "$status" | tail -n +1 | tr '\n' ' ' | cut -c1-200)"
    if [[ "$head" == FINISHED* ]]; then
      fetch || true
      colab stop -s "$SESSION" || true
      rm -f "$STATE"
      echo "Job $JOB finished; results in $RES"
      return 0
    fi
    sleep "$POLL"
  done
}

case "$CMD" in
  start)
    SCRIPT="${3:?script required}"; GPU="${4:-A100}"
    SESSION="blml-$JOB-$(date +%m%d%H%M)"
    if [[ "$GPU" == CPU ]]; then colab new -s "$SESSION" --high-mem   # CPU-only runtime (high-RAM shape: more cores)
    else colab new -s "$SESSION" --gpu "$GPU"; fi
    echo "$SESSION" > "$STATE"
    retry bash -c "echo \"import os; [os.makedirs('$REMOTE/'+d, exist_ok=True) for d in ('code','data','outputs')]\" | perl -e 'alarm shift; exec @ARGV' 300 colab exec -s $SESSION --timeout 240"
    for f in "$HERE"/*.py; do upload "$f" "$REMOTE/code/$(basename "$f")"; done
    for f in "$DATA"/*.npy; do upload "$f" "$REMOTE/data/$(basename "$f")"; done
    # Resume support: earlier results go back up so finished experiments are skipped.
    find "$RES" -type f ! -name '.*' | while read -r f; do
      rel="${f#$RES/}"
      echo "import os; os.makedirs(os.path.dirname('$REMOTE/outputs/$rel'), exist_ok=True)" | rexec >/dev/null
      upload "$f" "$REMOTE/outputs/$rel"
    done
    cat <<EOF | rexec
import subprocess, os
env = dict(os.environ, BLML_CODE_DIR='$REMOTE/code', BLML_DATA_DIR='$REMOTE/data',
           BLML_OUT_DIR='$REMOTE/outputs', PYTHONUNBUFFERED='1', BLML_PARTS='${BLML_PARTS:-parity,checks,frame,stream}', BLML_MAX_SWEEPS='${BLML_MAX_SWEEPS:-}', BLML_SETS='${BLML_SETS:-}', BLML_FLOW_PART='${BLML_FLOW_PART:-}', UCI_NTRAIN='${UCI_NTRAIN:-0}', UCI_SUB='${UCI_SUB:-20000}', UCI_EPOCHS='${UCI_EPOCHS:-12}', UCI_SKIP='${UCI_SKIP:-}', UCI_STREAM_FC='${UCI_STREAM_FC:-}', BLML_P1_PARTS='${BLML_P1_PARTS:-all}', BLML_P1_TRUTHS='${BLML_P1_TRUTHS:-T1,T2}', BLML_P2_MODELS='${BLML_P2_MODELS:-}', BLML_DTYPE='${BLML_DTYPE:-float32}', BLML_BG_HARM='${BLML_BG_HARM:-0,0}', BLML_COLLAPSE='${BLML_COLLAPSE:-0}', BLML_P2_RUNS='${BLML_P2_RUNS:-}', BLML_P2_COVSETS='${BLML_P2_COVSETS:-}', BLML_P1_DLIST='${BLML_P1_DLIST:-1,2,3,4,6,8}', BLML_P1_FC_MULTS='${BLML_P1_FC_MULTS:-1,2}', BLML_FC_MULTS='${BLML_FC_MULTS:-0.5,0.7,1,1.4,2,2.8}', BLML_FC_SUB='${BLML_FC_SUB:-fcsweep}', BLML_FC_TRUTHS='${BLML_FC_TRUTHS:-T1,T3}', BLML_EPOCHS='${BLML_EPOCHS:-}', BLML_REVISIT_SWEEPS='${BLML_REVISIT_SWEEPS:-}', BLML_BATCH_SWEEPS='${BLML_BATCH_SWEEPS:-}',
           TABPFN_TOKEN='${TABPFN_TOKEN:-$(cat "$HOME/.tabpfn_token" 2>/dev/null)}')
log = open('$REMOTE/outputs/job.log', 'a')
p = subprocess.Popen(['nohup', 'python', '$REMOTE/code/$(basename "$SCRIPT")'], cwd='$REMOTE/code',
                     env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open('$REMOTE/job.pid', 'w').write(str(p.pid))
print('launched pid', p.pid)
EOF
    follow ;;
  resume)
    SESSION="$(cat "$STATE")"; follow ;;
  fetch)
    SESSION="$(cat "$STATE")"; fetch && echo "synced to $RES" ;;
  stop)
    SESSION="$(cat "$STATE")"; fetch || true; colab stop -s "$SESSION"; rm -f "$STATE" ;;
  *) echo "unknown command $CMD"; exit 1 ;;
esac
