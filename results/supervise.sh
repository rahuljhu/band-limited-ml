#!/usr/bin/env bash
# Overnight watchdog. Every 3 min: if a job is unfinished and nothing is driving it, relaunch it
# (colab_job.sh resumes from saved results). Uses `resume` if the VM session is still recorded.
cd "$(dirname "$0")/.."
running() { pgrep -f "colab_job.sh (start|resume) $1( |\$)" >/dev/null; }
slots_free() { [ "$(colab sessions 2>/dev/null | grep -c '^\[blml')" -lt 2 ]; }
nfiles() { ls $1 2>/dev/null | wc -l | tr -d ' '; }
log() { echo "[$(date +%m-%d\ %H:%M)] $*" >> results/supervise.log; }
launch() {   # job, script, then env assignments
  local job=$1 script=$2; shift 2
  if [ -f results/.$job.session ]; then
    nohup ./colab_job.sh resume $job >> results/${job}_follow.log 2>&1 &
    log "resume $job (VM session still recorded)"
  elif slots_free; then
    env "$@" nohup ./colab_job.sh start $job $script >> results/${job}_follow.log 2>&1 &
    log "start $job"
  fi
}
beat=0
while true; do
  pgrep caffeinate >/dev/null || { nohup caffeinate -is >/dev/null 2>&1 & log "restarted caffeinate"; }
  t2=$(nfiles "results/p1t2/r1/T2_*.json"); t3=$(nfiles "results/p1x/r1/T3_*.json")
  tab=$(nfiles "results/p2tab/p2/*.json"); big=$(nfiles "results/p1big/big/*.json")
  [ "$t2" -lt 30 ] && ! running p1t2 && launch p1t2 p1_experiments.py BLML_P1_PARTS=rate BLML_P1_TRUTHS=T2
  [ "$t3" -lt 30 ] && ! running p1x && launch p1x p1_experiments.py BLML_P1_TRUTHS=T3
  if [ "$t3" -ge 30 ] && ! pgrep -f queue_tabpfn.sh >/dev/null; then
    cp results/p2tab/p2/*.json results/p2/p2/ 2>/dev/null
    [ "$tab" -lt 21 ] && ! running p2tab && launch p2tab p2_experiments.py BLML_P2_MODELS=TabPFN
  fi
  if [ "$t3" -ge 30 ] && [ "$tab" -ge 21 ] && ! pgrep -f queue_bign.sh >/dev/null; then
    [ "$big" -lt 6 ] && ! running p1big && launch p1big p1_experiments.py BLML_P1_PARTS=bign
  fi
  beat=$((beat+1)); [ $((beat % 10)) -eq 1 ] && log "ok: T2 $t2/30, T3 $t3/30, TabPFN $tab/21, bigN $big/6"
  sleep 180
done
