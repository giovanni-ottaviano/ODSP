#!/usr/bin/env bash
# ODSP benchmark suite: reproduce known results with the odsp driver, then
# compare them with benchmarks/analyse.py.
#
#   benchmarks/run_benchmarks.sh            run every job (about 1 hour at 10 threads)
#   benchmarks/run_benchmarks.sh 'ising|mv' run only the jobs whose name matches the regex
#   THREADS=6 benchmarks/run_benchmarks.sh  use at most 6 threads (default 10)
#
# Jobs run one at a time, each with at most $THREADS threads (the performance
# jobs with one). Results go to benchmarks/results/<job>_*.csv, the driver's
# output to <job>.out, and wall times to timings.txt. Seeds are fixed, so a rerun
# reproduces the same numbers.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$HERE")
O="$ROOT/bin/odsp"
R="$HERE/results"
FILTER=${1:-.}
TH=${THREADS:-10}
export ODSP_MAX_THREADS=${ODSP_MAX_THREADS:-$TH}

make -C "$ROOT" -s bin/odsp || exit 1
mkdir -p "$R"
LOG="$R/timings.txt"
echo "# $(date -Iseconds)  $("$O" --version)  filter '$FILTER'  threads $TH" >> "$LOG"

job() {   # job NAME args...
    local name=$1; shift
    [[ $name =~ $FILTER ]] || return 0
    local t0 t1 st
    t0=$(date +%s.%N)
    "$O" "$@" --csv "$R/$name" > "$R/$name.out" 2>&1
    st=$?
    t1=$(date +%s.%N)
    printf "%-28s status %d  %8.1f s\n" "$name" "$st" "$(echo "$t1 - $t0" | bc)" | tee -a "$LOG"
}

# ---- I. single-thread speed (run with the machine otherwise idle)
for m in "voter" "rf-voter" "ising --temperature 2.269185" "potts --q 3 --temperature 0.994973" "majority-vote --noise 0.075"; do
    tag=$(echo "$m" | cut -d' ' -f1)
    job "perf_${tag}_1000" --model $m --dims 1000x1000 --steps 20 --stop false --every 20 --observables m --threads 1 --seed 1
    job "perf_${tag}_64"   --model $m --dims 64x64 --steps 5000 --stop false --every 5000 --observables m --threads 1 --seed 1
done

# ---- A1. voter on the complete graph (exact birth-death chain)
job voter_complete --model voter --graph complete --task ensemble --replicas 4000 --steps 100000 \
    --sweep n=125,250,500,1000,2000 --threads $TH --seed 11

# ---- F. Deffuant on the complete graph
job deffuant --model deffuant --graph complete --n 1000 --task ensemble --replicas 100 --steps 100000 \
    --observables clusters,largest_cluster --sweep epsilon=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55 \
    --threads $TH --seed 12

# ---- G. Hegselmann-Krause on the complete graph
job hk --model hk --graph complete --n 1000 --task ensemble --replicas 100 --steps 100000 \
    --observables clusters,largest_cluster --sweep epsilon=0.1,0.15,0.18,0.2,0.22,0.25,0.3 \
    --threads $TH --seed 13

# ---- H. coarsening and ageing on 256x256
job coarsen_ising --model ising --temperature 1.5 --dims 256x256 --init random --task ensemble --replicas 40 \
    --steps 2000 --log-times --tw 10,30,100 --observables length,rho,abs_m,autocorrelation --correlation \
    --threads $TH --seed 14
job coarsen_voter --model rf-voter --dims 256x256 --task ensemble --replicas 40 \
    --steps 10000 --log-times --tw 10,30,100 --observables length,rho,autocorrelation --correlation \
    --threads $TH --seed 15

# ---- E. q-voter with independence on the complete graph
for N in 500 2000 8000; do
    job qvoter3_N$N --model qvoter --q 3 --graph complete --n $N --task stationary --observable m --init up \
        --replicas 10 --burn-in 5000 --samples 5000 --interval 10 --sweep p=0.2,0.32,0.33,0.333333,0.34,0.35 \
        --threads $TH --seed $((20 + N))
    job qvoter4_N$N --model qvoter --q 4 --graph complete --n $N --task stationary --observable m --init up \
        --replicas 10 --burn-in 5000 --samples 5000 --interval 10 --sweep p=0.15,0.26,0.27,0.272727,0.28,0.29 \
        --threads $TH --seed $((30 + N))
done

# ---- D. majority-vote model, 2D (q_c = 0.075, Ising class)
for L in 16 32 64; do
    case $L in 16) B=2000; S=20000; I=5;; 32) B=5000; S=20000; I=10;; 64) B=10000; S=10000; I=20;; esac
    job mv_L$L --model majority-vote --dims ${L}x${L} --init up --task stationary --observable m --replicas 10 \
        --burn-in $B --samples $S --interval $I --sweep noise=0.070,0.072,0.074,0.075,0.076,0.078,0.080 \
        --threads $TH --seed $((40 + L))
done

# ---- C. 3-state Potts, heat bath (T_c = 1/ln(1+sqrt 3) = 0.994973)
for L in 16 32 64; do
    case $L in 16) B=2000; S=20000; I=5;; 32) B=5000; S=20000; I=5;; 64) B=10000; S=10000; I=10;; esac
    job potts3_L$L --model potts --q 3 --dims ${L}x${L} --init all --value 0 --task stationary --observable order \
        --replicas 10 --burn-in $B --samples $S --interval $I \
        --sweep temperature=0.98,0.985,0.99,0.994973,1.0,1.005,1.01 --threads $TH --seed $((50 + L))
done

# ---- B. 2D Ising, Metropolis (T_c = 2.269185): scans for the crossings, then
# long runs at T_c for the universal values (one-point sweeps, for the table).
for L in 16 32 64; do
    case $L in 16) B=2000; S=20000; I=5;; 32) B=5000; S=20000; I=10;; 64) B=10000; S=10000; I=20;; esac
    job ising_L$L --model ising --dims ${L}x${L} --init up --task stationary --observable m --replicas 10 \
        --burn-in $B --samples $S --interval $I --sweep temperature=2.24,2.25,2.26,2.269185,2.28,2.29,2.30 \
        --threads $TH --seed $((60 + L))
done
job ising_L64_Tc --model ising --dims 64x64 --init up --task stationary --observable m --replicas 10 \
    --burn-in 20000 --samples 20000 --interval 40 --sweep temperature=2.269185 --threads $TH --seed 301
job ising_L128_Tc --model ising --dims 128x128 --init up --task stationary --observable m --replicas 10 \
    --burn-in 40000 --samples 10000 --interval 80 --blocks 10 --sweep temperature=2.269185 --threads $TH --seed 188

# ---- A2-A4. voter consensus times on random regular graphs and lattices
job voter_rr4 --model voter --graph regular --k 4 --task ensemble --replicas 300 --steps 300000 \
    --sweep n=500,1000,2000,4000,8000 --threads $TH --seed 71
job voter_2d_check --model voter --task ensemble --replicas 800 --steps 1000000 --sweep dims=32x32 \
    --threads $TH --seed 72
job rfvoter_2d --model rf-voter --task ensemble --replicas 800 --steps 1000000 \
    --sweep dims=16x16,32x32,64x64 --threads $TH --seed 73
job rfvoter_2d_L128 --model rf-voter --task ensemble --replicas 200 --steps 2000000 --sweep dims=128x128 \
    --threads $TH --seed 74
job rfvoter_3d --model rf-voter --task ensemble --replicas 300 --steps 1000000 \
    --sweep dims=8x8x8,12x12x12,16x16x16,20x20x20 --threads $TH --seed 75

echo "done" | tee -a "$LOG"
