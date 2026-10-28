# Probe-only helper: source me. run NAME cmd... logs to logs/NAME.log and records the exit code.
mkdir -p logs
run() { local n=$1; shift; echo "=== $* ==="; date; "$@" 2>&1 | tee -a "logs/$n.log"; local rc=${PIPESTATUS[0]}; echo "EXIT $rc for $n"; echo "$n $rc" >> logs/exitcodes.txt; date; return 0; }
U=http://127.0.0.1:11435/v1
A="$RUNNER_TEMP/agent"
FLAGS="--backend podman --openai-base-url $U --llm-timeout 2400"
