#!/usr/bin/env bash
# Dependency-safe orchestration for the E7 actor-strength experiment.
set -euo pipefail

repo=/root/autodl-tmp/AC-MPC
base=$repo/runs/o2o/diagnostics/E7_online_actor_strength_20260903
shared_dir=$base/E7_online_shared_warmup
shared_snapshot=$shared_dir/E7_online_shared_warmup_005000.pt
source=$repo/runs/o2o/diagnostics/no_xref_authority_refinement_20260902/runs/E7_D500_Q18/best_return.pt
launcher=$repo/manisoft_port/scripts/launch_manisoft_circle_E7_online_actor_strength.sh
python=$repo/.venv/bin/python
log=$base/pipeline.log

mkdir -p "$base"
exec >>"$log" 2>&1
echo "[$(date --iso-8601=seconds)] pipeline watcher started"

while tmux has-session -t ms_circle_E7_shared 2>/dev/null; do
  sleep 30
done

"$python" - "$shared_dir/latest.pt" "$source" <<'PY'
import sys, torch
shared_path, source_path = sys.argv[1:]
shared = torch.load(shared_path, map_location="cpu", weights_only=False)
source = torch.load(source_path, map_location="cpu", weights_only=False)
assert shared["phase"] == "online"
assert int(shared["online_step"]) == 5000
assert int(shared["online_replay"]["size"]) == 5000
assert int(shared["learner"]["actor_updates"]) == 0
a = shared["learner"]["actor"]
b = source["learner"]["actor"]
assert a.keys() == b.keys()
assert all(torch.equal(a[key], b[key]) for key in a)
print("shared validation passed: step=5000 replay=5000 actor_updates=0 actor=source")
PY

cp --reflink=auto "$shared_dir/latest.pt" "$shared_snapshot"
chmod 0444 "$shared_snapshot"
sha256sum "$shared_snapshot" >"$shared_snapshot.sha256"
echo "[$(date --iso-8601=seconds)] shared snapshot frozen"

"$launcher" screens
echo "[$(date --iso-8601=seconds)] G0-G3 launched"
