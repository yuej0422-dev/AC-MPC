#!/usr/bin/env bash
# Automatically enforce dependencies: shared -> R0-R3 screen -> clean winner.
set -euo pipefail

repo=/root/autodl-tmp/AC-MPC
base=$repo/runs/o2o/diagnostics/E7_full_capacity_online_residual_20260903
launcher=$repo/manisoft_port/scripts/launch_manisoft_circle_E7_fullres.sh
python=$repo/.venv/bin/python
shared_dir=$base/E7_fullres_shared_warmup
shared=$shared_dir/E7_fullres_shared_005000.pt
log=$base/pipeline.log

mkdir -p "$base"
exec >>"$log" 2>&1
echo "[$(date --iso-8601=seconds)] pipeline started"

while tmux has-session -t ms_circle_E7_fullres_shared 2>/dev/null; do sleep 30; done

"$python" - "$shared_dir/latest.pt" <<'PY'
import sys, torch
p=sys.argv[1]
x=torch.load(p,map_location="cpu",weights_only=False)
assert x["phase"] == "online"
assert int(x["online_step"]) == 5000
assert int(x["online_replay"]["size"]) == 5000
assert int(x["learner"]["actor_updates"]) == 0
actor=x["learner"]["actor"]
assert bool(actor["base_source_loaded"].item())
assert torch.count_nonzero(actor["controller.4.weight"]) == 0
assert torch.count_nonzero(actor["controller.4.bias"]) == 0
print("shared snapshot validation passed")
PY
cp --reflink=auto "$shared_dir/latest.pt" "$shared"
chmod 0444 "$shared"
sha256sum "$shared" >"$shared.sha256"
echo "[$(date --iso-8601=seconds)] shared snapshot frozen"

"$launcher" screens
for session in \
  ms_circle_E7_fullres_R0 ms_circle_E7_fullres_R1 \
  ms_circle_E7_fullres_R2 ms_circle_E7_fullres_R3; do
  while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
done

winner=$("$python" - "$base" <<'PY'
import json, pathlib, sys
base=pathlib.Path(sys.argv[1])
runs={
 "R0":base/"R0_frozen_base",
 "R1":base/"R1_fullres_D",
 "R2":base/"R2_fullres_DQ",
 "R3":base/"R3_fullres_DQa",
}
records={}
for name,path in runs.items():
 rows=[json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
 rows=[r for r in rows if r.get("phase")=="online_evaluation" and 5000 <= int(r["online_step"]) <= 7500]
 if len(rows) != 6:
  raise SystemExit(f"{name} has {len(rows)} evaluation rows, expected 6")
 records[name]={"best":max(rows,key=lambda r:r["return_mean"]),"rows":rows}
order=sorted(records,key=lambda n:records[n]["best"]["return_mean"],reverse=True)
winner=order[0]
if len(order)>1 and records[order[0]]["best"]["return_mean"]-records[order[1]]["best"]["return_mean"] < 3:
 rank={"R0":0,"R1":1,"R2":2,"R3":3}
 eligible=[n for n in order if records[order[0]]["best"]["return_mean"]-records[n]["best"]["return_mean"] < 3]
 winner=min(eligible,key=lambda n:rank[n])
if winner=="R3" and records["R3"]["best"]["return_mean"]-records["R2"]["best"]["return_mean"] < 3:
 winner="R2"
summary={"winner":winner,"ranking":order,"best":{n:records[n]["best"] for n in records}}
(base/"screen_selection.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(winner)
PY
)
echo "[$(date --iso-8601=seconds)] screen winner: $winner"
"$launcher" formal "$winner"
echo "[$(date --iso-8601=seconds)] formal launched"
while tmux has-session -t ms_circle_E7_fullres_formal 2>/dev/null; do sleep 30; done

formal=$base/E7_fullres_formal_${winner}
test -f "$formal/online_010000.pt"
"$python" "$repo/manisoft_port/scripts/analyze_manisoft_circle_E7_fullres.py" \
  --root "$base" \
  --source "$repo/runs/o2o/diagnostics/no_xref_authority_refinement_20260902/runs/E7_D500_Q18/best_return.pt" \
  --output "$repo/ManiSoft_circle_E7_full_capacity_online_residual.md"
echo "[$(date --iso-8601=seconds)] formal complete and report written"
