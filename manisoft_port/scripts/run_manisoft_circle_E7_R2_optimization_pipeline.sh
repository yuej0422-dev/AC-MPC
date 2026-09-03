#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/AC-MPC
python=$repo/.venv/bin/python
base=$repo/runs/o2o/diagnostics/E7_R2_online_optimization_20260903
launcher=$repo/manisoft_port/scripts/launch_manisoft_circle_E7_R2_optimization.sh
source_dir=$repo/runs/o2o/diagnostics/E7_full_capacity_online_residual_20260903/E7_fullres_formal_R2
milestone=$source_dir/online_010000.pt
recovery=$source_dir/latest.pt
source=$base/source/R2_online_010000_full.pt
log=$base/pipeline.log

mkdir -p "$base/source"
exec >>"$log" 2>&1
echo "[$(date --iso-8601=seconds)] pipeline started"

"$python" - "$milestone" "$recovery" "$base/source/source_identity.json" <<'PY'
import hashlib,json,sys,torch,pathlib
milestone,recovery,out=map(pathlib.Path,sys.argv[1:])
a=torch.load(milestone,map_location="cpu",weights_only=False)
b=torch.load(recovery,map_location="cpu",weights_only=False)
assert a["phase"]==b["phase"]=="online"
assert int(a["online_step"])==int(b["online_step"])==10000
assert a["online_replay"] is None
assert int(b["online_replay"]["size"])==10000

def digest(value):
 h=hashlib.sha256()
 def add(x):
  if isinstance(x,torch.Tensor):
   y=x.detach().cpu().contiguous().numpy(); h.update(str(y.dtype).encode()); h.update(str(y.shape).encode()); h.update(y.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x): h.update(str(k).encode()); add(x[k])
  elif isinstance(x,(list,tuple)):
   for y in x:add(y)
  else:h.update(repr(x).encode())
 add(value); return h.hexdigest()
for key in ("learner","rng"):
 assert digest(a[key])==digest(b[key]),key
identity={
 "milestone":str(milestone),"recovery":str(recovery),
 "milestone_learner_sha256":digest(a["learner"]),
 "recovery_learner_sha256":digest(b["learner"]),
 "rng_sha256":digest(b["rng"]),
 "online_step":10000,"online_replay_size":10000,
}
out.write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n")
print("10k milestone/recovery identity verified")
PY

if [[ ! -f "$source" ]]; then
  cp --reflink=auto "$recovery" "$source"
  chmod 0444 "$source"
  sha256sum "$source" >"$source.sha256"
fi

if [[ "${SKIP_SCREEN_LAUNCH:-0}" != 1 ]]; then
  "$launcher" screens
else
  echo "[$(date --iso-8601=seconds)] screen launch skipped; waiting for existing C0-C3 sessions"
fi
for session in ms_circle_E7_R2_C0 ms_circle_E7_R2_C1 ms_circle_E7_R2_C2 ms_circle_E7_R2_C3; do
  while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
done

winner=$("$python" - "$base" <<'PY'
import json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1])
runs={"C0":"C0_R2_lr1e6_int4","C1":"C1_R2_lr2e6_int4","C2":"C2_R2_lr1e6_int2","C3":"C3_R2_lr2e6_int2"}
info={}
for name,directory in runs.items():
 rows=[json.loads(x) for x in (root/directory/"metrics.jsonl").read_text().splitlines()]
 rows=[x for x in rows if x.get("phase")=="online_evaluation" and 10000<=int(x["online_step"])<=12500]
 assert [int(x["online_step"]) for x in rows]==[10000,10500,11000,11500,12000,12500]
 xs=[x["online_step"]/1000 for x in rows]; ys=[x["return_mean"] for x in rows]
 xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
 linear=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
 robust=statistics.median((ys[j]-ys[i])/(xs[j]-xs[i]) for i in range(len(xs)) for j in range(i+1,len(xs)))
 info[name]={"rows":rows,"best":max(rows,key=lambda x:x["return_mean"]),"linear_slope_per_1k":linear,"robust_slope_per_1k":robust,"tail_slope_per_1k":2*(ys[-1]-ys[-2])}
order=sorted(info,key=lambda n:info[n]["best"]["return_mean"],reverse=True)
winner=order[0]
if info[order[0]]["best"]["return_mean"]-info[order[1]]["best"]["return_mean"]<3:
 candidates=[n for n in order if info[order[0]]["best"]["return_mean"]-info[n]["best"]["return_mean"]<3]
 rising=[n for n in candidates if info[n]["tail_slope_per_1k"]>0]
 if rising:candidates=rising
 conservative={"C3":0,"C0":1,"C1":2,"C2":2}
 winner=min(candidates,key=lambda n:(info[n]["best"].get("implicit_policy_kl_from_shared_mean",0),conservative[n]))
summary={"winner":winner,"ranking":order,"groups":{n:{k:v for k,v in info[n].items() if k!="rows"} for n in info}}
(root/"screen_selection.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(winner)
PY
)
echo "[$(date --iso-8601=seconds)] screen winner: $winner"
"$launcher" formal "$winner"
while tmux has-session -t ms_circle_E7_R2_formal 2>/dev/null; do sleep 30; done

formal=$base/E7_R2_online_extended_${winner}
test -f "$formal/online_020000.pt"
"$python" "$repo/manisoft_port/scripts/analyze_manisoft_circle_E7_R2_optimization.py" \
  --root "$base" --winner "$winner" \
  --output "$repo/ManiSoft_circle_E7_R2_online_optimization.md"
echo "[$(date --iso-8601=seconds)] formal complete and report written"
