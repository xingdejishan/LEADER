import json, numpy as np
rows=[json.loads(l) for l in open("/data2/zhangcy/dsh-angle-20261024/proj/results/pilot40.jsonl")]
roles={r["key"]:r["role"] for r in json.load(open("/data2/zhangcy/dsh-angle-20261024/pilot_roles.json"))}
TR="/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth/%s.json"
truth={}
def gt(k):
    d,s=k.split("/")
    if d not in truth: truth[d]=json.load(open(TR%d))
    return np.asarray(truth[d][s]["T_WB"],float)
def err(P,T):
    er=float(np.degrees(np.arccos(np.clip((np.trace(P[:3,:3].T@T[:3,:3])-1)/2,-1,1))))
    et=float(np.linalg.norm(P[:3,3]-T[:3,3]))
    return et,er
hi=[]; ct=[]
for r in rows:
    T=gt(r["key"]); t0=np.asarray(r["T0"],float); fi=np.asarray(r["T"],float)
    ge=np.asarray(r["T_geometry"],float)
    vi=None if r["T_visual"] is None else np.asarray(r["T_visual"],float)
    arb=r.get("arbitration") or {}
    o=dict(key=r["key"],e0=err(t0,T)[1],ef=err(fi,T)[1],eg=err(ge,T)[1],
           ev=None if vi is None else err(vi,T)[1],
           tv=None if vi is None else err(vi,T)[0],
           c3=(arb.get("costs") or {}).get("route3"),cv=(arb.get("costs") or {}).get("visual"),
           thr=arb.get("thresh"),chosen=arb.get("chosen"))
    (hi if roles[r["key"]]=="high" else ct).append(o)
ap=[o for o in hi if o["ev"] is not None]
better=[o for o in ap if o["ev"]<=o["e0"]-0.5]; worse=[o for o in ap if o["ev"]>=o["e0"]+0.5]
print("high n=%d, with visual=%d, no visual=%d"%(len(hi),len(ap),len(hi)-len(ap)))
print("visual rotation better by >0.5deg: %d  (median gain %.2f)"%(len(better),np.median([o["e0"]-o["ev"] for o in better]) if better else 0))
print("visual rotation worse by >0.5deg: %d  (median loss %.2f)"%(len(worse),np.median([o["ev"]-o["e0"] for o in worse]) if worse else 0))
print("visual adopted:",sum(1 for o in hi if o["chosen"]=="visual"),"geometry kept:",sum(1 for o in hi if o["chosen"]=="route3"))
print("visual translation <=1m:",sum(1 for o in ap if o["tv"]<=1.0),"<=2m:",sum(1 for o in ap if o["tv"]<=2.0))
print("among the %d better ones, translation <=1m: %d"%(len(better),sum(1 for o in better if o["tv"]<=1.0)))
gap=[o["cv"]-o["c3"] for o in ap if o["c3"] is not None and o["cv"] is not None]
print("cost gap cv-c3: median %.3f, negative (visual cheaper) %d/%d"%(np.median(gap),sum(1 for g in gap if g<0),len(gap)))
print("decided(has thresh):",sum(1 for o in ap if o["thr"] is not None))
print("ctrl unchanged:",sum(1 for o in ct if abs(o["ef"]-o["e0"])<0.05),"/",len(ct))
