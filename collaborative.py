import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from lightglue.lightglue import normalize_keypoints, apply_cached_rotary_emb


def checked(fn, *args):
    if torch.is_grad_enabled() and any(x.requires_grad for x in args):
        return checkpoint(fn, *args)
    return fn(*args)


def masked_self(block, x, encoding, valid):
    count,n,d=x.shape
    qkv=block.Wqkv(x).view(count,n,4,64,3).transpose(1,2)
    q,k,v=qkv[...,0],qkv[...,1],qkv[...,2]
    q=apply_cached_rotary_emb(encoding,q)
    k=apply_cached_rotary_emb(encoding,k)
    sim=(q@k.transpose(-1,-2))/8
    sim=sim.masked_fill(~valid[:,None,None,:],-1e9)
    context=sim.softmax(-1)@v
    message=block.out_proj(context.transpose(1,2).reshape_as(x))
    return x+block.ffn(torch.cat([x,message],-1))


class Interaction(nn.Module):
    def __init__(self, template):
        super().__init__()
        self.source = copy.deepcopy(template.self_attn)
        self.target = copy.deepcopy(template.self_attn)
        self.confidence = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.register_buffer("source_gate", torch.ones(()))
        self.register_buffer("target_gate", torch.ones(()))
        self.register_buffer("propagation_gate", torch.ones(()))

    def source_message(self, qi, kj, vj, encoding_q, encoding_k, known, key_valid):
        q = apply_cached_rotary_emb(encoding_q, qi)
        k = apply_cached_rotary_emb(encoding_k, kj)
        sim_geo = torch.einsum('bhid,bhjd->bhij', q, k) / 8
        sim_visual = torch.einsum('bhid,bhjd->bhij', qi, kj) / 8
        sim = torch.where(known[None, None, :, None], sim_geo, sim_visual)
        sim=sim.masked_fill(~key_valid[None,None,None,:],-float('inf'))
        return sim.softmax(-1) @ vj

    def forward(self, q, refs, projection, known, valid, posenc, enable_source=True, enable_target=True):
        count, n, d = refs.shape
        if enable_target:
            t = q.transpose(0, 1)
            qkv = self.target.Wqkv(t).view(t.shape[0], count, 4, 64, 3).transpose(1, 2)
            a, b, c = qkv[..., 0], qkv[..., 1], qkv[..., 2]
            similarity = a @ b.transpose(-1, -2) / 8
            if count > 1:
                similarity = similarity.masked_fill(torch.eye(count, device=q.device, dtype=torch.bool)[None, None], -float("inf"))
            attn = similarity.softmax(-1)
            message = self.target.out_proj((attn @ c).transpose(1, 2).reshape_as(t))
            delta = self.target.ffn(torch.cat([t, message], -1)).transpose(0, 1)
            q = q + self.target_gate * delta
        if enable_source:
            qkv = self.source.Wqkv(refs).view(count, n, 4, 64, 3).transpose(1, 2)
            a, b, c = qkv[..., 0], qkv[..., 1], qkv[..., 2]
            messages = []
            for i in range(count):
                context = torch.zeros_like(a[i:i+1])
                for j in range(count):
                    if i == j:
                        continue
                    enc_q = posenc(projection[i, j][None])
                    enc_k = posenc(projection[j, j][None])
                    context = context + checked(self.source_message, a[i:i+1], b[j:j+1], c[j:j+1],
                                                enc_q, enc_k, known[i, j],valid[j])
                context = context / max(1, count - 1)
                messages.append(context)
            message = torch.cat(messages).transpose(1, 2).reshape(count, n, d)
            message = self.source.out_proj(message)
            refs = refs + self.source_gate * self.source.ffn(torch.cat([refs, message], -1))
        return q, refs

    def cross(self, block, q, refs, tracks, valid, layer, enable_propagation=True):
        count, n, _ = refs.shape
        def heads(x):
            return x.view(count, x.shape[1], 4, 64).transpose(1, 2)
        qk, rk = heads(block.to_qk(q)), heads(block.to_qk(refs))
        qv, rv = heads(block.to_v(q)), heads(block.to_v(refs))
        sim = (qk * block.scale ** 0.5) @ (rk * block.scale ** 0.5).transpose(-1, -2)
        sim=sim.masked_fill(~valid[:,None,None,:],-1e9)
        aqr = sim.softmax(-1)
        arq = sim.transpose(-1, -2).softmax(-1).nan_to_num()
        logits = self.confidence(refs.detach()).squeeze(-1)
        confidence = logits.sigmoid()
        if enable_propagation:
            merged = []
            for i in range(count):
                numerator = torch.zeros_like(arq[i])
                denominator = torch.zeros(n, device=q.device, dtype=q.dtype)
                for j in range(count):
                    if i == j:
                        continue
                    ids = tracks[i, j]
                    linked = ids >= 0
                    ids = ids.clamp_min(0)
                    weight = confidence[j, ids] * linked
                    numerator = numerator + arq[j, :, ids, :] * weight[None, :, None]
                    denominator = denominator + weight
                borrowed = numerator / denominator.clamp_min(1e-8)[None, :, None]
                use = (denominator > 0) & valid[i]
                mixed = 0.25 * arq[i] + 0.75 * borrowed
                delta = torch.where(use[None, :, None], mixed-arq[i], torch.zeros_like(mixed))
                merged.append(arq[i] + self.propagation_gate * delta)
            arq = torch.stack(merged)
        mq = (aqr @ rv).transpose(1, 2).reshape_as(q)
        mr = (arq @ qv).transpose(1, 2).reshape_as(refs)
        q = q + block.ffn(torch.cat([q, block.to_out(mq)], -1))
        refs = refs + block.ffn(torch.cat([refs, block.to_out(mr)], -1))
        return q, refs, logits


class CollaborativeV8(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.interactions = nn.ModuleList([Interaction(t) for t in base.transformers])
        self.reference_posenc = copy.deepcopy(base.posenc)

    def forward(self, query, references, geometry, enabled=True, ablate=None):
        count = len(references)
        q = self.base.input_proj(query['descriptors']).expand(count, -1, -1)
        sizes=[x['descriptors'].shape[1] for x in references]
        width=max(sizes)
        r = self.base.input_proj(torch.cat([F.pad(x['descriptors'],(0,0,0,width-size)) for x,size in zip(references,sizes)]))
        qk = normalize_keypoints(query['keypoints'], query['image_size'])
        rk = torch.cat([F.pad(normalize_keypoints(x['keypoints'], x['image_size']),(0,0,0,width-size)) for x,size in zip(references,sizes)])
        qe = self.base.posenc(qk.expand(count, -1, -1))
        re = self.base.posenc(rk)
        confidence_logits, layer_matches = [], []
        for i, (layer, interaction) in enumerate(zip(self.base.transformers, self.interactions)):
            def execute(q0, r0, qe0, re0, ix=i):
                block = self.base.transformers[ix]
                extra = self.interactions[ix]
                q0 = block.self_attn(q0, qe0)
                valid=geometry['valid']
                r0 = masked_self(block.self_attn,r0,re0,valid)
                if enabled:
                    q0, r0 = extra(q0, r0, geometry['projection'], geometry['known'], valid, self.reference_posenc,
                                   ablate != 'source', ablate != 'target')
                    return extra.cross(block.cross_attn, q0, r0, geometry['tracks'], valid, ix, ablate != 'propagation')
                return extra.cross(block.cross_attn,q0,r0,geometry['tracks'],valid,ix,False)
            q, r, logits = checked(execute, q, r, qe, re)
            confidence_logits.append(logits)
            if self.training:
                with torch.no_grad():
                    sim = q @ r.transpose(-1, -2) / 16
                    sim=sim.masked_fill(~geometry['valid'][:,None,:],-1e9)
                    probability=(sim.log_softmax(-1)+sim.log_softmax(-2)).exp()
                    bestq=probability.argmax(1)
                    bestr=probability.argmax(2)
                    reciprocal=bestr.gather(1,bestq)==torch.arange(r.shape[1],device=r.device)[None]
                    valid=reciprocal&(probability.max(1).values>.32)
                    layer_matches.append(torch.where(valid,bestq,bestq.new_full(bestq.shape,-1)))
        all_scores=[]
        for i,size in enumerate(sizes):
            score,_=self.base.log_assignment[-1](q[i:i+1],r[i:i+1,:size])
            if size<width:
                score=torch.cat([score[:,:,:size],score.new_full((1,score.shape[1],width-size),-1e9),score[:,:,-1:]],-1)
            all_scores.append(score)
        scores=torch.cat(all_scores)
        return scores, confidence_logits, layer_matches
