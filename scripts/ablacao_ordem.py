"""Ablação D2/D3/bins no checkpoint final — teste da pergunta do orientador."""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from src.weights import build_param_type_map, param_group
from src.complexity import lmc_complexity, sample_entropy_1d
import torchvision

RUNS = ["PathMNIST_100ep_seed42", "PathMNIST_100ep_seed123", "PathMNIST_100ep_seed2024"]
CKPT = "results/runs/{r}/checkpoints/resume_epoch_100.pt"

m = torchvision.models.resnet18(weights=None, num_classes=9)
ptmap = build_param_type_map(m)

def groups_from_sd(sd):
    buckets = {}
    for name, t in sd.items():
        if not isinstance(t, torch.Tensor) or t.numel() == 0: continue
        g = param_group(name, ptmap.get(name, 'Other'))
        if g in ('other', 'bias'): continue
        buckets.setdefault(g, []).append((name, t))
    return buckets

def flat_concat(items, transpose_conv=False):
    arrs = []
    for name, t in items:
        a = t.detach().cpu().numpy().astype(np.float64)
        if transpose_conv and a.ndim == 4:
            a = a.transpose(1, 0, 2, 3)   # [C_in, C_out, H, W]
        arrs.append(a.ravel())
    return np.concatenate(arrs)

def sub_random(a, k, seed=42):
    return np.random.default_rng(seed).choice(a, size=k, replace=False)

def sub_systematic(a, k):
    step = max(1, len(a) // k)
    return a[::step][:k]

out = []
for run in RUNS:
    sd = torch.load(CKPT.format(r=run), map_location='cpu', weights_only=False)
    sd = sd.get('model_state', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    b = groups_from_sd(sd)
    conv = flat_concat(b['conv'])
    conv_T = flat_concat(b['conv'], transpose_conv=True)
    gamma = flat_concat(b['bn_gamma'])
    row = {'run': run}
    # --- D3: aleatório (código atual) vs sistemático (o que a tese diz) ---
    for tag, s in [('rand_seed42', sub_random(conv, 10000, 42)),
                   ('rand_seed7',  sub_random(conv, 10000, 7)),
                   ('sistematico', sub_systematic(conv, 10000))]:
        row[f'conv_sampen_{tag}'] = round(float(sample_entropy_1d(s)), 4)
        row[f'conv_lmc_{tag}']    = round(float(lmc_complexity(s)['complexity']), 4)
    # --- D2: transposto vs nativo, com subamostragem sistemática ---
    sT = sub_systematic(conv_T, 10000)
    row['conv_sampen_sistematico_TRANSPOSTO'] = round(float(sample_entropy_1d(sT)), 4)
    row['conv_lmc_sistematico_TRANSPOSTO']    = round(float(lmc_complexity(sT)['complexity']), 4)
    # --- bn_gamma: nao subamostrado (n=4800) -> ordem intacta ---
    row['gamma_sampen'] = round(float(sample_entropy_1d(gamma)), 4)
    row['gamma_lmc']    = round(float(lmc_complexity(gamma)['complexity']), 4)
    # --- sensibilidade a bins (LMC) em bn_gamma ---
    for nb in (20, 50, 100, 200):
        row[f'gamma_lmc_bins{nb}'] = round(float(lmc_complexity(gamma, n_bins=nb)['complexity']), 4)
    out.append(row)
    print(json.dumps(row, indent=2), flush=True)

json.dump(out, open('/tmp/claude-15919/-home-users-u7594034--rea-de-trabalho-Masters-Claude/462e9c77-a67c-4003-a6d0-7d7f22d4ea03/scratchpad/ablacao_ordem.json','w'), indent=2)
print("OK")
