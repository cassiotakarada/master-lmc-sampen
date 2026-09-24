import numpy as np, torch, torchvision, sys, time, json, os
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP); sys.path.insert(0, os.getcwd())
from scipy.spatial import cKDTree
from src.weights import build_param_type_map, param_group

m = torchvision.models.resnet18(weights=None, num_classes=9)
ptmap = build_param_type_map(m)
sd = torch.load('results/runs/PathMNIST_100ep_seed42/checkpoints/resume_epoch_100.pt',
                map_location='cpu', weights_only=False)
if isinstance(sd, dict):
    for k in ('model_state','state_dict','model'):
        if k in sd: sd = sd[k]; break
x = np.concatenate([t.detach().cpu().numpy().astype(np.float64).ravel()
                    for n,t in sd.items()
                    if isinstance(t, torch.Tensor) and param_group(n, ptmap.get(n,'Other'))=='conv'])
N = len(x); r = 0.2*float(np.std(x)); n_tpl = N-2
print(f'n={N:,}  r={r:.6g}  sd={np.std(x):.6g}  curtose={float(((x-x.mean())**4).mean()/x.var()**2):.1f}', flush=True)

def pares(length):
    idx = np.arange(length)[None,:] + np.arange(n_tpl)[:,None]
    tpl = np.ascontiguousarray(x[idx])
    t0 = time.perf_counter()
    tree = cKDTree(tpl)
    print(f'  m={length}: arvore construida em {time.perf_counter()-t0:.0f}s', flush=True)
    t1 = time.perf_counter()
    c = tree.count_neighbors(tree, r, p=np.inf) - n_tpl
    print(f'  m={length}: {c:,} pares em {time.perf_counter()-t1:.0f}s', flush=True)
    return c

t0 = time.perf_counter()
B = pares(2); A = pares(3)
v = float(-np.log(A/B)); dt = time.perf_counter()-t0
print(f'SampEn POPULACAO COMPLETA = {v:.4f}   (subamostrada = 2.0109)', flush=True)
print(f'tempo total: {dt/60:.1f} min', flush=True)
json.dump({'n':int(N),'A':int(A),'B':int(B),'sampen_full':v,'segundos':dt},
          open(os.path.join(SP,'sampen_full_conv.json'),'w'), indent=2)
print('FIM', flush=True)
