import torch

from dvos.metrics import jaccard
from dvos.propagate import neighborhood_mask, propagate


def test_neighborhood_mask_counts():
    nb = neighborhood_mask(5, 5, radius=1)
    assert nb.shape == (25, 25)
    assert nb[12].sum() == 9 and nb[0].sum() == 4


def moving_square(T=6, h=16, w=16, C=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats, masks = [], []
    for t in range(T):
        m = torch.zeros(h, w, dtype=torch.bool)
        m[4:10, 2 + t : 8 + t] = True
        f = 0.05 * torch.randn(C, h, w, generator=g)
        f[0] += m.float() * 2.0
        f[1] += (~m).float() * 2.0
        feats.append(f)
        masks.append(m)
    return feats, masks


def test_propagation_follows_a_moving_square():
    feats, masks = moving_square()
    labels0 = torch.stack([(~masks[0]).float(), masks[0].float()])
    out = propagate(feats, labels0, n_last=3, topk=5, radius=2, temperature=0.07)
    assert len(out) == len(feats) and out[-1].shape == (2, 16, 16)
    pred = out[-1].argmax(0).bool().numpy()
    assert jaccard(pred, masks[-1].numpy()) > 0.8
