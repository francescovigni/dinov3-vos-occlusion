import torch

from dvos.model import MemoryBank, MemoryVOS, bce_dice, readout, to_feature_res, track


def test_memory_bank_keeps_first_entry():
    mem = MemoryBank(max_size=3)
    for i in range(6):
        mem.add(torch.full((1, 4, 2, 2), float(i)), torch.full((1, 8, 2, 2), float(i)))
    k, v = mem.read()
    assert len(mem) == 3 and k.shape == (1, 4, 12) and v.shape == (1, 8, 12)
    assert k[0, 0, 0].item() == 0.0 and k[0, 0, -1].item() == 5.0
    fifo = MemoryBank(max_size=3, keep_first=False)
    for i in range(6):
        fifo.add(torch.full((1, 4, 2, 2), float(i)), torch.full((1, 8, 2, 2), float(i)))
    assert fifo.read()[0][0, 0, 0].item() == 3.0


def test_readout_topk_matches_full_when_topk_large():
    torch.manual_seed(0)
    qk, mk, mv = torch.randn(1, 4, 6), torch.randn(1, 4, 10), torch.randn(1, 8, 10)
    assert torch.allclose(readout(qk, mk, mv, topk=None), readout(qk, mk, mv, topk=10), atol=1e-6)
    assert readout(qk, mk, mv, topk=3).shape == (1, 8, 6)


def test_forward_shapes_and_track_gating():
    torch.manual_seed(0)
    model = MemoryVOS(c_in=16, c_key=8, c_value=16, hidden=16, readout_topk=4).eval()
    T, h, w = 5, 6, 8
    feats = torch.randn(T, 16, h, w)
    first = torch.zeros(1, 1, 4 * h, 4 * w)
    first[..., 4:12, 8:20] = 1.0
    k, v = model.encode(feats[0:1], to_feature_res(first, (h, w)))
    out = model(feats[1:2], k.flatten(2), v.flatten(2))
    assert out["mask_logits"].shape == (1, 1, 4 * h, 4 * w) and out["vis_logit"].shape == (1, 1)
    probs, vis = track(model, feats, first, memory_max=4, vis_gate=0.5)
    assert len(probs) == T and len(vis) == T and vis[0] == 1.0
    probs_gated, _ = track(model, feats, first, memory_max=4, vis_gate=1.5)
    assert probs_gated[1].shape == probs[1].shape


def test_bce_dice_finite_and_zero_at_perfect():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1.0
    loss = bce_dice(torch.where(target > 0, 20.0, -20.0), target)
    assert torch.isfinite(loss) and loss.item() < 0.15


def test_sample_indices_respects_gaps_and_bounds():
    import random

    from dvos.train import sample_indices

    rng = random.Random(0)
    for _ in range(200):
        idx = sample_indices(n=40, length=12, gap_max=3, rng=rng)
        assert len(idx) == 12 and idx[0] >= 0 and idx[-1] <= 39
        assert all(1 <= b - a <= 3 for a, b in zip(idx[:-1], idx[1:], strict=True))
    short = sample_indices(n=5, length=12, gap_max=3, rng=rng)
    assert short == [0, 1, 2, 3, 4]
