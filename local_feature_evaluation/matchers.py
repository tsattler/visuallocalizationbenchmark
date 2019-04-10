import torch


# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(descriptors1, descriptors2):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()
