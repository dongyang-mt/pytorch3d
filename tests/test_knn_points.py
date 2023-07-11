import torch
import torch_musa
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

def test_knn_gather(device = torch.device("cpu")):
    N, P1, P2, K, D = 4, 16, 12, 8, 3
    x = torch.rand((N, P1, D), device=device)
    y = torch.rand((N, P2, D), device=device)
    lengths1 = torch.randint(low=1, high=P1, size=(N,), device=device)
    lengths2 = torch.randint(low=1, high=P2, size=(N,), device=device)
    print("input lengths1: ", lengths1)
    out = knn_points(x, y, lengths1=lengths1, lengths2=lengths2, K=K)

    x_musa = x.to("musa")
    y_musa = y.to("musa")
    lengths1_musa = lengths1.to("musa")
    lengths2_musa = lengths2.to("musa")
    out_musa = knn_points(x_musa, y_musa, lengths1=lengths1_musa, lengths2=lengths2_musa, K=K)
    print("out_musa: ", out_musa)
    print(torch.allclose(out.dists, out_musa.dists.cpu()))
    print(torch.allclose(out.idx, out_musa.idx.cpu()))

test_knn_gather()
