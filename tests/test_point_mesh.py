import torch
import torch_musa

from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, packed_to_list, Pointclouds

def get_random_musa_device() -> str:
    """
    Function to get a random GPU device from the
    available devices. This is useful for testing
    that custom cuda kernels can support inputs on
    any device without having to set the device explicitly.
    """
    num_devices = torch.musa.device_count()
    device_id = (
        torch.randint(high=num_devices, size=(1,)).item() if num_devices > 1 else 0
    )
    return "musa:%d" % device_id
    # return "cpu"
print(get_random_musa_device())

def init_meshes_clouds(
        batch_size: int = 10,
        num_verts: int = 1000,
        num_faces: int = 3000,
        num_points: int = 3000,
        device: str = "cpu",
    ):
        device = torch.device(device)
        nump = torch.randint(low=1, high=num_points, size=(batch_size,))
        numv = torch.randint(low=3, high=num_verts, size=(batch_size,))
        numf = torch.randint(low=1, high=num_faces, size=(batch_size,))
        verts_list = []
        faces_list = []
        points_list = []
        for i in range(batch_size):
            # Randomly choose vertices
            verts = torch.rand((numv[i], 3), dtype=torch.float32, device=device)
            verts.requires_grad_(True)

            # Randomly choose faces. Our tests below compare argmin indices
            # over faces and edges. Argmin is sensitive even to small numeral variations
            # thus we make sure that faces are valid
            # i.e. a face f = (i0, i1, i2) s.t. i0 != i1 != i2,
            # otherwise argmin due to numeral sensitivities cannot be resolved
            faces, allf = [], 0
            validf = numv[i].item() - numv[i].item() % 3
            while allf < numf[i]:
                ff = torch.randperm(numv[i], device="cpu")[:validf].view(-1, 3).to(device)
                faces.append(ff)
                allf += ff.shape[0]
            faces = torch.cat(faces, 0)
            if faces.shape[0] > numf[i]:
                faces = faces[: numf[i]]

            verts_list.append(verts)
            faces_list.append(faces)

            # Randomly choose points
            points = torch.rand((nump[i], 3), dtype=torch.float32, device=device)
            points.requires_grad_(True)

            points_list.append(points)

        meshes = Meshes(verts_list, faces_list)
        pcls = Pointclouds(points_list)

        return meshes, pcls


def test_point_mesh_edge_distance():
    """
    Test point_mesh_edge_distance from pytorch3d.loss
    """
    device = get_random_musa_device()
    N, V, F, P = 4, 32, 16, 24
    meshes, pcls = init_meshes_clouds(N, V, F, P, device=device)

    # clone and detach for another backward pass through the op
    verts_op = [verts.clone().detach() for verts in meshes.verts_list()]
    for i in range(N):
        verts_op[i].requires_grad = True

    faces_op = [faces.clone().detach() for faces in meshes.faces_list()]
    meshes_op = Meshes(verts=verts_op, faces=faces_op)
    points_op = [points.clone().detach() for points in pcls.points_list()]
    for i in range(N):
        points_op[i].requires_grad = True
    pcls_op = Pointclouds(points_op)

    # Cuda implementation: forward & backward
    loss_op = point_mesh_edge_distance(meshes_op, pcls_op)

    # Naive implementation: forward & backward
    edges_packed = meshes.edges_packed()
    edges_list = packed_to_list(edges_packed, meshes.num_edges_per_mesh().tolist())
    loss_naive = torch.zeros(N, dtype=torch.float32, device=device)
    for i in range(N):
        points = pcls.points_list()[i]
        verts = meshes.verts_list()[i]
        v_first_idx = meshes.mesh_to_verts_packed_first_idx()[i]
        edges = verts[edges_list[i] - v_first_idx]

        num_p = points.shape[0]
        num_e = edges.shape[0]
        dists = torch.zeros((num_p, num_e), dtype=torch.float32, device=device)
        print(dists)
    #     for p in range(num_p):
    #         for e in range(num_e):
    #             dist = _point_to_edge_distance(points[p], edges[e])
    #             dists[p, e] = dist

    #     min_dist_p, min_idx_p = dists.min(1)
    #     min_dist_e, min_idx_e = dists.min(0)

    #     loss_naive[i] = min_dist_p.mean() + min_dist_e.mean()
    # loss_naive = loss_naive.mean()

    # # NOTE that hear the comparison holds despite the discrepancy
    # # due to the argmin indices returned by min(). This is because
    # # we don't will compare gradients on the verts and not on the
    # # edges or faces.

    # # Compare forward pass
    # self.assertClose(loss_op, loss_naive)

    # # Compare backward pass
    # rand_val = torch.rand(1).item()
    # grad_dist = torch.tensor(rand_val, dtype=torch.float32, device=device)

    # loss_naive.backward(grad_dist)
    # loss_op.backward(grad_dist)

    # # check verts grad
    # for i in range(N):
    #     self.assertClose(
    #         meshes.verts_list()[i].grad, meshes_op.verts_list()[i].grad
    #     )
    #     self.assertClose(pcls.points_list()[i].grad, pcls_op.points_list()[i].grad)

test_point_mesh_edge_distance()