import torch
import torch.nn as nn
from torch_geometric.nn import fps, knn, knn_interpolate
from torch_geometric.utils import scatter, to_dense_batch

class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=8):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp_out = nn.Sequential(
            nn.Linear(in_channels * self.k, out_channels),
        )

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)
        
        ids, msk = to_dense_batch(id_k_neighbor[1], id_k_neighbor[0])

        # Max pool onto each cluster the features from knn in points
        D = x.shape[-1]
        x_out = x[ids]
        x_out[~msk] = 0
        x_out = x_out.reshape(-1, self.k * D)
        x_out = self.mlp_out(x_out)
        
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch
    

class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        """
        x: [B, N, d_x]
        x_sub: [B, N_, d_x_in]
        pos: [B, N, d_p]
        pos_sub: [B, N_, d_p]
        """
        # transform low-res features and reduce the number of features
        B, N_, d_x_i = x_sub.shape
        x_sub = self.mlp_sub(x_sub.reshape(B * N_, d_x_i)).reshape(B, N_, -1)

        d_x = x_sub.shape[-1]
        B, N, d_p = pos.shape
        # interpolate low-res feats to high-res points
        with torch.no_grad():
            dist = square_distance(pos, pos_sub)  # [B, N, N']
            v, idx = torch.topk(dist, 3, dim=-1, largest=False)  # [B, N, 3]
            w = 1.0 / v  # [B, N, 3]
        y = torch.gather(x_sub.unsqueeze(1).repeat(1, N, 1, 1), 2, idx.unsqueeze(-1).repeat(1, 1, 1, d_x))  # [B, N, 3, d_x]
        y = y * w.unsqueeze(-1)  # [B, N, 3, d_x]
        yy = y.sum(dim=2)  # [B, N, d_x]
        ww = w.sum(dim=2)
        x_interpolated = yy / ww.unsqueeze(-1) # [B, N, d_x]

        x_ = self.mlp(x_interpolated.reshape(B*N, -1)).reshape(B, N, -1)

        return x_


def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-16, max=None)
    return dist


def sort_and_gather(x):
    B, N, d = x.shape
    y = torch.gather(x, 1, x[..., 0].sort()[1].unsqueeze(-1).repeat(1,1,d))
    return y