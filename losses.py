import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../IcosahedralCNN'))
from icocnn.utils.ico_geometry import get_ico_faces
sys.path.append(os.path.join(os.path.dirname(__file__),'../PythonFunctions'))
from mesh.utils import compute_vertex_normals, compute_laplacian_batch, compute_adjacency_matrix_sparse

""" Base Loss """
class Point2Point_Loss(torch.nn.Module):
    def __init__(self, subdivisions, factor_pos, factor_nor, factor_lap):
        super(Point2Point_Loss, self).__init__()

        self.factor_pos = factor_pos
        self.factor_nor = factor_nor
        self.factor_lap = factor_lap

        self.loss_posMSE = torch.nn.MSELoss()
        self.loss_norCos = torch.nn.CosineSimilarity(dim=2)
        self.loss_lap = torch.nn.MSELoss()

        # index buffers for icosahedral corner averaging
        base_height = 2 ** subdivisions
        top_corner_src_y = torch.arange(5) * base_height
        top_corner_src_x = torch.tensor([0])
        bottom_corner_src_y = torch.arange(1, 6) * base_height - 1
        bottom_corner_src_x = torch.tensor([-1])
        corner_src_y = torch.stack((top_corner_src_y, bottom_corner_src_y))
        corner_src_x = torch.stack((top_corner_src_x, bottom_corner_src_x))
        self.register_buffer('corner_src_y', corner_src_y)
        self.register_buffer('corner_src_x', corner_src_x)

        # faces
        ico_faces = get_ico_faces(subdivisions)
        self.register_buffer('ico_faces', torch.from_numpy(ico_faces))

        # adjacency matrix
        vert_len = ico_faces.max() + 1
        adj_mat = compute_adjacency_matrix_sparse(vert_len, self.ico_faces)
        self.register_buffer('adj_mat', adj_mat)

        self.last_loss_mse = 0
        self.last_loss_cos = 0
        self.last_loss_lap = 0
        self.last_loss_total = 0

    def forward(self, inputs, target):
        # compute pole vertices
        v_tmp = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        v_poles = torch.mean(inputs[:, :, self.corner_src_y, self.corner_src_x], -1)
        v = torch.cat((v_tmp, v_poles), dim=2).transpose(1, 2).contiguous()

        # compute vertex normals
        v_normals = compute_vertex_normals(v, self.ico_faces)

        # compute vertex Laplacians
        lap = compute_laplacian_batch(v, self.adj_mat)

        # prepare targets
        tmp_targets = target.transpose(1, 2).contiguous()

        data_v = v
        data_normals = v_normals
        data_lap = lap

        targets_v = tmp_targets[:, :, :3]
        targets_normals = tmp_targets[:, :, 3:6]
        targets_lap = tmp_targets[:, :, 6:9]

        # compute loss
        l_posMSE = self.loss_posMSE(data_v, targets_v)
        self.last_loss_mse = l_posMSE.item()

        l_norCos = torch.mean(1 - self.loss_norCos(data_normals, targets_normals))
        self.last_loss_cos = l_norCos

        l_lap = self.loss_lap(data_lap, targets_lap)
        self.last_loss_lap = l_lap

        loss = self.factor_pos * l_posMSE + self.factor_nor * l_norCos + self.factor_lap * l_lap
        self.last_loss_total = loss.item()
        return loss

    def get_last_losses(self):
        return self.last_loss_mse, self.last_loss_cos, self.last_loss_lap, self.last_loss_total

class KLD_Loss(torch.nn.Module):
    # KL divergence losses summed over all elements and batch
    def __init__(self):
        super(KLD_Loss, self).__init__()

    def forward(self, output, target):
        output, mu, logvar = output
        mu = torch.flatten(mu, start_dim=1)
        logvar = torch.flatten(logvar, start_dim=1)
        if mu.dim() == 2:
            dim = (1)
        else:
            dim = (1,2,3)
        if self.factor_kl:
	        # see Appendix B from VAE paper:
	        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	        # https://arxiv.org/abs/1312.6114
	        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            self.loss = torch.mean(-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
        else:
            self.loss = torch.tensor(0.)
        return self.loss

    def get_last_losses(self):
        return 0, 0, 0, 0, -self.loss.item()

    def get_factor(self):
        return self.factor_kl

    def update_factor(self, epoch, factor_step_size, factor_gamma):
        if epoch % factor_step_size == 0:
            self.factor_kl *= factor_gamma

""" Dervied Loss """
class P2P_Loss(Point2Point_Loss):
    def __init__(self, subdivisions, factor_pos, factor_nor, factor_lap):
        super(P2P_Loss, self).__init__(subdivisions, factor_pos, factor_nor, factor_lap)

    def forward(self, input, target):
        return Point2Point_Loss.forward(self, input, target)

    def get_last_losses(self):
        return self.last_loss_mse, self.last_loss_cos.item(), self.last_loss_lap.item(), 0., self.last_loss_total

class P2PKLD_Loss(P2P_Loss, KLD_Loss):
    # p2p + kld loss
    def __init__(self, subdivisions, factor_pos, factor_nor, factor_lap, factor_kl):
        super(P2PKLD_Loss, self).__init__(subdivisions, factor_pos, factor_nor, factor_lap)
        self.factor_kl = factor_kl

    def forward(self, output, target):
        self.kld_loss = KLD_Loss.forward(self, output, target)
        output, mu, logvar = output
        self.recons_loss = P2P_Loss.forward(self, output, target)
        self.loss = self.recons_loss + self.factor_kl * self.kld_loss
        return self.loss

    def get_last_losses(self):
        return self.recons_loss.item(), 0, 0, -self.kld_loss.item(), self.loss.item()
