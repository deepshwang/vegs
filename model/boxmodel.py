import torch
from utils.graphics_utils import quaternion_to_matrix, matrix_to_quaternion

class BoxModel():
    
    def __init__(self, obj, args):
        self.obj = obj
        self.lr = args.boxmodel_lr
        self.delta_r = torch.tensor([1., 0., 0., 0.], device='cuda', requires_grad=True)
        self.delta_s = torch.tensor([1., 1., 1.], device='cuda', requires_grad=True)
        self.delta_t = torch.tensor([0., 0., 0.], device='cuda', requires_grad=True)
        self.optimizer = torch.optim.Adam([self.delta_r, self.delta_s, self.delta_t], lr=self.lr)
        self.box2world = self.obj_box2world()
        self.lambda_reg = args.boxmodel_lambda_reg

    def obj_box2world(self):
        box2world = torch.eye(4)
        box2world[:3, :3] = torch.from_numpy(self.obj.R).float()
        box2world[:3, 3] = torch.from_numpy(self.obj.T).float()
        box2world = box2world.cuda()
        return box2world

    def scale_to_mat3(self):
        d_s = torch.eye(3).cuda()
        d_s[0, 0] = self.delta_s[0]
        d_s[1, 1] = self.delta_s[1]
        d_s[2, 2] = self.delta_s[2]
        return d_s

    @property
    def d_box2world(self):
        d_box2world = torch.eye(4).cuda()
        d_s_mat = self.scale_to_mat3()
        d_r_mat = quaternion_to_matrix(self.delta_r)
        d_sr_mat = torch.matmul(d_s_mat, d_r_mat)
        d_box2world[:3, :3] = d_sr_mat
        d_box2world[:3, 3] = self.delta_t
        return d_box2world

    def adjustbox2world(self):
        # return self.box2world
        return torch.matmul(self.box2world, self.d_box2world) 
    
    def regularize(self, iteration):
        loss = torch.norm(self.delta_r - torch.tensor([1., 0., 0., 0], device='cuda', requires_grad=False)) + torch.norm(self.delta_s - 1) + torch.norm(self.delta_t)
        loss = self.lambda_reg * loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_deltas(self):
        deltas = []
        with torch.no_grad():
            deltas.append(torch.norm(self.delta_r.detach().cpu() - torch.tensor([1., 0., 0., 0.])).item())
            deltas.append(torch.norm(self.delta_s.detach().cpu() - 1.).item())
            deltas.append(torch.norm(self.delta_t.detach().cpu()).item())
        return deltas