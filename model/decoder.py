import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDecoder(nn.Module):

    def __init__(self, inst_info, d_latent=512, hidden=[1, 1], num_points=1024, max_sh_degree=4):
        super(GaussianDecoder, self).__init__()
        self.d_latent = d_latent
        self.num_points = num_points
        self.max_sh_degree = max_sh_degree
        self.dim_feat = 3 * (max_sh_degree + 1) ** 2
        self.xyz_out_dim= 3 * num_points
        self.dim_scaling = 3
        self.dim_rotation = 4
        self.dim_opacity = 1
        self.feat_out_dim = num_points * (self.dim_feat + self.dim_scaling + self.dim_rotation + self.dim_opacity) # (X, SH, scaling) * 3 + rotation + opacity
        self.embeddings = nn.Embedding(len(inst_info), d_latent).cuda()
        self.inst2idx={}
        for i in range(len(inst_info)):
            self.inst2idx[inst_info[i]] = torch.LongTensor([i]).cuda()

        decoder = []
        # Hidden layers
        for i in range(len(hidden) - 1):
            decoder.append(nn.Linear(d_latent * hidden[i], d_latent * hidden[i + 1]))
            decoder.append(nn.LeakyReLU(0.1)) 
        self.decoder = nn.ModuleList(decoder)
        self.xyz_head = nn.Sequential(nn.Linear(d_latent * hidden[-1], d_latent * hidden[-1]),
                                      nn.LeakyReLU(0.1),
                                      nn.Linear(d_latent * hidden[-1], self.xyz_out_dim)) 
        self.feat_head = nn.Sequential(nn.Linear(d_latent * hidden[-1], d_latent * hidden[-1]),
                                       nn.LeakyReLU(0.1),
                                       nn.Linear(d_latent * hidden[-1], self.feat_out_dim))

        #decoder.append(nn.Linear(d_latent * hidden[-1], self.out_dim))
        

    def forward(self, inst):
        i = self.inst2idx[inst]
        x = self.embeddings(i)
        for dec in self.decoder:
            x = dec(x)
        _xyz = self.xyz_head(x).reshape(self.num_points, -1)
        #_xyz = torch.clamp(_xyz, -0.5, 0.5)
        # _xyz = 0.5 * torch.tanh(_xyz) # (num_points, 3)
        
        # DEBUG
        # _xyz = torch.rand_like(_xyz) - 0.5

        x = self.feat_head(x).reshape(self.num_points, -1)
        

        _features_dc = x[:, :3][:, None, :] # (num_points, 1, 3)
        
        end_idx = 3 * ((self.max_sh_degree + 1) ** 2)
        _features_rest = x[:, 3:end_idx].reshape(self.num_points, -1, 3) # (num_points, 15, 3)  

        # TODO: Fix this
        _scaling = 0.005 * F.sigmoid(x[:, end_idx:end_idx + 3]) # (num_points, 3)
        end_idx += 3

        _rotation = x[:, end_idx:end_idx + 4] # (num_points, 4)
        end_idx += 4

        _opacity = F.relu(x[:, end_idx:end_idx + 1]) # (num_points, 1)

        return _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity