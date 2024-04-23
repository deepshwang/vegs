from utils.graphics_utils import cam_normal_to_world_normal, quaternion_to_matrix

def loss_normal_guidance(viewpoint_cam, cov_quat, cov_scale):
    norm_pred = viewpoint_cam.original_normal
    _, H, W = norm_pred.shape

    # rendering result
    cov_scale = cov_scale.permute(1,2,0).reshape(-1, 1, 3).contiguous()  # cov_scale.shape: torch.Size([n_pix, 1, 3])    
    cov_quat = cov_quat.permute(1,2,0).reshape(-1, 4).contiguous()
    cov_rot = quaternion_to_matrix(cov_quat)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
    cov_rs = cov_rot.detach() * cov_scale                            # cov_rs.shape: torch.Size([n_pix, 3, 3])

    # off-the-shelf normal 
    norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)    # norm_pred_world.shape: torch.Size([3, 376, 1408])
    norm_pred_world = norm_pred_world.permute(1,2,0).reshape(-1, 3).contiguous()             # norm_pred_world.shape: torch.Size([n_pix, 3])
    norm_pred_world = norm_pred_world[:, :, None].repeat(1,1,3)                 # norm_pred_world.shape: torch.Size([n_pix, 3, 3])

    #loss =  (cov_rs * norm_pred_world).sum(dim=-2).abs().mean()
    #loss =  (cov_rot * norm_pred_world).sum(dim=-2).abs().mean()
    
    #TODO: Parametrize the labmda weigihts
    loss =  0.8 * (cov_rot * norm_pred_world).sum(dim=-2).abs().mean() + 0.2 * (cov_rs * norm_pred_world).sum(dim=-2).abs().mean()
    return loss