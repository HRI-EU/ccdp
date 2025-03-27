import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from typing import *
import numpy as np
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import math
cur_dir = '/hri/localdisk/Amir/Codes_h100/Failure_Recovery/Picking'
from ccdp_utils import cosine_beta_schedule, ConditionalUnet1D
from jsonrpcserver import method, serve, Success
import socket
import random


n_history = 2
n_pred = 8
applied = 8
box_locations = [
    torch.tensor([-0.5,0]),
    torch.tensor([0.,0.3]),
    torch.tensor([0.5,0]),
]
obj_height = 0.05




n_diff_iter = 100
beta_schedule = cosine_beta_schedule(n_diff_iter).to(device)
alpha = 1.0 - beta_schedule
alpha_cumprod = torch.cumprod(alpha, dim=0)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)


model_a_h = ConditionalUnet1D(3, 14, diffusion_step_embed_dim = 256).to(device)
model_a = ConditionalUnet1D(3, 0, diffusion_step_embed_dim = 256).to(device)
model_a_s = ConditionalUnet1D(3, 6, diffusion_step_embed_dim = 256).to(device)
model_avoid = ConditionalUnet1D(3, 8,  diffusion_step_embed_dim = 256).to(device)


model_a_h.load_state_dict(torch.load(cur_dir + "/3_box_model_a_h_ddpm.pth"))
model_a.load_state_dict(torch.load(cur_dir + "/3_box_model_a_ddpm.pth"))
model_a_s.load_state_dict(torch.load(cur_dir + "/3_box_model_a_s_ddpm.pth"))
model_avoid.load_state_dict(torch.load(cur_dir + "/3_box_simple_avoide_mod_ddpm.pth"))



def sample_with_avoidace(hist, avoid =  [], n_samples = 30):
    label = hist.reshape([n_samples,-1])
    scaled_label = label.to(dtype=torch.float32).to(device) 
    x_0 = torch.randn(n_samples,n_pred,3).to(device)
    x = x_0.clone()
    Avoid_Label = []
    for av in avoid:
        Avoid_Label.append(av.reshape([1,-1]).to(device).to(dtype=torch.float32).repeat([n_samples,1]))
    with torch.no_grad():
        for t in reversed(range(n_diff_iter)):
            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            beta_t = beta_schedule[t]
            alpha_t = alpha[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_cump_t = sqrt_alpha_cumprod[t]
            sqrt_one_minus_alpha_cump_t = sqrt_one_minus_alpha_cumprod[t]
            for j in range(1):
                if t > 0:
                    noise = torch.randn_like(x).to(device)
                else:
                    noise = 0
                pred_noise_a = model_a(x,t)
                dx_a =  (x - beta_t/sqrt_one_minus_alpha_cump_t * pred_noise_a) / sqrt_alpha_t - x 
                pred_noise_s = model_a_s(x,t, scaled_label[:,-6:])
                dx_s =  (x - beta_t/sqrt_one_minus_alpha_cump_t * pred_noise_s) / sqrt_alpha_t - x 
                
                
                pred_noise_h = model_a_h(x,t, scaled_label)
                dx_h =  (x - beta_t/sqrt_one_minus_alpha_cump_t * pred_noise_h) / sqrt_alpha_t - x 
                
                dx_avoid = torch.zeros_like(dx_h)
                w_h = 1
                for f_id, av in enumerate(Avoid_Label):
                    w_h =  0
                    aug_label = torch.cat([av, scaled_label[:,-6:]], axis = 1)
                    pred_noise_i = model_avoid(x,t, aug_label)
                    dx_i =  (x - beta_t/sqrt_one_minus_alpha_cump_t * pred_noise_i) / sqrt_alpha_t - x 
                    dx_avoid += 1/len(Avoid_Label)* (dx_i - dx_a)
                    # inverse diffusion step (remove noise)
                dx = 1 * dx_a + 0*(1-w_h)* (dx_s - dx_a) + w_h * (dx_h - dx_a) +  dx_avoid +  noise * beta_t**0.5
                x += dx# + noise * one_minus_alpha_t#(dx_1 + dx_2 )/3 + noise * one_minus_alpha_t 
    return x




@method
def get_traj(ee_pos_list, obj_loc_list, failure_modes):
    # torch.manual_seed(42)
    # random.seed(42)
    # np.random.seed(42) 
    ee_pos_arr = np.array(ee_pos_list)
    obj_loc_arr = np.array(obj_loc_list)
    avoiding_label = []
    for lis in failure_modes:
        avoiding_label.append(torch.from_numpy(np.array(lis)).to(device))
    print(avoiding_label)    
    ee_pos = torch.from_numpy(ee_pos_arr).to(device).reshape([1,1,3])
    obj_loc = torch.from_numpy(obj_loc_arr).to(device).reshape([1,2])

    traj = ee_pos.repeat([1,n_history +1,1])
    grasped_traj = torch.zeros([1,n_history + 1,1]).to(device)

    hist = torch.cat([traj, grasped_traj],axis = 2).reshape([1,-1])
    label = torch.cat([hist, obj_loc], axis = 1)

    future = sample_with_avoidace(label,avoid=avoiding_label, n_samples = 1)
    future_reshaped = future.reshape([1,n_pred,3])
    applied_future = future_reshaped[:,:applied,:]
    grasped_future = torch.zeros([1,applied,1]).to(device)
    grasped = 0
    for j in range(applied):
        state_robot = applied_future[:,j,:]
        dist_obj = torch.norm(state_robot[0,:2] - obj_loc[0,:])
        if (state_robot[0,2] - obj_height) < 0.02 and dist_obj.item() < 0.02:
            grasped = 1
        grasped_future[0,j,0] = grasped
    
    result_torch = torch.cat([applied_future, grasped_future],axis = 2)
    result_numpy = result_torch.cpu().numpy()
    return Success(result_numpy.tolist())




host = socket.gethostbyname(socket.gethostname())  # Get local IP address
port = 5000
print(f"Serving on {host}:{port}")
serve(host, port)  