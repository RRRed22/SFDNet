import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import pickle
import cv2
from torchvision import transforms
import numpy as np
import glob
from tqdm import tqdm
import os


# derive from https://github.com/TJ-IPLab/DNet/blob/master/layers.py
class HeightRecovery(nn.Module):
    """Layer to estimate scale through dense geometrical constrain
    """
    def __init__(self):
        super(HeightRecovery, self).__init__()
        self.batch_size = 1
        self.height = 966
        self.width = 1280


    def img2cam(self, norm, theta_lut, angle_lut):
        norm = norm.reshape(norm.size(0), 1, -1)  # B x 1 x HW
        # angle in the image plane
        theta = theta_lut.permute(0, 2, 1)
        # Obtain angle of incidence from radius
        angle_of_incidence = (angle_lut.permute(0, 2, 1)).to(device=norm.device)

        r_world = torch.sin(angle_of_incidence) * norm
        x = r_world * torch.cos(theta)
        y = r_world * torch.sin(theta)
        # Obtain `z` from the norm
        z = torch.cos(angle_of_incidence) * norm

        cam_coords = torch.cat((x, y, z), 1)
        cam_coords = torch.cat(
            [cam_coords, torch.ones(cam_coords.size(0), 1, cam_coords.shape[2]).to(device=norm.device)], 1)

        return cam_coords


    # derived from https://github.com/zhenheny/LEGO
    def get_surface_normal(self, cam_points, nei=1):
        cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
        cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
        cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
        cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
        cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
        cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
        cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
        cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
        cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

        vector_x0   = cam_points_x0   - cam_points_ctr
        vector_y0   = cam_points_y0   - cam_points_ctr
        vector_x1   = cam_points_x1   - cam_points_ctr
        vector_y1   = cam_points_y1   - cam_points_ctr
        vector_x0y0 = cam_points_x0y0 - cam_points_ctr
        vector_x0y1 = cam_points_x0y1 - cam_points_ctr
        vector_x1y0 = cam_points_x1y0 - cam_points_ctr
        vector_x1y1 = cam_points_x1y1 - cam_points_ctr

        normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
        normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
        normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
        normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

        normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
        normals = F.normalize(normals, dim=1)

        refl = nn.ReflectionPad2d(nei)
        normals = refl(normals)

        return normals

    def get_ground_mask(self, cam_points, normal_map, threshold=5):
        b, _, h, w = normal_map.size()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        threshold = math.cos(math.radians(threshold))
        ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
        vertical = torch.cat((zeros, ones, zeros), dim=1)

        cosine_sim = cos(normal_map, vertical).unsqueeze(1)
        vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

        y = cam_points[:,1,:,:].unsqueeze(1)
        ground_mask = vertical_mask.masked_fill(y <= 0, False)

        return ground_mask

    def forward(self, norm, theta_lut, angle_lut, road_mask):
        cam_points = self.img2cam(norm, theta_lut, angle_lut)
        cam_points = cam_points.reshape(1, 4, self.height, self.width)
        surface_normal = self.get_surface_normal(cam_points)
    
        cam_heights = (cam_points[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1)
        cam_heights_np = cam_heights * road_mask
        cam_heights_np = cam_heights_np.squeeze(0).squeeze(0).cpu().numpy()
        
        surface_normal_np = surface_normal
        surface_normal_np = surface_normal_np.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return cam_heights_np, surface_normal_np
        

if __name__ == "__main__":

    data_path = "/home/unun/data/Synwoodscape_distance/"

    min_dist = 0.1
    max_dist = 80

    angle_dict = {
        "FV": ["59.99","69.99", "79.99", "89.99"],
        "MVL": ["13.72", "53.72", "73.72"],
        "RV": ["41.28", "61.28", "71.28", "81.28"],
        "MVR": ["14.19", "54.19", "74.19"]
    }

    # Load LUTs
    with open("LUTs_1280x966.pkl", "rb") as f:
        LUTs = pickle.load(f)
    
    height_class = HeightRecovery()

    cam_side = input("What cam side to calculate: ")
    assert cam_side in ["FV", "MVL", "MVR", "RV"], "should not be other cam sides"

    
    if not os.path.exists(os.path.join(data_path, f"height_maps_aug_{cam_side}_{max_dist}")):
        os.mkdir(os.path.join(data_path, f"height_maps_aug_{cam_side}_{max_dist}"))
    if not os.path.exists(os.path.join(data_path, f"normal_maps_aug_{cam_side}_{max_dist}")):
        os.mkdir(os.path.join(data_path, f"normal_maps_aug_{cam_side}_{max_dist}"))

    depth_files = sorted(glob.glob(os.path.join(data_path, f"depth_maps_aug_{cam_side}/*.npy")))
    road_mask_files = sorted(glob.glob(os.path.join(data_path, f"road_mask_aug_{cam_side}/*.png")))

    assert len(depth_files) ==len(road_mask_files), "Number of depth files should be equal to that of road mask files"

    theta_lut_front = LUTs[cam_side]["theta"]
    angle_maps_lut_front = LUTs[cam_side]["angle_maps"]

    theta_lut_front_tensor = torch.from_numpy(theta_lut_front).cuda().unsqueeze(0)
    angle_maps_lut_front_tensor = torch.from_numpy(angle_maps_lut_front).cuda().unsqueeze(0)

    for i in tqdm(range(len(depth_files))):
        _depth_file = depth_files[i]
        _road_mask_file = road_mask_files[i]

        assert _depth_file.split("/")[-1].split(".npy")[0] == _road_mask_file.split("/")[-1].split(".png")[0], "Should corresponded name."

        angle_depth = _depth_file.split("/")[-1].split(".npy")[0].split("_")[-1]
        angle_road_mask = _road_mask_file.split("/")[-1].split(".png")[0].split("_")[-1]

        assert angle_depth == angle_road_mask, "should be same angle slant"

        height_save_path = os.path.join(data_path, f"height_maps_aug_{cam_side}_{max_dist}", _depth_file.split("/")[-1])
        normal_save_path = os.path.join(data_path, f"normal_maps_aug_{cam_side}_{max_dist}", _depth_file.split("/")[-1])

        # height_save_path = os.path.join("height.npy")
        # normal_save_path = os.path.join("normal.npy")

        if angle_depth in angle_dict[cam_side]:

            if (not os.path.exists(height_save_path)) or (not os.path.exists(normal_save_path)):
            # if not os.path.exists(height_save_path):
            # if not os.path.exists(normal_save_path):
                depth = np.load(_depth_file).astype(np.float32)
                depth = np.clip(depth, min_dist, max_dist)  # Remember to clip here!!!!!
                depth_tensor = torch.from_numpy(depth).cuda().reshape(1, 1280 * 966)

                road_mask = cv2.imread(_road_mask_file, cv2.IMREAD_GRAYSCALE)
                road_mask_tensor = transforms.ToTensor()(road_mask).cuda().unsqueeze(0).bool()

                cam_heights_np, normal_np = height_class(
                    depth_tensor, theta_lut_front_tensor, angle_maps_lut_front_tensor, road_mask_tensor)
            
                np.save(height_save_path, cam_heights_np)
                np.save(normal_save_path, normal_np)
    
    print("Done!")
