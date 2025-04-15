from PIL import Image
import numpy as np
import argparse
import json
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm

class Synwoodscape_aug:
    def __init__(self, args):
        self.data_path = "/home/unun/data/Synwoodscape_distance"

        self.aug_type = args.warp
        aug_obj = {"rgb": "rgb_images", 
                   "seg": "semantic_annotations",
                   "seg_label": "semantic_digit",
                   "flow": "optical_flow",
                   "motion": "motion_annotations",
                   "mask": "self_mask",
                   "ins": "instance_annotations",
                   "road_mask": "road_mask",
                   "dynamic_mask": "dynamic_mask",
                   }[self.aug_type]
        self.warp_src_path = os.path.join(self.data_path, aug_obj)
        if self.aug_type in ["motion", "ins"]:
            self.warp_src_path = os.path.join(self.data_path, aug_obj, "gtLabels")
        if self.aug_type == "seg":
            self.warp_src_path = os.path.join(self.data_path, aug_obj, "rgbLabels")
        if self.aug_type == "seg_label":
            self.warp_src_path = os.path.join(self.data_path, "semantic_annotations", "gtLabels")

        self.warp_dst_path = os.path.join(self.data_path, f"{aug_obj}_aug_{args.cam_side}")
        if not os.path.exists(self.warp_dst_path):
            os.mkdir(self.warp_dst_path)

        self.depth_src_path = os.path.join(self.data_path, "depth_maps/raw_data")
        self.depth_dst_path = os.path.join(self.data_path, f"depth_maps_aug_{args.cam_side}")
        if not os.path.exists(self.depth_dst_path):
            os.mkdir(self.depth_dst_path)

        self.prj_mask_dst_path = os.path.join(self.data_path, f"proj_mask_{args.cam_side}")
        if not os.path.exists(self.prj_mask_dst_path):
            os.mkdir(self.prj_mask_dst_path)

        self.calibration_src_path = os.path.join(self.data_path, "calibration_data")
        self.height = 966
        self.width = 1280
        self.angle_orig = {"FV": 59.99, "MVL": 13.72, "MVR": 14.19, "RV": 41.28}

        # Augment with selected angles
        self.slanted_angles = {"FV": [69.99, 79.99, 89.99],
                               "MVL": [53.72, 73.72],
                               "RV": [61.28, 71.28, 81.28],
                               "MVR": [54.19, 74.19]
        }


        with open('LUTs.pkl', 'rb') as f:
            self.luts = pickle.load(f)


    def get_intrinsics_dict(self, cam_side):
        data = json.load(open(os.path.join(self.data_path, "calibration_data", f"{cam_side}.json")))
        intrinsics = data['intrinsic']
        return intrinsics

    def create_fisheye_raw_point_cloud(self, rgb, norm, angle_lut, theta_lut):
        rgb = list(map(lambda x: x.reshape(x.shape[1]*x.shape[0], 1), cv2.split(rgb)))

        def _img2world(norm, angle_of_incidence, theta):
            
            norm = norm.reshape((norm.shape[0]*norm.shape[1], -1))
            r_world = norm * np.sin(angle_of_incidence)
            x = r_world * np.cos(theta)
            y = r_world * np.sin(theta)
            # Obtain `z` from the norm
            z = np.cos(angle_of_incidence) * norm
            return x, y, z

        x, y, z = _img2world(norm, angle_lut, theta_lut)

        if len(rgb) == 3:   # color
            point_cloud = np.hstack([x, y, z, rgb[0], rgb[1], rgb[2]])
        elif len(rgb) == 1: # grey
            point_cloud = np.hstack([x, y, z, rgb[0]])

        return point_cloud


    def warp(self, ori_rgb, ori_depth, intrinsics, point_cloud, angle_diff):

        # intrinsics
        aspect_ratio = intrinsics['aspect_ratio']
        k1 = intrinsics['k1']
        k2 = intrinsics['k2']
        k3 = intrinsics['k3']
        k4 = intrinsics['k4']
        cx = intrinsics['cx_offset']
        cy = intrinsics['cy_offset']

        # vacant array
        warp_rgb = np.zeros_like(ori_rgb)
        warp_depth = np.zeros_like(ori_depth)

        # rotation
        angle_diff = np.radians(angle_diff)
        R = np.array([[1,0,0],
                      [0, np.cos(-angle_diff), -np.sin(-angle_diff)], 
                      [0, np.sin(-angle_diff), np.cos(-angle_diff)]])
        rotated_point = (R @ point_cloud[:, :3].T).T        
        rotated_point = np.concatenate((rotated_point, point_cloud[:, 3:]), 1)

        # center point
        px = cx + self.width/2 - 0.5
        py = cy + self.height/2 - 0.5

        # projection
        #rotated_point = rotated_point.reshape((height, width, 9))


        if len(ori_rgb.shape) == 3: # rgb
            rotated_point = rotated_point.reshape((self.height, self.width, 6))
        elif len(ori_rgb.shape) == 2:   # grey
            rotated_point = rotated_point.reshape((self.height, self.width, 4))

        chi = np.sqrt(rotated_point[:,:,0] **2 + rotated_point[:,:,1] **2)
        theta = np.pi/2 - np.arctan2(rotated_point[:,:,2], chi)
        rho = k1 * theta + k2 * theta**2 + k3 * theta**3 + k4 * theta**4
        
        x = rho * rotated_point[:,:,0] / (chi+1e-6) + px
        y = rho * rotated_point[:,:,1] / (chi+1e-6) * aspect_ratio + py

        x_mask = np.logical_and(x>=0, x<self.width)
        y_mask = np.logical_and(y>=0, y<self.height)
        
        x = (x * x_mask)#.reshape(width * height, -1)
        y = (y * y_mask)#.reshape(width * height, -1)
        z = rotated_point[:,:,2]
        corner_pixel = z[0,0]
        z_mask = z.copy()

        for i in range(self.width):
            for j in range(self.height):
                if z_mask[j,i] == corner_pixel:
                    z_mask[j,i] = 0
                else:
                    z_mask[j,i] = 1
                if int(x[j,i]) == 0 or int(y[j,i]) == 0:
                    continue
                if int(x[j,i]) >= self.width or int(y[j,i]) >= self.height:
                    continue
                
                if len(ori_rgb.shape) == 3: # rgb
                    warp_rgb[j,i] = [rotated_point[int(y[j,i]), int(x[j,i]), 5], rotated_point[int(y[j,i]), int(x[j,i]), 4], rotated_point[int(y[j,i]), int(x[j,i]), 3]]
                elif len(ori_rgb.shape) == 2:   # grey
                    warp_rgb[j,i] = rotated_point[int(y[j,i]), int(x[j,i]), 3]

                warp_depth[j,i] = np.linalg.norm(rotated_point[int(y[j,i]), int(x[j,i]), :3])

        if len(ori_rgb.shape) == 3: # rgb
            z_mask = np.transpose(np.tile(z_mask, (3, 1, 1)), (1, 2, 0))
        elif len(ori_rgb.shape) == 2:   # grey
            z_mask = z_mask

        warp_rgb_ = warp_rgb * z_mask

        mask = (x_mask & y_mask).reshape(self.height, self.width)
        mask_image = (mask * 255).astype(np.uint8)
            
        return warp_rgb_, warp_depth, mask_image


    def run(self):

        img_files = sorted(glob(os.path.join(self.warp_src_path, f"*{args.cam_side}.png")))
        depth_files = sorted(glob(os.path.join(self.depth_src_path, f"*{args.cam_side}.npy")))
        print(len(img_files), len(depth_files))
        assert len(img_files) == len(depth_files), "Number of image files not equal to depth files."

        for depth_path in tqdm(depth_files):

            depth_filename = depth_path.split("/")[-1]
            index, cam_side = depth_filename.split(".npy")[0].split("_")

            color_path = os.path.join(self.warp_src_path, "{}_{}.png".format(index, cam_side))
            color = np.array(Image.open(color_path))
            dense_depth = np.load(depth_path)
            angle_lut = self.luts[cam_side]["angle_maps"]
            theta_lut = self.luts[cam_side]["theta"]

            intrinsics = self.get_intrinsics_dict(cam_side)
            pcl = self.create_fisheye_raw_point_cloud(color, dense_depth, angle_lut, theta_lut)

            for slant_angle in self.slanted_angles[cam_side]:

                color_aug_path = os.path.join(self.warp_dst_path, "{}_{}_{}.png".format(index, cam_side, slant_angle))
                depth_aug_path = os.path.join(self.depth_dst_path, "{}_{}_{}.npy".format(index, cam_side, slant_angle))
                proj_mask_path = os.path.join(self.prj_mask_dst_path, "{}_{}_{}.png".format(index, cam_side, slant_angle))

                # To prevent from repeat calculating. If one of the main thing exist, then calculate
                if (not os.path.exists(color_aug_path)) or (not os.path.exists(depth_aug_path)) or (not os.path.exists(proj_mask_path)):
                    
                    w_color, w_dep, w_mask = self.warp(color, dense_depth, intrinsics, pcl, self.angle_orig[cam_side]-slant_angle)
                    
                    cv2.imwrite(color_aug_path, w_color)

                    if not os.path.exists(depth_aug_path):
                        np.save(depth_aug_path, w_dep)
                    
                    if not os.path.exists(proj_mask_path):
                        cv2.imwrite(proj_mask_path, w_mask)


        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do data augmentation")
    parser.add_argument('--warp', type=str, required=True, help="What to warp", default="rgb",
                        choices=["rgb", "seg", "seg_label", "motion", "mask", "flow", "ins", "road_mask", "dynamic_mask"])
    parser.add_argument("--cam_side", type=str, required=True, help="Cam side", default="FV",
                        choices=["FV", "MVL", "MVR", "RV"])
    args = parser.parse_args()

    synwood = Synwoodscape_aug(args)
    synwood.run()
