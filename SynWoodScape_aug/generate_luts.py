import glob
import json
import os
import pickle
import numpy as np
from tqdm import tqdm

class LUTs_generation:
    def __init__(self):
        self.data_path = "/home/unun/data/Synwoodscape_distance"
        self.height = 966
        self.width = 1280

    def get_K_D(self, intrinsic):
        """Scale the intrinsic"""
        D = np.array([intrinsic["k1"], intrinsic["k2"], intrinsic["k3"], intrinsic["k4"]], dtype=np.float32)
        K = np.array([1.0, intrinsic["aspect_ratio"], intrinsic["cx_offset"], intrinsic["cy_offset"]])
        K[2] += self.width / 2 - 0.5
        K[3] += self.height / 2 - 0.5

        return K, D

    def inverse_poly_lut(self):
        """Create LUTs for the polynomial projection model as there is no analytical inverse"""

        # First read intrinsics from calibration data
        jsons = sorted(list(glob.glob(os.path.join(self.data_path, "calibration_data", "*.json"))))

        intrinsics = dict()
        for json_file in jsons:
            data = json.load(open(json_file))
            intrinsics[data["name"]] = data["intrinsic"]

        # Calculate LUTs for each camera
        LUTs = dict()

        # Four views of camera intrinsic
        for cam_side, intrinsic in intrinsics.items():
            
            print(f"\n###### Calculating LUTs for {cam_side}")
            K, D = self.get_K_D(intrinsic)

            x = np.linspace(0, self.width - 1, self.width)
            y = np.linspace(0, self.height - 1, self.height)
            mesh_x, mesh_y = np.meshgrid(x, y)
            mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)

            x_cam = (mesh_x - K[2]) / K[0] # (x-cx)/fx
            y_cam = (mesh_y - K[3]) / K[1] # (y-cy)/fy

            r = np.sqrt(x_cam * x_cam + y_cam * y_cam)
            theta_LUT = np.arctan2(y_cam, x_cam).astype(np.float32)
            angle_LUT = np.zeros_like(r, dtype=np.float32)

            for i, _r in tqdm(enumerate(r)):
                a = np.roots([D[3], D[2], D[1], D[0], -_r])
                a = np.real(a[a.imag == 0])
                try:
                    a = np.min(a[a >= 0])
                    angle_LUT[i] = a
                except ValueError:  # raised if `a` is empty.
                    print(f"Field angle of incident ray is empty")
                    pass

            LUTs[cam_side] = dict(theta=theta_LUT, angle_maps=angle_LUT)
        
        with open("LUTs.pkl", "wb") as f:
            pickle.dump(LUTs, f, pickle.HIGHEST_PROTOCOL)

        print("Done.")


if __name__ == "__main__":
    LUTs_class = LUTs_generation()
    LUTs_class.inverse_poly_lut()
