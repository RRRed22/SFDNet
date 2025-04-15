import glob
import json
import os
import pickle
import numpy as np
from tqdm import tqdm
import sys


def get_intrinsics():
    jsons = sorted(list(glob.glob(os.path.join("calibration_intrinsics", "*.json"))))

    intrinsics = dict()
    for json_file in jsons:
        data = json.load(open(json_file))
        intrinsics[data["name"]] = data["intrinsic"]
    return intrinsics


def scale_intrinsic(intrinsic):
    """Scale the intrinsic"""

    D = np.array([intrinsic["k1"], intrinsic["k2"], intrinsic["k3"], intrinsic["k4"]], dtype=np.float32)
    K = np.array([1.0, intrinsic["aspect_ratio"], intrinsic["cx_offset"], intrinsic["cy_offset"]])
    K[2] += 1280 / 2 - 0.5
    K[3] += 966 / 2 - 0.5

    
    K_scale = K.copy()
    K_scale[2] *= (1280 / 1280)
    K_scale[3] *= (966 / 966)

    D_scale = D.copy()
    D_scale *= (1280 / 1280)

    return K_scale, D_scale


def inverse_poly_lut(intrinsics):
    """Create LUTs for the polynomial projection model as there is no analytical inverse"""

    LUTs = dict()

    # Four views of camera intrinsic
    for cam_side, intrinsic in intrinsics.items():
        
        print(f"\n###### Calculating LUTs for {cam_side}")
        K_scale, D_scale = scale_intrinsic(intrinsic)

        x = np.linspace(0, 1280 - 1, 1280)
        y = np.linspace(0, 966 - 1, 966)
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)

        x_cam = (mesh_x - K_scale[2]) / K_scale[0]  # (x - cx) / fx
        y_cam = (mesh_y - K_scale[3]) / K_scale[1]  # (y - cy) / fy

        r = np.sqrt(x_cam * x_cam + y_cam * y_cam)
        theta_LUT = np.arctan2(y_cam, x_cam).astype(np.float32)
        angle_LUT = np.zeros_like(r, dtype=np.float32)

        for i, _r in tqdm(enumerate(r)):
            a = np.roots([D_scale[3], D_scale[2], D_scale[1], D_scale[0], -_r])
            a = np.real(a[a.imag == 0])
            try:
                a = np.min(a[a >= 0])
                angle_LUT[i] = a
            except ValueError:  # raised if `a` is empty.
                print(f"Field angle of incident ray is empty")
                pass

        LUTs[cam_side] = dict(theta=theta_LUT, angle_maps=angle_LUT)
    
    with open("LUTs_1280x966.pkl", "wb") as f:
        pickle.dump(LUTs, f, pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == "__main__":

    camera_intrinsics = get_intrinsics()
    inverse_poly_lut(camera_intrinsics)
