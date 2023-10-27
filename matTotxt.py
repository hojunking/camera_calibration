import numpy as np
import h5py

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    # Given intrinsics fx, fy, cx, cy
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))

# Load the .mat file
with h5py.File('../data/nyu_depth_v2_labeled.mat', 'r') as file:
    depths = np.array(file['depths'])

point_cloud = depth_to_pointcloud(np.transpose(depths[:,:,0], (1, 0)), fx=525.0, fy=525.0, cx=319.5, cy=239.5)
np.savetxt('output.xyz', point_cloud, delimiter=' ')
