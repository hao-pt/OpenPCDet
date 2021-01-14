from mayavi import mlab
import numpy as np
import open3d as o3d

from tools.visual_utils import visualize_utils as V

lidar_list = ["/home/hp/Desktop/our_data/datasample_0701/pcd_lidar_final/1609906415_969976902_9012.pcd"]
pred_list = ["/home/hp/Desktop/Weighted-Boxes-Fusion/output/pillar_second_fusion/1609906415_969976902_9012.txt"]

CLASS_NAME = {"car": 0,
             "motocycle": 1,
             "bicycle": 2,
             "truck": 3}

idx = 0
points = np.asarray(o3d.io.read_point_cloud(lidar_list[idx]).points)
padding = np.zeros((points.shape[0], 1))
points = np.concatenate((points, padding), axis=-1)
pred_boxes = []
pred_scores = []
pred_labels = []
with open(pred_list[idx]) as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
for line in lines:
    data_line = line.split(" ")
    pred_labels.append(CLASS_NAME.get(data_line[0], 4))
    pred_boxes.append([float(x) for x in data_line[1:8]])
    pred_scores.append(float(data_line[-1]))

pred_labels = np.asarray(pred_labels)
pred_boxes = np.asarray(pred_boxes)
pred_scores = np.asarray(pred_scores)
# print(pred_labels, pred_boxes, pred_scores)

fig_3d = mlab.figure(bgcolor=(0, 0, 0), size=(800, 450))
V.draw_scenes(
        points=points, ref_boxes=pred_boxes,
        ref_scores=pred_scores, ref_labels=pred_labels
    )
mlab.savefig("demo.png")
fig_3d


mlab.show(stop=True)