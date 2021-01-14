from mayavi import mlab
import numpy as np
import open3d as o3d
import os
from os import path as osp
import argparse

from tools.visual_utils import visualize_utils as V

parser = argparse.ArgumentParser(description="Plot lidar points and 3D boxes in scene")
parser.add_argument("--lidar_dir", help="Directory of lidar data")
parser.add_argument("--pred_dir", help="Directory of prediction results")
parser.add_argument("--gt_dir", help="Directory of groundtruth data")
parser.add_argument("--output_dir", help="Output directory of visualization")
args = parser.parse_args()

# lidar_list = ["/home/hp/Desktop/our_data/datasample_0701/pcd_lidar_final/1609906415_969976902_9012.pcd"]
# pred_list = ["/home/hp/Desktop/Weighted-Boxes-Fusion/output/pillar_second_fusion/1609906415_969976902_9012.txt"]

CLASS_NAME = {"car": 0,
             "motocycle": 1,
             "bicycle": 2,
             "truck": 3}

def read_annot(path):
    pred_boxes = []
    pred_scores = []
    pred_labels = []

    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        data_line = line.split(" ")
        pred_labels.append(CLASS_NAME.get(data_line[0].lower(), 4))
        pred_boxes.append([float(x) for x in data_line[1:8]])
        pred_scores.append(float(data_line[-1]))
    
    return pred_boxes, pred_scores, pred_labels

def list_dir(path):
    if not path:
        return []
    
    path_list = []
    for dirpath, folder, files in os.walk(path):
        for file in files:
            path_list.append(osp.join(dirpath, file))
    
    return path_list

def main(args):
    lidar_list = list_dir(args["lidar_dir"])
    pred_list = list_dir(args["pred_dir"])
    gt_list = list_dir(args["gt_dir"])
    os.makedirs(args["output_dir"], exist_ok=True)

    idx = 0
    for idx in range(len(lidar_list)):
        filename = lidar_list[idx].split(os.sep)[-1].split('.')[0]
        points = np.asarray(o3d.io.read_point_cloud(lidar_list[idx]).points)
        padding = np.zeros((points.shape[0], 1))
        points = np.concatenate((points, padding), axis=-1)

        # read pred annot
        pred_boxes, pred_scores, pred_labels = read_annot(pred_list[idx])
        pred_labels = np.asarray(pred_labels)
        pred_boxes = np.asarray(pred_boxes)
        pred_scores = np.asarray(pred_scores)
        # print(pred_labels, pred_boxes, pred_scores)

        # read gt annot
        bt_boxes, gt_scores, gt_labels = None, None, None
        if gt_list:
            gt_boxes, gt_scores, gt_labels = read_annot(gt_list[idx])
            gt_labels = np.asarray(gt_labels)
            gt_boxes = np.asarray(gt_boxes)
            gt_scores = np.asarray(gt_scores)


        fig_3d = mlab.figure(bgcolor=(0, 0, 0), size=(800, 450))
        V.draw_scenes(
                points=points, ref_boxes=pred_boxes, gt_boxes=gt_boxes,
                ref_scores=pred_scores, ref_labels=pred_labels
            )
        mlab.savefig(osp.join(args["output_dir"], filename+".png"))

        mlab.close(all=True)
        # fig_3d
        # mlab.show(stop=True)

if __name__ == "__main__":
    main(args.__dict__)

