import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config


class VideoTracker(object):
    def __init__(self,cfg,isCuda,isDisplay,imgSeqPath):
        self.cfg = cfg
        self.isDisplay = isDisplay
        self.imgSeqPath = imgSeqPath
        use_cuda = isCuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if isDisplay:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

        self.frameCount = len(os.listdir(imgSeqPath))



    # def __init__(self, cfg, args):
    #     self.cfg = cfg
    #     self.args = args
    #     use_cuda = args.use_cuda and torch.cuda.is_available()
    #     if not use_cuda:
    #         warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
    #
    #     if args.display:
    #         cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    #         cv2.resizeWindow("test", args.display_width, args.display_height)
    #
    #
    #     if args.cam != -1:
    #         print("Using webcam " + str(args.cam))
    #         self.vdo = cv2.VideoCapture(args.cam)
    #     else:
    #         self.vdo = cv2.VideoCapture()
    #     self.detector = build_detector(cfg, use_cuda=use_cuda)
    #     self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
    #     self.class_names = self.detector.class_names

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        # fs = open(resultFile,'w+')
        while idx_frame <= self.frameCount-1:
            idx_frame += 1

            start = time.time()
            ori_im = cv2.imread(self.imgSeqPath+str(idx_frame)+".jpg")

            # _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids == 0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                print(bbox_xywh)
                print(cls_conf)

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)


                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)



            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

            if self.isDisplay:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)




def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=640)
    parser.add_argument("--display_height", type=int, default=480)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    imgSeqPath = "F:\\zte_compe_data\\trackdata\\A-data\\Track1\\"
    resultFile = "./Results/Track1.txt"


    with VideoTracker(cfg,1,1,imgSeqPath) as vdo_trk:
        vdo_trk.run()
