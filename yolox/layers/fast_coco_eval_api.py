#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file comes from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/fast_eval_api.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Megvii Inc. All rights reserved.

import copy
import time

import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# import torch first to make yolox._C work without ImportError of libc10.so
# in YOLOX, env is already set in __init__.py.
import torch
from yolox import _C

# from .box_iou_rotated import pairwise_iou_rotated
# from .rotated_coco_evaluation import BoxMode, RotatedBoxes


class COCOeval_opt(COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the functions evaluateImg()
    and accumulate() are implemented in C++ to speedup evaluation
    """

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        g = [g['bbox'][:4] for g in gt]
        d = [d['bbox'][:4] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

        # g = torch.FloatTensor([g['bbox'][:5] for g in gt])
        # d = torch.FloatTensor([d['bbox'][:5] for d in dt])

        # # compute iou between each dt and gt region
        # ious = pairwise_iou_rotated(d,g)
        # return ious

    # @staticmethod
    # def is_rotated(box_list):
    #     if type(box_list) == np.ndarray:
    #         return box_list.shape[1] == 5
    #     elif type(box_list) == list:
    #         if box_list == []:  # cannot decide the box_dim
    #             return False
    #         return np.all(
    #             np.array(
    #                 [
    #                     (len(obj) == 5) and ((type(obj) == list) or (type(obj) == np.ndarray))
    #                     for obj in box_list
    #                 ]
    #             )
    #         )
    #     return False

    # @staticmethod
    # def boxlist_to_tensor(boxlist, output_box_dim):
    #     if type(boxlist) == np.ndarray:
    #         box_tensor = torch.from_numpy(boxlist)
    #     elif type(boxlist) == list:
    #         if boxlist == []:
    #             return torch.zeros((0, output_box_dim), dtype=torch.float32)
    #         else:
    #             box_tensor = torch.FloatTensor(boxlist)
    #     else:
    #         raise Exception("Unrecognized boxlist type")

    #     input_box_dim = box_tensor.shape[1]
    #     if input_box_dim != output_box_dim:
    #         if input_box_dim == 4 and output_box_dim == 5:
    #             box_tensor = BoxMode.convert(box_tensor, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)
    #         else:
    #             raise Exception(
    #                 "Unable to convert from {}-dim box to {}-dim box".format(
    #                     input_box_dim, output_box_dim
    #                 )
    #             )
    #     return box_tensor

    # def compute_iou_dt_gt(self, dt, gt, is_crowd):
    #     if self.is_rotated(dt) or self.is_rotated(gt):
    #         # TODO: take is_crowd into consideration
    #         assert all(c == 0 for c in is_crowd)
    #         dt = RotatedBoxes(self.boxlist_to_tensor(dt, output_box_dim=5))
    #         gt = RotatedBoxes(self.boxlist_to_tensor(gt, output_box_dim=5))
    #         return pairwise_iou_rotated(dt, gt)
    #     else:
    #         # This is the same as the classical COCO evaluation
    #         return maskUtils.iou(dt, gt, is_crowd)

    # def computeIoU(self, imgId, catId):
    #     p = self.params
    #     if p.useCats:
    #         gt = self._gts[imgId, catId]
    #         dt = self._dts[imgId, catId]
    #     else:
    #         gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
    #         dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    #     if len(gt) == 0 and len(dt) == 0:
    #         return []
    #     inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
    #     dt = [dt[i] for i in inds]
    #     if len(dt) > p.maxDets[-1]:
    #         dt = dt[0 : p.maxDets[-1]]

    #     assert p.iouType == "bbox", "unsupported iouType for iou computation"

    #     g = [g["bbox"] for g in gt]
    #     d = [d["bbox"] for d in dt]

    #     # compute iou between each dt and gt region
    #     iscrowd = [int(o["iscrowd"]) for o in gt]

    #     # Note: this function is copied from cocoeval.py in cocoapi
    #     # and the major difference is here.
    #     ious = self.compute_iou_dt_gt(d, g, iscrowd)
    #     return ious


    def evaluate(self):
        """
        Run per image evaluation on given images and store results in self.evalImgs_cpp, a
        datastructure that isn't readable from Python but is used by a c++ implementation of
        accumulate().  Unlike the original COCO PythonAPI, we don't populate the datastructure
        self.evalImgs because this datastructure is a computational bottleneck.
        :return: None
        """
        tic = time.time()

        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print(
                "useSegm (deprecated) is not None. Running {} evaluation".format(
                    p.iouType
                )
            )
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()

        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]

        # <<<< Beginning of code differences with original COCO API

        def convert_instances_to_cpp(instances, is_det=False):
            # Convert annotations for a list of instances in an image to a format that's fast
            # to access in C++
            instances_cpp = []
            for instance in instances:
                instance_cpp = _C.InstanceAnnotation(
                    int(instance["id"]),
                    instance["score"] if is_det else instance.get("score", 0.0),
                    instance["area"],
                    bool(instance.get("iscrowd", 0)),
                    bool(instance.get("ignore", 0)),
                )
                instances_cpp.append(instance_cpp)
            return instances_cpp

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++
        ground_truth_instances = [
            [convert_instances_to_cpp(self._gts[imgId, catId]) for catId in p.catIds]
            for imgId in p.imgIds
        ]
        detected_instances = [
            [
                convert_instances_to_cpp(self._dts[imgId, catId], is_det=True)
                for catId in p.catIds
            ]
            for imgId in p.imgIds
        ]
        ious = [[self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds]

        if not p.useCats:
            # For each image, flatten per-category lists into a single list
            ground_truth_instances = [
                [[o for c in i for o in c]] for i in ground_truth_instances
            ]
            detected_instances = [
                [[o for c in i for o in c]] for i in detected_instances
            ]

        # Call C++ implementation of self.evaluateImgs()
        self._evalImgs_cpp = _C.COCOevalEvaluateImages(
            p.areaRng,
            maxDet,
            p.iouThrs,
            ious,
            ground_truth_instances,
            detected_instances,
        )
        self._evalImgs = None

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("COCOeval_opt.evaluate() finished in {:0.2f} seconds.".format(toc - tic))
        # >>>> End of code differences with original COCO API

    def accumulate(self):
        """
        Accumulate per image evaluation results and store the result in self.eval.  Does not
        support changing parameter settings from those used by self.evaluate()
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not hasattr(self, "_evalImgs_cpp"):
            print("Please run evaluate() first")

        self.eval = _C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)

        # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
        self.eval["recall"] = np.array(self.eval["recall"]).reshape(
            self.eval["counts"][:1] + self.eval["counts"][2:]
        )

        # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
        # num_area_ranges X num_max_detections
        self.eval["precision"] = np.array(self.eval["precision"]).reshape(
            self.eval["counts"]
        )
        self.eval["scores"] = np.array(self.eval["scores"]).reshape(self.eval["counts"])
        toc = time.time()
        print(
            "COCOeval_opt.accumulate() finished in {:0.2f} seconds.".format(toc - tic)
        )
