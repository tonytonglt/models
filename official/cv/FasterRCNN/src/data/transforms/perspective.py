import sys
import numpy as np
import cv2
import math
import copy

sys.path.append('../')
from data.general import resample_polys, poly2box, in_range

__all__ = ['RandomPerspective']


class RandomPerspective:
    """
    Args:
        degrees (float): the rotate range to apply, transform range is [-10, 10]
        translate (float): the translate range to apply, transform range is [-0.1, 0.1]
        scale (float): the scale range to apply, transform range is [0.1, 2]
        shear (float): the shear range to apply, transform range is [-2, 2]
        perspective (float): the perspective range to apply, transform range is [0, 0.001]
    """
    def __init__(self, degrees=10, translate=.1, scale=.1, shear=2, perspective=0.0, border=(0, 0)):
        if not (in_range(degrees, -10, 10) and in_range(translate, -0.1, 0.1) and in_range(scale, 0.1, 2) and in_range(shear, -2, 2) and in_range(perspective, 0, 0.001)):
            raise ValueError('{}: input value is invalid!'.format(self))
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def __call__(self, img, w, h, gt_bbox, gt_class, *gt_poly):
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = np.random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = np.random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = np.random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = np.random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(gt_bbox)
        new_bbox = np.zeros((n, 4))
        new_poly = [None] * n
        if n:
            if gt_poly:
                resample_result = resample_polys(*gt_poly)  # upsample
                for i, poly in enumerate(resample_result):
                    xy = np.ones((len(poly), 3))
                    xy[:, :2] = poly
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]  # perspective rescale or affine

                    # clip
                    new_bbox[i] = poly2box(xy, width, height)
                    xy[:, 0] = xy[:, 0].clip(0, width)
                    xy[:, 1] = xy[:, 1].clip(0, height)
                    new_poly[i] = xy
            else:
                xy = np.ones((n * 4, 3))
                xy[:, :2] = gt_bbox[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new_bbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new_bbox[:, [0, 2]] = new_bbox[:, [0, 2]].clip(0, width)
                new_bbox[:, [1, 3]] = new_bbox[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=gt_bbox.T * s, box2=new_bbox.T, area_thr=0.01 if gt_poly else 0.10)
            gt_class = gt_class[i]
            gt_bbox = gt_bbox[i]
            gt_bbox = new_bbox[i]
            if gt_poly:
                filter_result = []
                for j, value in enumerate(i):
                    if value:
                        filter_result.append(new_poly[j])
        if gt_poly:
            return img, width, height, gt_bbox, gt_class, gt_poly
        else:
            return img, width, height, gt_bbox, gt_class


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates