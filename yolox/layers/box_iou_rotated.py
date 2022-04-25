import torch

def pairwise_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.
    Arguments:
        boxes1 (Tensor[N, 5])
        boxes2 (Tensor[M, 5])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    boxes1[:,4] = -boxes1[:,4]
    boxes2[:,4] = -boxes2[:,4]
    return torch.ops.yolox.box_iou_rotated(boxes1.type(torch.float32), boxes2.type(torch.float32))