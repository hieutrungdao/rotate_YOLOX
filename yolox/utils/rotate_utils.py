from json.tool import main
import numpy as np
import torch
import shapely

try:
    # if error in importing polygon_inter_union_cuda, polygon_b_inter_union_cuda, please cd to ./iou_cuda and run "python setup.py install"
    from polygon_inter_union_cuda import polygon_inter_union_cuda, polygon_b_inter_union_cuda
    polygon_inter_union_cuda_enable = True
    polygon_b_inter_union_cuda_enable = True
except Exception as e:
    print(f'Warning: "polygon_inter_union_cuda" and "polygon_b_inter_union_cuda" are not installed.')
    print(f'The Exception is: {e}.')
    polygon_inter_union_cuda_enable = False
    polygon_b_inter_union_cuda_enable = False

def order_corners(boxes):
    """
        Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
        Sorted corners have the following restrictions: 
                                y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
    """
    
    if boxes.shape[0] == 0:
        return torch.empty(0, 8, device=boxes.device)
    boxes = boxes.view(-1, 4, 2)
    x = boxes[..., 0]
    y = boxes[..., 1]
    y_sorted, y_indices = torch.sort(y) # sort y
    idx = torch.arange(0, y.shape[0], dtype=torch.long, device=boxes.device)
    complete_idx = idx[:, None].repeat(1, 4)
    x_sorted = x[complete_idx, y_indices]
    x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    y_sorted[idx, :2] = y_sorted[idx, :2][complete_idx[:, :2], x_bottom_indices]
    y_sorted[idx, 2:4] = y_sorted[idx, 2:4][complete_idx[:, 2:4], x_top_indices]
    
    # prevent the ambiguous case when the diagonal of the quadrilateral is parallel to the x-axis
    special = (y_sorted[:, 1] == y_sorted[:, 2]) & (x_sorted[:, 1] > x_sorted[:, 2])
    if idx[special].shape[0] != 0:
        x_sorted_1 = x_sorted[idx[special], 1].clone()
        x_sorted[idx[special], 1] = x_sorted[idx[special], 2]
        x_sorted[idx[special], 2] = x_sorted_1
    return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()


def polygon_b_inter_union_cpu(boxes1, boxes2):
    """
        iou computation (polygon) with cpu for class Polygon_ComputeLoss in loss.py;
        Boxes and Anchors having the same shape: nx8;
        Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
    """

    n = boxes1.shape[0]
    inter = torch.zeros(n,)
    union = torch.zeros(n,)
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4,2)).convex_hull
        polygon2 = shapely.geometry.Polygon(boxes2[i, :].view(4,2)).convex_hull
        if polygon1.intersects(polygon2):
            try:
                inter[i] = polygon1.intersection(polygon2).area
                union[i] = polygon1.union(polygon2).area
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured')
    return inter, union


def polygon_bbox_iou(boxes1, boxes2, eps=1e-7, device="cuda:0", ordered=False):
    """
        Compute iou of polygon boxes for class Polygon_ComputeLoss in loss.py via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
    """
    if isinstance(boxes1, (np.ndarray)):
        boxes1 = torch.from_numpy(boxes1)
    if isinstance(boxes2, (np.ndarray)):
        boxes2 = torch.from_numpy(boxes2)
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone().to(device)), order_corners(boxes2.clone().to(device))
    else:
        boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)
    
    if torch.cuda.is_available() and polygon_b_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside inter_union_cuda must be torch.cuda.float, not double type or half type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_b_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.
        
        inter_nan, union_nan = inter.isnan(), union.isnan()
        if inter_nan.any() or union_nan.any():
            inter2, union2 = polygon_b_inter_union_cuda(boxes1_, boxes2_)  # Careful that order should be: boxes1_, boxes2_.
            inter2, union2 = inter2.T, union2.T
            inter = torch.where(inter_nan, inter2, inter)
            union = torch.where(union_nan, union2, union)
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_b_inter_union_cpu(boxes1, boxes2)
    union += eps

    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0

    return iou


def xywha2xyxyxyxy(box):
    """
    use radian
    X=x*cos(a)-y*sin(a)
    Y=x*sin(a)+y*cos(a)
    """
    batch = box.shape[0]

    center = box[:,:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]*np.pi/180

    # calculate two vector
    verti = np.empty((batch,2), dtype=np.float32)
    verti[:,0] = (h/2) * np.sin(rad)
    verti[:,1] = - (h/2) * np.cos(rad)

    hori = np.empty((batch,2), dtype=np.float32)
    hori[:,0] = (w/2) * np.cos(rad)
    hori[:,1] = (w/2) * np.sin(rad)

    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    return np.concatenate([tl,tr,br,bl], axis=1)

if __name__ == "__main__":
    None
