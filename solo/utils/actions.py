import torch


def get_crop_diffparams(src1, src2):
    cy1 = src1[:, 0] #+ 0.5 * src1[:, 2]
    cx1 = src1[:, 1] #+ 0.5 * src1[:, 3]
    h1, w1 = src1[:, 2], src1[:, 3]

    cy2 = src2[:, 0] #+ 0.5 * src2[:, 2]
    cx2 = src2[:, 1] #+ 0.5 * src2[:, 3]
    h2, w2 = src2[:, 2], src2[:, 3]


    d_cx = cx1 - cx2
    d_cy = cy1 - cy2
    d_sx = torch.sqrt(w1 / w2)
    d_sy = torch.sqrt(h1 / h2)

    # if len(src1) > 4:
    # return torch.stack((d_cx, d_cy, d_sx, d_sy, torch.abs(src1[:, 4] - src2[:, 4])), dim=1)
    return torch.stack((d_cx, d_cy, d_sx, d_sy), dim=1)