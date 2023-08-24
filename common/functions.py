import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.einops import rearrange, repeat
import cv2
import pydegensac
import random

def mask_border(m, b, v):
    m[:, :b]=v
    m[:, :, :b]=v
    m[:, :, :, :b]=v
    m[:, :, :, :, :b]=v
    m[:, -b:0]=v
    m[:, :, -b:0]=v
    m[:, :, :, -b:0]=v
    m[:, :, :, :, -b:0]=v

def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd]= v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0-bd:] = v
        m[b_idx, :, w0-bd:] = v
        m[b_idx, :, :, h1-bd:] = v
        m[b_idx, :, :, h1-bd:] = v

def cal_reproj_dists(p1s, p2s, homography):
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist

def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors = np.append([0.], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs = []
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rcs_ = np.append(recalls[:last_index], recalls[last_index-1])
        errs_ = np.append(errors[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return np.array(aucs)

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid 

def _transform_inv(img,mean,std):
    img = img * std + mean
    img  = np.uint8(img * 255.0)
    img = img.transpose(1,2,0)
    return img

def draw_match(match_mask,query,refer,x, y, homo_filter=True,patch_size=8):
    grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    _pts = grid[match_mask]
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in _pts:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[0] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[0] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if query_pts.shape[0] > 0:
        xx_mean = np.mean((refer_pts - query_pts-x-512)[:, 0], axis=0)
        yy_mean = np.mean((refer_pts - query_pts-y)[:, 1], axis=0)
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match2(matches,query,refer,x, y, homo_filter=True,patch_size=8, th=2):
    matches = matches.detach().cpu().numpy()
    #grid = make_grid(matches.shape[1],matches.shape[0])
    #_pts = grid[matches]
    query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        if np.abs(x1-512-x0-x) < th and np.abs(y1-y-y0) < th:
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if len(query_pts) > 0:
        xx_means = int(np.mean((refer_pts - query_pts-x-512)[:, 0], axis=0))
        yy_means = int(np.mean((refer_pts - query_pts-y)[:, 1], axis=0))
    else:
        xx_means, yy_means, xx_std, yy_std = 0,0,0,0
    if query_pts.shape[0] > 0:
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
        #cv2.rectangle(out_img, (x+512+xx_means,y+yy_means), (x+512+512+xx_means, y+512+yy_means), (0, 255, 0), 2)

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    cv2.putText(out_img, 'mean:' + str(xx_means) + ' ' + str(yy_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match3(matches,query,refer,x, y, homo_filter=True,patch_size=8):
    matches = matches.detach().cpu().numpy()
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = []
    refer_pts = []
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    return out_img

def draw_match4(mkpts0, mkpts1, query,refer, match_mask, homo_filter=True,patch_size=8):

    def get_gt_match(match_mask):
        grid = make_grid(match_mask.shape[1],match_mask.shape[0])
        _pts = grid[match_mask]
        #out_img = np.concatenate([query,refer],axis=1).copy()
        query_pts = []
        refer_pts = []
        wq = query.shape[1]
        wr = refer.shape[1]
        qcols = wq // patch_size
        rcols = wr // patch_size
        for pt in _pts:
            x0 = patch_size/2 + (pt[1] % qcols) * patch_size
            y0 = patch_size/2 + (pt[1] // qcols) * patch_size
            x1 = patch_size/2 + (pt[0] % rcols) * patch_size
            y1 = patch_size/2 + (pt[0] // rcols) * patch_size
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
            # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
        query_pts = np.asarray(query_pts,np.float32)
        refer_pts = np.asarray(refer_pts,np.float32)

        return query_pts, refer_pts

    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    gt_query_pts, gt_refer_pts = get_gt_match(match_mask)
    mkpts11, mkpts10 = [],[]
    if query_pts.shape[1] == 0 or gt_query_pts.shape[0] == 0:
        return out_img
    for idx, mkp in enumerate(query_pts):
        m_gt_mkpts1 = gt_refer_pts[np.where((gt_query_pts == mkp).all(1))]
        if m_gt_mkpts1.shape[0]!=0:
            if np.linalg.norm(refer_pts[idx] - m_gt_mkpts1.squeeze()) < 10:
                mkpts10.append(query_pts[idx])
                mkpts11.append(refer_pts[idx])

    if len(mkpts10) > 0:
        query_pts, refer_pts = np.stack(mkpts10), np.stack(mkpts11)
        refer_pts[:, 0] = refer_pts[:, 0] + 320
    else:
        query_pts, refer_pts = np.array([[]]), np.array([[]])

    if homo_filter and query_pts.shape[0] > 1:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    """
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    """
    cv2.putText(out_img, str(len(mkpts10)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match5(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=10):
    query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + 512
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-512-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])
            x_mean.append(x1-512-x0-x)
            y_mean.append(y1-y-y0)

    if len(x_mean) != 0:
        x_means = int(np.mean(x_mean))
        y_means = int(np.mean(y_mean))
    else:
        x_means,y_means = 0,0

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if query_pts.shape[0] > 0:
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
        #cv2.rectangle(out_img, (x+512+x_means,y+y_means), (x+512+512, y+512), (0, 255, 0), 2)


    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    cv2.putText(out_img, 'mean:' + str(x_means) + ' ' + str(y_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def get_mkpts(matches, query, refer, patch_size=8):
    #grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    #_pts = grid[match_mask]
    #matches = match_mask.detach().cpu().numpy()
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        query_pts.append(torch.Tensor([x0, y0]))
        refer_pts.append(torch.Tensor([x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts

def batch_get_mkpts(matches, query, refer, patch_size=8):
    #grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    #_pts = grid[match_mask]
    #matches = match_mask.detach().cpu().numpy()
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        query_pts.append(torch.Tensor([pt[0], x0, y0]))
        refer_pts.append(torch.Tensor([pt[0], x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 3).cuda()
        refer_pts = torch.Tensor(0, 3).cuda()

    return query_pts, refer_pts



def rocket_score(mkpts0, mkpts1, query, refer, x, y, th=2):
    query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()

    #query_pts = q_pts[np.where((r_pts-q_pts)[:, 1] < 259)[0]]
    #refer_pts = r_pts[np.where((r_pts-q_pts)[:, 1] < 259)[0]]
    query_pts, refer_pts  = query_pts[np.where((refer_pts-query_pts)[:, 1] > 0)[0]], refer_pts[np.where((refer_pts-query_pts)[:, 1] > 0)[0]]
    query_pts, refer_pts  = query_pts[np.where((refer_pts-query_pts)[:, 1] < 289)[0]], refer_pts[np.where((refer_pts-query_pts)[:, 1] < 289)[0]]
    query_pts, refer_pts  = query_pts[np.where((refer_pts-query_pts)[:, 0] > 0)[0]], refer_pts[np.where((refer_pts-query_pts)[:, 0] > 0)[0]]
    query_pts, refer_pts  = query_pts[np.where((refer_pts-query_pts)[:, 0] < 289)[0]], refer_pts[np.where((refer_pts-query_pts)[:, 0] < 289)[0]]

    refer_pts[:, 0] = refer_pts[:, 0] + 512


    if len(refer_pts) == 0:
        return out_img, (0,0)
    H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
    q_pts = []
    r_pts = []
    for i in range(query_pts.shape[0]):
        if mask[i]:
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            q_pts.append(query_pts[i])
            r_pts.append(refer_pts[i])
    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    #print(refer_pts-query_pts)
    #indexes = [x/10. for x in range(60)]
    #scores = [round(x+0.1, 1) if x>=1 else 0 for x in indexes[::-1]]
    try:
        r_x_means = int(np.mean((refer_pts)[:, 0], axis=0))
        r_y_means = int(np.mean((refer_pts)[:, 1], axis=0))
        q_x_means = int(np.mean((query_pts)[:, 0], axis=0))
        q_y_means = int(np.mean((query_pts)[:, 1], axis=0))
    except:
        r_x_means = r_y_means = q_x_means = q_y_means = 0


    real_x, real_y = r_x_means - q_x_means - 512, r_y_means - q_y_means
    cv2.rectangle(out_img, (real_x+512,real_y), (real_x+512+512, real_y+512), (0, 255, 255), 2)
    print(x, y, "=>", real_x, real_y)


    return out_img, (real_x, real_y)


def rocket_match_num(mkpts0, mkpts1, query, refer, x, y, match_prec):
    query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + 512
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
    query_pts = [x for i, x in enumerate(query_pts) if mask[i]]
    refer_pts = [x for i, x in enumerate(refer_pts) if mask[i]]
    for pix_th in range(1, 11):
        q_pts = []
        r_pts = []
        for q_pt, r_pt in zip(query_pts, refer_pts):
            x0, y0 = q_pt
            x1, y1 = r_pt
            if np.abs(x1-512-x0-x) < pix_th and np.abs(y1-y-y0) < pix_th:
                q_pts.append([x0,y0])
                r_pts.append([x1,y1])
                x_mean.append(x1-512-x0-x)
                y_mean.append(y1-y-y0)

        match_prec[pix_th].append(len(q_pts) / len(query_pts))

    return match_prec

def rocket_match_num_with_gt(match_mask, mkpts0, mkpts1, query, refer, match_prec, patch_size=8):
    def get_gt_match(match_mask):
        grid = make_grid(match_mask.shape[1],match_mask.shape[0])
        _pts = grid[match_mask]
        #out_img = np.concatenate([query,refer],axis=1).copy()
        query_pts = []
        refer_pts = []
        wq = query.shape[1]
        wr = refer.shape[1]
        qcols = wq // patch_size
        rcols = wr // patch_size
        for pt in _pts:
            x0 = patch_size/2 + (pt[1] % qcols) * patch_size
            y0 = patch_size/2 + (pt[1] // qcols) * patch_size
            x1 = patch_size/2 + (pt[0] % rcols) * patch_size
            y1 = patch_size/2 + (pt[0] // rcols) * patch_size
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
            # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
        query_pts = np.asarray(query_pts,np.float32)
        refer_pts = np.asarray(refer_pts,np.float32)

        return query_pts, refer_pts

    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    gt_query_pts, gt_refer_pts = get_gt_match(match_mask)
    if len(gt_query_pts) == 0:
        return match_prec
    for i in range(1, 11):
        mkpts11, mkpts10 = [],[]
        if query_pts.shape[1] == 0:
            match_prec[i].append(0)
            continue
        for idx, mkp in enumerate(query_pts):
            m_gt_mkpts1 = gt_refer_pts[np.where((gt_query_pts == mkp).all(1))]
            if m_gt_mkpts1.shape[0]!=0:
                if np.linalg.norm(refer_pts[idx] - m_gt_mkpts1.squeeze()) < i:
                    mkpts10.append(query_pts[idx])
                    mkpts11.append(refer_pts[idx])

        if len(mkpts10) > 0:
            q_pts, r_pts = np.stack(mkpts10), np.stack(mkpts11)
        else:
            q_pts, r_pts = np.array([[]]), np.array([[]])

        match_prec[i].append( len(q_pts) / len(gt_query_pts))

    return match_prec

def draw_match_nir(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=10):
    #query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + 512
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-512-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])
            x_mean.append(x1-512-x0-x)
            y_mean.append(y1-y-y0)

    if len(x_mean) != 0:
        x_means = int(np.mean(x_mean))
        y_means = int(np.mean(y_mean))
    else:
        x_means,y_means = 0,0

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)


    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            # else:
            #     cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    # cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    # cv2.putText(out_img, 'mean:' + str(x_means) + ' ' + str(y_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match_rgbd(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=1e3):
    #query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + 640
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-640-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            #else:
            #    cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    #cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    #cv2.putText(out_img, 'mean:' + str(x_means) + ' ' + str(y_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match_se(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=1e3):
    #query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    #refer = cv2.copyMakeBorder(refer, 128, 0, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + wq
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-wq-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])
            x_mean.append(x1-wq-x0-x)
            y_mean.append(y1-y-y0)


    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if abs(query_pts[i, 0]- refer_pts[i, 0] + 512) < 4 and abs(query_pts[i, 1] - refer_pts[i, 1]) < 4:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    #cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    #cv2.putText(out_img, 'mean:' + str(x_means) + ' ' + str(y_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match_sp(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=10):
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + wq
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-wq-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])
            x_mean.append(x1-wq-x0-x)
            y_mean.append(y1-y-y0)

    if len(x_mean) != 0:
        x_means = int(np.mean(x_mean))
        y_means = int(np.mean(y_mean))
    else:
        x_means,y_means = 0,0

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    cv2.putText(out_img, 'mean:' + str(x_means) + ' ' + str(y_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def eval_mma(match_mask, mkpts0, mkpts1, query, refer, i_err, thres=[1,2,3,4,5,6,7,8,9,10],patch_size=8, ransac_thres=3):
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()

    def get_gt_match(match_mask):
        grid = make_grid(match_mask.shape[1],match_mask.shape[0])
        _pts = grid[match_mask]
        query_pts = []
        refer_pts = []
        wq = query.shape[1]
        wr = refer.shape[1]
        qcols = wq // patch_size
        rcols = wr // patch_size
        for pt in _pts:
            x0 = patch_size/2 + (pt[1] % qcols) * patch_size
            y0 = patch_size/2 + (pt[1] // qcols) * patch_size
            x1 = patch_size/2 + (pt[0] % rcols) * patch_size
            y1 = patch_size/2 + (pt[0] // rcols) * patch_size
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
        query_pts = np.asarray(query_pts,np.float32)
        refer_pts = np.asarray(refer_pts,np.float32)

        return query_pts, refer_pts

    q_pts, r_pts = get_gt_match(match_mask)
    if q_pts.shape[0]== 0:
        return i_err, 0
    H_gt, inliers = cv2.findHomography(q_pts, r_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thres)
    if len(mkpts1) == 0 or H_gt is None:
        dist = np.array([float("inf")])
    else:
        dist = cal_reproj_dists(mkpts0, mkpts1, H_gt)
    for thr in thres:
        i_err[thr] += np.mean(dist <= thr)
    return i_err, 1

def eval_homography(match_mask, mkpts0, mkpts1, query, refer,patch_size=8, ransac_thres=3):
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    h,w = query.shape[:2]

    def get_gt_match(match_mask):
        grid = make_grid(match_mask.shape[1],match_mask.shape[0])
        _pts = grid[match_mask]
        #out_img = np.concatenate([query,refer],axis=1).copy()
        query_pts = []
        refer_pts = []
        wq = query.shape[1]
        wr = refer.shape[1]
        qcols = wq // patch_size
        rcols = wr // patch_size
        for pt in _pts:
            x0 = patch_size/2 + (pt[1] % qcols) * patch_size
            y0 = patch_size/2 + (pt[1] // qcols) * patch_size
            x1 = patch_size/2 + (pt[0] % rcols) * patch_size
            y1 = patch_size/2 + (pt[0] // rcols) * patch_size
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
            # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
        query_pts = np.asarray(query_pts,np.float32)
        refer_pts = np.asarray(refer_pts,np.float32)

        return query_pts, refer_pts
    q_pts, r_pts = get_gt_match(match_mask)
    if q_pts.shape[0]== 0:
        return np.nan
    if mkpts0.shape[0] == 0:
        return np.nan
    H_gt, inliers = cv2.findHomography(q_pts, r_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thres)
    if H_gt is None:
        return np.nan
    corners = np.array([[0,0,1],[0,w-1,1],[h-1,0,1],[h-1,w-1,1]])
    H_pred, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=ransac_thres)

    real_warped_corners = np.dot(corners, np.transpose(H_gt))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    if H_pred is None:
        return np.nan
    warped_corners = np.dot(corners, np.transpose(H_pred))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    print(corner_dist)
    return corner_dist

def eval_src_mma(mkpts0, mkpts1, query, refer, i_err, thres=[1,2,3,4,5,6,7,8,9,10],patch_size=8, ransac_thres=3):
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    if len(mkpts1) >3:
        H_pred, inliers = pydegensac.findHomography(mkpts0, mkpts1, ransac_thres)
        if len([mkpts0[x, :] for x in np.where(inliers>0)[0]]) > 0:
            mkpts0 = np.stack([mkpts0[x, :] for x in np.where(inliers>0)[0]])
            mkpts1= np.stack([mkpts1[x, :] for x in np.where(inliers>0)[0]])
    H_gt = np.array([[1,0,0],[0,1,0],[0,0,1]])

    if len(mkpts1) == 0 or H_gt is None:
        dist = np.array([float("inf")])
    else:
        dist = cal_reproj_dists(mkpts0, mkpts1, H_gt)
    for thr in thres:
        i_err[thr] += np.mean(dist <= thr)
    return i_err, 1

def eval_src_homography(mkpts0, mkpts1, query, refer,patch_size=8, ransac_thres=3):
    np.set_printoptions(suppress=True)
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    h,w = query.shape[:2]
    H_gt = np.array([[1,0,0],[0,1,0],[0,0,1]])
    if mkpts0.shape[0] < 4:
        return np.nan

    corners = np.array([[0,0,1],[0,w-1,1],[h-1,0,1],[h-1,w-1,1]])
    #H_pred, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=ransac_thres)
    H_pred, inliers = pydegensac.findHomography(mkpts0, mkpts1, ransac_thres)

    if H_pred is None:
        return np.nan
    real_warped_corners = np.dot(corners, np.transpose(H_gt))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H_pred))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    #print(H_pred)
    #print(corner_dist)
    return corner_dist

def draw_match_kpts(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=1e3):
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + wq
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-wq-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        pts = random.sample([(q_pts[i], r_pts[i]) for i in range(query_pts.shape[0]) if mask[i]], 5)
        for i in range(len(pts)):
            cv2.rectangle(out_img, (int(pts[i][0][0]-8),int(pts[i][0][1]-8)), (int(pts[i][0][0]+8),int(pts[i][0][1]+8)), (0, 0, 255), 1)
            cv2.circle(out_img, (int(pts[i][0][0]),int(pts[i][0][1])),2, (0, 255, 255), -1)
            cv2.rectangle(out_img, (int(pts[i][1][0]-8),int(pts[i][1][1]-8)), (int(pts[i][1][0]+8),int(pts[i][1][1]+8)), (0, 0, 255), 1)
            cv2.circle(out_img, (int(pts[i][1][0]),int(pts[i][1][1])), 2,(0, 255, 255), -1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    #cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    #cv2.putText(out_img, 'mean:' + str(x_means) + ' ' + str(y_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img
