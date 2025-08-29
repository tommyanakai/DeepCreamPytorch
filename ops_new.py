import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random as rr
import math as mt
import cv2
import math


# ----------------------------
# Square mask
# ----------------------------
def make_sq_mask(size, m_size, batch_size):
    start_x = rr.randint(0, size - m_size - 1)
    start_y = rr.randint(0, size - m_size - 1)

    temp = np.ones([batch_size, size, size, 3], dtype=np.float32)
    temp[:, start_x:start_x + m_size, start_y:start_y + m_size, :] = 0

    return temp, start_x, start_y


# ----------------------------
# Softmax (TF-style)
# ----------------------------
def softmax(x):
    k = torch.exp(x - 3.0)
    denom = k.sum(dim=3, keepdim=True)
    return torch.exp(x - 3.0) / denom


# ----------------------------
# Variance & Std
# ----------------------------
def reduce_var(x, axis=None, keepdims=False):
    mean = torch.mean(x, dim=axis, keepdim=True)
    devs_squared = (x - mean) ** 2
    return torch.mean(devs_squared, dim=axis, keepdim=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return torch.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


# ----------------------------
# L2 normalization
# ----------------------------
def l2_norm(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)


# ----------------------------
# Free-form mask (single)
# ----------------------------
def ff_mask(size, b_size, maxLen, maxWid, maxAng, maxNum, maxVer,
            minLen=20, minWid=15, minVer=5):

    mask = np.ones((b_size, size, size, 3), dtype=np.float32)
    num = rr.randint(3, maxNum)

    for i in range(num):
        startX = rr.randint(0, size)
        startY = rr.randint(0, size)
        numVer = rr.randint(minVer, maxVer)
        width = rr.randint(minWid, maxWid)

        for j in range(numVer):
            angle = rr.uniform(-maxAng, maxAng)
            length = rr.randint(minLen, maxLen)

            endX = min(size - 1, max(0, int(startX + length * mt.sin(angle))))
            endY = min(size - 1, max(0, int(startY + length * mt.cos(angle))))

            lowx, highx = sorted([startX, endX])
            lowy, highy = sorted([startY, endY])

            if abs(startY - endY) + abs(startX - endX) != 0:
                wlx = max(0, lowx - int(abs(width * mt.cos(angle))))
                whx = min(size - 1, highx + 1 + int(abs(width * mt.cos(angle))))
                wly = max(0, lowy - int(abs(width * mt.sin(angle))))
                why = min(size - 1, highy + 1 + int(abs(width * mt.sin(angle))))

                for x in range(wlx, whx):
                    for y in range(wly, why):
                        d = abs((endY - startY) * x - (endX - startX) * y - endY * startX + startY * endX) / mt.sqrt(
                            (startY - endY) ** 2 + (startX - endX) ** 2)
                        if d <= width:
                            mask[:, x, y, :] = 0

            # circles at endpoints
            for (cx, cy) in [(startX, startY), (endX, endY)]:
                for x2 in range(max(0, cx - width), min(size, cx + width + 1)):
                    for y2 in range(max(0, cy - width), min(size, cy + width + 1)):
                        if np.sqrt((cx - x2) ** 2 + (cy - y2) ** 2) <= width:
                            mask[:, x2, y2, :] = 0

            startX, startY = endX, endY

    return mask


# ----------------------------
# Free-form mask batch
# ----------------------------
def ff_mask_batch(size, b_size, maxLen, maxWid, maxAng, maxNum, maxVer,
                  minLen=20, minWid=15, minVer=5):

    temp = ff_mask(size, 1, maxLen, maxWid, maxAng, maxNum, maxVer,
                   minLen=minLen, minWid=minWid, minVer=minVer)[0]
    mask = None
    for ib in range(b_size):
        if ib == 0:
            mask = np.expand_dims(temp, 0)
        else:
            mask = np.concatenate((mask, np.expand_dims(temp, 0)), 0)

        temp = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
        if ib == 3:
            temp = cv2.flip(temp, 0)
    return mask


# ----------------------------
# Spectral Normalization (manual)
# ----------------------------
def spectral_norm(w, u=None, iteration=1, eps=1e-12):
    w_mat = w.view(w.size(0), -1)
    if u is None:
        u = F.normalize(torch.randn(1, w_mat.size(0), device=w.device), dim=1, eps=eps)

    for _ in range(iteration):
        v = F.normalize(torch.matmul(u, w_mat), dim=1, eps=eps)
        u = F.normalize(torch.matmul(v, w_mat.t()), dim=1, eps=eps)

    sigma = torch.matmul(torch.matmul(u, w_mat), v.t())
    w_norm = w_mat / sigma
    return w_norm.view_as(w), u


# ----------------------------
# Convolution + Spectral Norm
# ----------------------------
class ConvolutionSN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        # weight [out_channels, in_channels, k, k]
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.u = None   # for spectral norm tracking

    def forward(self, x):
        w_sn, self.u = spectral_norm(self.weight, self.u)

        # --- SAME padding like TensorFlow ---
        if self.stride == 1:
            padding = self.kernel_size // 2
            return F.conv2d(x, w_sn, bias=self.bias, stride=1, padding=padding)

        else:
            in_h, in_w = x.shape[2:]
            out_h = math.ceil(in_h / self.stride)
            out_w = math.ceil(in_w / self.stride)

            pad_h = max((out_h - 1) * self.stride + self.kernel_size - in_h, 0)
            pad_w = max((out_w - 1) * self.stride + self.kernel_size - in_w, 0)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            return F.conv2d(x, w_sn, bias=self.bias, stride=self.stride, padding=0)


# ----------------------------
# Dense RED + SN
# ----------------------------
class DenseRED_SN(nn.Module):
    def __init__(self, h, w, c, name=None):
        super(DenseRED_SN, self).__init__()
        self.h, self.w, self.c = h, w, c

        # weight shape: (h*w, 1, c, 1) in TF → treat as parameter
        self.weight = nn.Parameter(torch.empty(h * w, 1, c, 1))
        nn.init.xavier_uniform_(self.weight)

        # bias shape: (1, h, w, 1) → PyTorch (1, 1, h, w)
        self.bias = nn.Parameter(torch.zeros(1, 1, h, w))

    def forward(self, x):
        """
        x: [B, C, H, W] (PyTorch convention)
        """
        B, C, H, W = x.shape
        assert H == self.h and W == self.w and C == self.c, \
            f"Expected ({self.c}, {self.h}, {self.w}), got ({C}, {H}, {W})"

        # Apply spectral norm per pixel weight
        sn_w_list = []

        for it in range(self.h * self.w):
            w_pixel = self.weight[it:it + 1, :, :, :]  # (1, 1, c, 1)
            sn_w_pixel, _ = spectral_norm(w_pixel)  # Use spectral_norm from previous snippet
            sn_w_list.append(sn_w_pixel)

        sn_w = torch.cat(sn_w_list, dim=0)  # (h*w, ?)
        sn_w = sn_w.view(self.h, self.w, self.c, 1).permute(3, 2, 0, 1)  # (1, h, w, c)
        # broadcast multiply
        out = (x * sn_w).sum(dim=1, keepdim=True) + self.bias  # reduce along channel

        return out
