
import torch
import torch.nn as nn
import torch.nn.functional as F
import ops_new as ops  # the converted ops.py above


# small conv/convnn blocks (unchanged semantic behavior)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, dilation=1):
        super().__init__()
        pad = (dilation, dilation, dilation, dilation)
        self.pad = pad
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=stride,
                              padding=0, dilation=dilation)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        x = F.pad(x, self.pad, mode='reflect')
        x = self.conv(x)
        return self.act(x)


class ConvNN(nn.Module):
    def __init__(self, in_ch, dims1, dims2, out_h, out_w, k_size=3):
        super().__init__()
        self.pad = (1,1,1,1)
        self.conv1 = nn.Conv2d(in_ch, dims1, kernel_size=k_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dims1, dims2, kernel_size=k_size, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv1.weight); nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.xavier_normal_(self.conv2.weight); nn.init.constant_(self.conv2.bias, 0.0)
        self.out_h, self.out_w = out_h, out_w
        self.act = nn.ELU(inplace=True)
    def forward(self, x):
        x = F.pad(x, self.pad, mode='reflect'); x = self.act(self.conv1(x))
        x = F.pad(x, self.pad, mode='reflect'); x = self.act(self.conv2(x))
        return F.interpolate(x, size=(self.out_h, self.out_w), mode='nearest')

# ---------------------------
# Encoder
# ---------------------------


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Normal conv layers
        self.cl1 = nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=0)
        self.cl2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.cl3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.cl4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.cl5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.cl6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)

        # Dilated conv layers (all padding=0, dilation>1)
        self.dcl1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=2)
        self.dcl2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=4)
        self.dcl3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=8)
        self.dcl4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=16)

    def forward(self, x):

        x = F.elu(self.cl1(F.pad(x, (2, 2, 2, 2), mode='reflect')))
        x = F.elu(self.cl2(F.pad(x, (1, 1, 1, 1), mode='reflect')))
        x = F.elu(self.cl3(F.pad(x, (1, 1, 1, 1), mode='reflect')))
        x = F.elu(self.cl4(F.pad(x, (1, 1, 1, 1), mode='reflect')))
        x = F.elu(self.cl5(F.pad(x, (1, 1, 1, 1), mode='reflect')))
        x = F.elu(self.cl6(F.pad(x, (1, 1, 1, 1), mode='reflect')))
        x = F.elu(self.dcl1(F.pad(x, (2, 2, 2, 2), mode='reflect')))
        x = F.elu(self.dcl2(F.pad(x, (4, 4, 4, 4), mode='reflect')))
        x = F.elu(self.dcl3(F.pad(x, (8, 8, 8, 8), mode='reflect')))
        x = F.elu(self.dcl4(F.pad(x, (16, 16, 16, 16), mode='reflect')))
        return x

# ---------------------------
# Decoder
# ---------------------------
class Decoder(nn.Module):
    def __init__(self, size1=256, size2=256):
        super().__init__()
        self.dl1 = ConvNN(256, 128, 128, size1 // 4, size2 // 4)
        self.dl2 = ConvNN(128, 64, 64, size1 // 2, size2 // 2)
        self.dl3 = ConvNN(64, 32, 32, size1, size2)
        self.dl4 = ConvNN(32, 16, 16, size1, size2)
        self.out_conv = nn.Conv2d(16, 3, 3, stride=1, padding=1)
        nn.init.xavier_normal_(self.out_conv.weight); nn.init.constant_(self.out_conv.bias, 0.0)

    def forward(self, x):
        x = self.dl1(x)
        x = self.dl2(x)
        x = self.dl3(x)
        x = self.dl4(x)
        x = self.out_conv(x)
        x = torch.clamp(x, -1.0, 1.0)
        return x



# -----------------------------
# Discriminator (RED variant)
# -----------------------------
class Discriminator_RED(nn.Module):
    def __init__(self, input_c=3, h=256, w=256):
        super(Discriminator_RED, self).__init__()

        self.l1 = ops.ConvolutionSN(input_c, 64, 5, 2)
        self.l2 = ops.ConvolutionSN(64, 128, 5, 2)
        self.l3 = ops.ConvolutionSN(128, 256, 5, 2)
        self.l4 = ops.ConvolutionSN(256, 256, 5, 2)
        self.l5 = ops.ConvolutionSN(256, 256, 5, 2)
        self.l6 = ops.ConvolutionSN(256, 512, 5, 2)

        # dense_RED_SN at the end
        self.l7 = ops.DenseRED_SN(h // (2 ** 6), w // (2 ** 6), 512)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.l2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.l3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.l4(x), 0.2, inplace=True)
        x = F.leaky_relu(self.l5(x), 0.2, inplace=True)
        x = F.leaky_relu(self.l6(x), 0.2, inplace=True)

        x = self.l7(x)  # RED spectral normalized dense
        return x


class ContextualBlock(nn.Module):
    def __init__(self, channels, k_size, lamda, stride=1):
        super(ContextualBlock, self).__init__()
        self.channels = channels
        self.k_size = k_size
        self.lamda = lamda
        self.stride = stride

        # 1x1 fusion conv at the end
        self.fuse_conv = nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, bg_in, fg_in, mask):
        """
        bg_in: [B, C, H, W]
        fg_in: [B, C, H, W]
        mask:  [B, 1, H0, W0]  (or [B, H0, W0] or NHWC variant) - nearest-resized to H,W and first channel used
        """
        B, C, H, W = bg_in.shape
        k_size, stride = self.k_size, self.stride
        pad = k_size // 2
        eps = 1e-6

        # --- normalize mask to [B,1,H0,W0] (take first channel if multi-channel or NHWC)
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask_in = mask
        elif mask.dim() == 4 and mask.shape[-1] == 1:  # NHWC possibly
            mask_in = mask.permute(0, 3, 1, 2)[:, :1, :, :]
        elif mask.dim() == 3:
            mask_in = mask.unsqueeze(1)
        else:
            # fallback: take first channel
            mask_in = mask[:, :1, :, :]

        mask_in = mask_in.to(dtype=bg_in.dtype, device=bg_in.device)

        # Resize mask to match H,W using nearest (same as tf.image.resize_nearest_neighbor)
        mask_rs = F.interpolate(mask_in, size=(H, W), mode='nearest')  # [B,1,H,W]
        mask_r = mask_rs.expand(-1, C, -1, -1)  # [B,C,H,W]

        # masked background
        bg = bg_in * mask_r  # [B,C,H,W]

        # Extract bg patches with VALID padding and stride=self.stride
        # unfold_bg: [B, C*k*k, L] where L is number of background patch centers
        unfold_bg = F.unfold(bg, kernel_size=k_size, dilation=1, padding=0, stride=stride)
        L = unfold_bg.shape[-1]  # number of background patches (c in TF)

        # Extract fg patches with stride=1 and SAME padding (pad computed above)
        unfold_fg = F.unfold(fg_in, kernel_size=k_size, dilation=1, padding=pad, stride=1)  # [B, C*k*k, H*W]
        # reshape to [B, H, W, C*k*k]
        k1dim = C * k_size * k_size
        unfold_fg = unfold_fg.transpose(1, 2).contiguous().view(B, H, W, k1dim)

        ACL_list = []
        # iterate batch (TF did this per-batch)
        for ib in range(B):
            # unfold_bg[ib]: [C*k*k, L]
            k1_cols = unfold_bg[ib]  # shape: [k1dim, L]

            # squared norm of each background patch vector: [L]
            k1d = (k1_cols ** 2).sum(dim=0)  # [L]

            # Prepare weight for 1x1 conv (similarity): PyTorch conv weight shape: [out_channels, in_channels, 1, 1]
            # out_channels = L, in_channels = k1dim
            weight_k1 = k1_cols.t().contiguous().view(L, k1dim, 1, 1)  # [L, k1dim, 1, 1]

            # k2: patches reshaped to conv-transpose weight
            # want weight shape for conv_transpose2d: [in_channels=L, out_channels=C, k, k]
            # start with k1_cols.T shape [L, k1dim] and reshape to [L, C, k, k]
            k2 = k1_cols.t().contiguous().view(L, C, k_size, k_size)  # [L, C, k, k]

            # fg vectors for this batch: ww [H, W, k1dim]
            ww = unfold_fg[ib]  # [H, W, k1dim]
            # sums of squares per spatial location
            wwd = (ww ** 2).sum(dim=2, keepdim=True)  # [H, W, 1]

            # ft: [1, in_channels=k1dim, H, W]
            ft = ww.permute(2, 0, 1).unsqueeze(0).contiguous()

            # similarity conv: conv2d(ft, weight_k1) -> [1, out_channels=L, H, W]
            # then permute to [1, H, W, L] to match TF shapes
            CS = F.conv2d(ft, weight=weight_k1, bias=None, stride=1, padding=0)  # [1, L, H, W]
            CS = CS.permute(0, 2, 3, 1).contiguous()  # [1, H, W, L]

            # tt: broadcast addition of k1d and wwd -> shape [1, H, W, L]
            tt = k1d.view(1, 1, 1, L) + wwd.unsqueeze(0)  # [1, H, W, L]

            # DS1, DS2 normalization across patch-dimension (L)
            DS1 = tt - 2.0 * CS  # [1, H, W, L]
            mean = DS1.mean(dim=3, keepdim=True)
            std = DS1.std(dim=3, unbiased=False, keepdim=True)
            DS2 = (DS1 - mean) / (std + eps)
            DS2 = -torch.tanh(DS2)

            # attention weights CA over the L background patches for each spatial location
            CA = F.softmax(self.lamda * DS2, dim=3)  # [1, H, W, L]

            # prepare CA as conv_transpose input: [N, in_channels=L, H, W]
            CA_p = CA.permute(0, 3, 1, 2).contiguous()  # [1, L, H, W]

            # Conv-transpose: input channels = L, out_channels = C, kernel = (k,k)
            ACLt = F.conv_transpose2d(CA_p, weight=k2, bias=None, stride=1, padding=pad)  # [1, C, H_out, W_out]

            # crop center to exactly H,W (TF used output_shape to force that)
            _, _, H_out, W_out = ACLt.shape
            start_h = (H_out - H) // 2
            start_w = (W_out - W) // 2
            if start_h != 0 or start_w != 0:
                ACLt = ACLt[:, :, start_h:start_h + H, start_w:start_w + W]
            else:
                ACLt = ACLt[:, :, :H, :W]

            ACLt = ACLt / (k_size ** 2)

            ACL_list.append(ACLt)  # each ACLt: [1, C, H, W]

        # stack
        ACL = torch.cat(ACL_list, dim=0)  # [B, C, H, W]

        # blend with mask (same as TF: ACL = bg + ACL * (1 - mask_r))
        ACL = bg + ACL * (1.0 - mask_r)

        # fuse
        con1 = torch.cat([bg_in, ACL], dim=1)  # [B, 2C, H, W]
        out = F.elu(self.fuse_conv(con1))

        return out