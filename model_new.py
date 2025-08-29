import torch
import torch.nn as nn
import torch.nn.functional as F
import module_new as mm

class InpaintNN(nn.Module):
    def __init__(self, input_height=256, input_width=256, bar_checkpoint=None, mosaic_checkpoint=None, is_mosaic=False):
        super(InpaintNN, self).__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.is_mosaic = is_mosaic

        # Define components from your module.py
        self.encoder = mm.Encoder()                # originally mm.encoder(...)
        self.contextual_block = mm.ContextualBlock(channels=256, k_size=3, lamda=50)  # originally mm.contextual_block(...)
        self.decoder = mm.Decoder()                # originally mm.decoder(...)
        self.discriminator_red = mm.Discriminator_RED()  # originally mm.discriminator_red(...)

        # Load pretrained weights if given
        if bar_checkpoint and not is_mosaic:
            self.load_state_dict(torch.load(bar_checkpoint, map_location="cpu"))
        elif mosaic_checkpoint and is_mosaic:
            self.load_state_dict(torch.load(mosaic_checkpoint, map_location="cpu"))

    def forward(self, X, Y, MASK, iters=0):
        """
        X: censored input  (B,3,H,W)
        Y: ground truth    (B,3,H,W)
        MASK: mask         (B,3,H,W)
        iters: training iteration (for alpha schedule)
        """

        # Concatenate input & mask
        input_cat = torch.cat([X, MASK], dim=1)  # (B,6,H,W)

        # Encode
        vec_en = self.encoder(input_cat)

        # Contextual block
        vec_con = self.contextual_block(vec_en, vec_en, MASK)

        # Decode
        I_co = self.decoder(vec_en)   # coarse
        I_ge = self.decoder(vec_con)  # refined

        # Merge with ground truth using mask
        image_result = I_ge * (1 - MASK) + Y * MASK

        # Discriminator outputs
        D_real_red = self.discriminator_red(Y)
        D_fake_red = self.discriminator_red(image_result.detach())

        # --------- Losses ---------
        loss_D_red = torch.mean(F.relu(1 + D_fake_red)) + torch.mean(F.relu(1 - D_real_red))
        loss_D = loss_D_red

        loss_gan_red = -torch.mean(D_fake_red)
        loss_gan = loss_gan_red

        loss_s_re = torch.mean(torch.abs(I_ge - Y))
        loss_hat = torch.mean(torch.abs(I_co - Y))

        # SSIM surrogate: we’ll use luminance Y channel (approx.)
        A = 0.299 * image_result[:,0:1] + 0.587 * image_result[:,1:2] + 0.114 * image_result[:,2:3]
        B = 0.299 * Y[:,0:1] + 0.587 * Y[:,1:2] + 0.114 * Y[:,2:3]
        ssim = torch.mean((2*A*B + 1e-5) / (A**2 + B**2 + 1e-5))

        alpha = iters / 1e6
        loss_G = 0.1 * loss_gan + 10 * loss_s_re + 5 * (1 - alpha) * loss_hat

        return image_result, loss_D, loss_G, ssim

    def predict(self, X, Y, MASK):
        self.eval()
        with torch.no_grad():
            out = self.forward(X, Y, MASK)
        return out["image_result"]
