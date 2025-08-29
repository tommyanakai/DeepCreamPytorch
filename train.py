# train_inpaint.py  (Not Tested Yet!)
import os
import glob
from PIL import Image
import random
import math
from tqdm import tqdm
from model_new import InpaintNN  
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

# ---- Import your module file ----
# Assume your class is defined in inpaint_module.py, change if needed:
# from inpaint_module import InpaintNN
# For this example we assume the class definition you posted is available as InpaintNN in the current scope.
# If it's in another file, uncomment the import above and remove the next mock import.
# -----------------------------------------------------------------------------
# If your class is inside a file named "module.py" do:
# from module import InpaintNN
# -----------------------------------------------------------------------------

# ---------------------------
# Dataset: paired images + masks
# ---------------------------
class PairedImageMaskDataset(Dataset):
    """
    Expects structure:
        root/images/*.png  (ground-truth images)
        root/masks/*.png   (binary masks, same filenames)
    Produces (X, Y, MASK) where:
       Y: ground truth image (3 x H x W)
       X: censored input = Y * MASK + noise for masked area
       MASK: 3-channel mask with 1 on valid pixels, 0 where missing (same as your forward expects MASK such that result = I_ge*(1-MASK) + Y*MASK)
    """
    def __init__(self, root, input_height=256, input_width=256, augment=True):
        super().__init__()
        self.img_paths = sorted(glob.glob(os.path.join(root, "images", "*.*")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, "masks", "*.*")))
        assert len(self.img_paths) > 0, "No images found"
        # If masks are missing for some, try to match by filename:
        img_basenames = [os.path.splitext(os.path.basename(p))[0] for p in self.img_paths]
        mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in self.mask_paths}
        # Keep only pairs with both
        paired = []
        for p, name in zip(self.img_paths, img_basenames):
            if name in mask_dict:
                paired.append((p, mask_dict[name]))
        if len(paired) == 0:
            raise RuntimeError("No matching image/mask pairs found. Check dataset structure.")
        self.pairs = paired

        self.resize = T.Resize((input_height, input_width))
        self.to_tensor = T.ToTensor()
        self.augment = augment
        # simple augmentations
        self.hflip = T.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, mask_p = self.pairs[idx]
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")  # single channel

        # Resize
        img = self.resize(img)
        mask = self.resize(mask)

        # augmentation
        if self.augment and random.random() < 0.5:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)

        Y = self.to_tensor(img)  # (3, H, W) in range [0,1]
        mask_t = self.to_tensor(mask)  # (1, H, W), mask pixel values in [0,1]
        # Convert to binary mask (1 where valid, 0 where hole). Assume mask white=valid (>=0.5)
        mask_bin = (mask_t > 0.5).float()

        # Make 3-channel mask because your network expects (B,3,H,W)
        MASK = mask_bin.repeat(3, 1, 1)

        # Create censored input X: keep masked (valid) pixels, set holes to 0 (or noise)
        # Based on your forward: image_result = I_ge * (1 - MASK) + Y * MASK
        # Here MASK==1 means we keep Y (valid), MASK==0 means hole
        X = Y * MASK + (1 - MASK) * 0.0  # zeros in holes; you could use noise instead

        return X, Y, MASK

# ---------------------------
# Helpers
# ---------------------------
def save_checkpoint(model, optimizer_G, optimizer_D, iters, save_dir="checkpoints", name="inpaint"):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "iters": iters,
        "model_state": model.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict()
    }
    path = os.path.join(save_dir, f"{name}_iter_{iters}.pth")
    torch.save(state, path)
    print("Saved checkpoint:", path)

def load_checkpoint_if_exists(model, optimizer_G, optimizer_D, ckpt_path):
    if ckpt_path and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(st["model_state"])
        if optimizer_G is not None and "optimizer_G" in st:
            optimizer_G.load_state_dict(st["optimizer_G"])
        if optimizer_D is not None and "optimizer_D" in st:
            optimizer_D.load_state_dict(st["optimizer_D"])
        print(f"Loaded checkpoint {ckpt_path} (iters={st.get('iters', 'unknown')})")
        return st.get("iters", 0)
    return 0

# ---------------------------
# Training entrypoint
# ---------------------------
def train(
    data_root,
    batch_size=8,
    input_height=256,
    input_width=256,
    lr_G=1e-4,
    lr_D=4e-4,
    betas=(0.5, 0.999),
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
    resume_checkpoint=None,
    print_every=100,
    save_every_iters=2000
):
    # Create dataset / dataloader
    ds = PairedImageMaskDataset(data_root, input_height=input_height, input_width=input_width, augment=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Instantiate model
    model = InpaintNN(input_height=input_height, input_width=input_width)
    model.to(device)

    # Separate parameters for generator and discriminator
    # Generator params: encoder, contextual_block, decoder
    gen_params = list(model.encoder.parameters()) + list(model.contextual_block.parameters()) + list(model.decoder.parameters())
    disc_params = list(model.discriminator_red.parameters())

    optimizer_G = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_G, betas=betas)
    optimizer_D = optim.Adam([p for p in disc_params if p.requires_grad], lr=lr_D, betas=betas)

    start_iter = load_checkpoint_if_exists(model, optimizer_G, optimizer_D, resume_checkpoint)

    iters = start_iter
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (X, Y, MASK) in pbar:
            X = X.to(device)        # (B,3,H,W)
            Y = Y.to(device)
            MASK = MASK.to(device)  # (B,3,H,W)
            iters += 1

            # ---------------------
            # 1) Update Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # call forward to compute D loss; model.forward returns (image_result, loss_D, loss_G, ssim)
            # We only want to update discriminator with loss_D
            with torch.set_grad_enabled(True):
                image_result, loss_D, loss_G_val, ssim_val = model.forward(X, Y, MASK, iters=iters)
                # loss_D is a tensor already meaned
                loss_D.backward()
                optimizer_D.step()

            # ---------------------
            # 2) Update Generator (encoder/contextual+decoder)
            # ---------------------
            optimizer_G.zero_grad()
            # We need to recompute forward because discriminator changed and image_result was detached before for D update
            with torch.set_grad_enabled(True):
                image_result, loss_D2, loss_G, ssim_val = model.forward(X, Y, MASK, iters=iters)
                # loss_G is the generator loss to descend
                loss_G.backward()
                optimizer_G.step()

            # Logging / progress bar
            if iters % print_every == 0:
                pbar.set_postfix({
                    "iter": iters,
                    "loss_D": f"{loss_D.item():.4f}",
                    "loss_G": f"{loss_G.item():.4f}",
                    "ssim": f"{ssim_val.item():.4f}"
                })

            # Save checkpoint
            if iters % save_every_iters == 0:
                save_checkpoint(model, optimizer_G, optimizer_D, iters, save_dir=checkpoint_dir)

        # Optionally save at epoch end
        save_checkpoint(model, optimizer_G, optimizer_D, iters, save_dir=checkpoint_dir)

    print("Training finished.")

# ---------------------------
# Small validation helper (single batch)
# ---------------------------
def validate_one_batch(model, val_loader, device="cuda"):
    model.eval()
    with torch.no_grad():
        X, Y, MASK = next(iter(val_loader))
        X = X.to(device); Y = Y.to(device); MASK = MASK.to(device)
        image_result, loss_D, loss_G, ssim_val = model.forward(X, Y, MASK, iters=0)
    model.train()
    return image_result, ssim_val

# ---------------------------
# If run as script
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root folder with subfolders 'images' and 'masks'")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lrG", type=float, default=1e-4)
    parser.add_argument("--lrD", type=float, default=4e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--save_every_iters", type=int, default=2000)
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        batch_size=args.batch_size,
        input_height=args.height,
        input_width=args.width,
        lr_G=args.lrG,
        lr_D=args.lrD,
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume,
        print_every=args.print_every,
        save_every_iters=args.save_every_iters
    )
