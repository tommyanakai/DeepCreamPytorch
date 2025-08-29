import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch

try:
    import file
    from model_new import InpaintNN
    from libs.utils import *
except ImportError as e:
    print(f"Error importing local libraries: {e}")
    sys.exit(1)

class Decensor:
    def __init__(self):
        args = self.get_args()
        self.is_mosaic = args.is_mosaic
        self.variations = args.variations
        self.mask_color = [0.0, 1.0, 0.0]  # Green (0, 255, 0) normalized
        self.decensor_input_path = args.decensor_input
        self.decensor_input_original_path = args.decensor_input_original
        self.decensor_output_path = args.decensor_output
        self.clean_up_input_dirs = args.clean_up_input_dirs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.warm_up = False
        os.makedirs(self.decensor_output_path, exist_ok=True)
        self.files_removed = []

    def copy_to(self, img:np.ndarray, img_name:str , output_dir:str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img_pil = Image.fromarray(img.astype('uint8'))
      
        save_path = os.path.join(output_dir, img_name)
        img_pil.save(save_path)
        self.insert_progress(f"Image copied to {save_path}!\n")
   

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="Decensor images")
        parser.add_argument('--decensor_input', type=str, default="./decensor_input", help='Input directory for colored images')
        parser.add_argument('--decensor_input_original', type=str, default="./decensor_input_original", help='Input directory for original images (mosaic mode)')
        parser.add_argument('--decensor_output', type=str, default="./decensor_output", help='Output directory for decensored images')
        parser.add_argument('--is_mosaic', type=bool ,default=False, help='Enable mosaic decensoring')
        parser.add_argument('--variations', type=int, default=1, help='Number of variations per image')
        parser.add_argument('--clean_up_input_dirs',type=bool, default=False, help='Clean input directories after processing')
        return parser.parse_args()

    def insert_progress(self, message):
        print(message)

    def load_model(self):
        self.insert_progress("Loading model ... please wait ...\n")
        if self.model is None:
            self.model = InpaintNN().to(self.device)
            checkpoint_path = "./models/mosaic.pth" if self.is_mosaic else "./models/bar.pth"
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint)
        self.warm_up = True
        self.insert_progress("Loading model finished!\n")

    def find_mask(self, colored):
        mask = np.ones(colored.shape, np.uint8)
        i, j = np.where(np.all(colored[0] == self.mask_color, axis=-1))
        mask[0, i, j] = 0
        return mask

    def apply_variant(self, image, variant_number):
        if variant_number == 0:
            return image
        elif variant_number == 1:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif variant_number == 2:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    def decensor_image_variations(self, ori, colored, file_name):
        for i in range(self.variations):
            self.decensor_image_variation(ori, colored, i, file_name)

    def decensor_image_variation(self, ori, colored, variant_number, file_name):
        ori = self.apply_variant(ori, variant_number)
        colored = self.apply_variant(colored, variant_number)
        width, height = ori.size
        has_alpha = False
        if ori.mode == "RGBA":
            has_alpha = True
            alpha_channel = np.asarray(ori)[:, :, 3]
            alpha_channel = np.expand_dims(alpha_channel, axis=-1)
            ori = ori.convert('RGB')

        ori_array = image_to_array(ori)
        ori_array = np.expand_dims(ori_array, axis=0)

        if self.is_mosaic:
            colored = colored.convert('RGB')
            color_array = image_to_array(colored)
            color_array = np.expand_dims(color_array, axis=0)
            mask = self.find_mask(color_array)
            mask_reshaped = mask[0, :, :, :] * 255.0
            mask_img = Image.fromarray(mask_reshaped.astype('uint8'))
        else:
            mask = self.find_mask(ori_array)

        regions = find_regions(colored.convert('RGB'), [0, 255, 0])
        self.insert_progress(f"Found {len(regions)} censored regions in this image!\n")

        if len(regions) == 0 and not self.is_mosaic:
            self.insert_progress("No green (0,255,0) regions detected!\n")
            self.copy_to(np.asarray(ori), file_name, self.decensor_output_path)
            return

        output_img_array = ori_array[0].copy()

        for region_counter, region in enumerate(regions, 1):
            self.insert_progress(f"Decensoring censor {region_counter}/{len(regions)}\n")
            bounding_box = expand_bounding(ori, region, expand_factor=1.5)
            crop_img = ori.crop(bounding_box)
            crop_img = crop_img.resize((256, 256))
            crop_img_array = image_to_array(crop_img)
            mask_img = Image.fromarray((mask[0, :, :, :] * 255.0).astype('uint8'))
            mask_img = mask_img.crop(bounding_box).resize((256, 256))
            mask_array = image_to_array(mask_img)

            if not self.is_mosaic:
                a, b = np.where(np.all(mask_array == 0, axis=-1))
                crop_img_array[a, b, :] = 0.

            crop_img_array = np.expand_dims(crop_img_array, axis=0)
            mask_array = np.expand_dims(mask_array, axis=0)
            crop_img_array = crop_img_array * 2.0 - 1
            crop_img_array = torch.tensor(crop_img_array).permute(0, 3, 1, 2).float().to(self.device)
            mask_array = torch.tensor(mask_array).permute(0, 3, 1, 2).float().to(self.device)
            pred_img_array, _, _, _ = self.model(crop_img_array, crop_img_array, mask_array)
            pred_img_array = pred_img_array.permute(0, 2, 3, 1).detach().cpu().numpy()
            pred_img_array = np.squeeze(pred_img_array, axis=0)
            pred_img_array = (255.0 * ((pred_img_array + 1.0) / 2.0)).astype(np.uint8)

            bounding_width = bounding_box[2] - bounding_box[0]
            bounding_height = bounding_box[3] - bounding_box[1]
            pred_img = Image.fromarray(pred_img_array.astype('uint8'))
            pred_img = pred_img.resize((bounding_width, bounding_height), resample=Image.BICUBIC)
            pred_img_array = image_to_array(pred_img)
            pred_img_array = np.expand_dims(pred_img_array, axis=0)

            for i in range(len(ori_array)):
                for col in range(bounding_width):
                    for row in range(bounding_height):
                        bounding_width_index = col + bounding_box[0]
                        bounding_height_index = row + bounding_box[1]
                        if (bounding_width_index, bounding_height_index) in region:
                            output_img_array[bounding_height_index][bounding_width_index] = pred_img_array[i, :, :, :][row][col]
            self.insert_progress(f"{region_counter} out of {len(regions)} regions decensored.\n")

        output_img_array = output_img_array * 255.0
        if has_alpha:
            output_img_array = np.concatenate((output_img_array, alpha_channel), axis=2)

        output_img = Image.fromarray(output_img_array.astype('uint8'))
        output_img = self.apply_variant(output_img, variant_number)
        base_name, ext = os.path.splitext(file_name)
        file_name = f"{base_name} {variant_number}{ext}"
        save_path = os.path.join(self.decensor_output_path, file_name)
        output_img.save(save_path)
        self.insert_progress(f"Decensored image saved to {save_path}!\n")

    def decensor_all_images_in_folder(self):
        if not self.warm_up:
            self.load_model()
        input_color_dir = self.decensor_input_path
        file_names = os.listdir(input_color_dir)
        valid_formats = {".png", ".jpg", ".jpeg"}
        file_names, self.files_removed = file.check_file(input_color_dir, self.decensor_output_path, False)

        if self.is_mosaic:
            ori_dir = self.decensor_input_original_path
            test_file_names = os.listdir(ori_dir)
            missing_files = []
            for file_name in file_names:
                color_basename, color_ext = os.path.splitext(file_name)
                if color_ext.casefold() in valid_formats:
                    found = False
                    for test_file_name in test_file_names:
                        test_basename, test_ext = os.path.splitext(test_file_name)
                        if test_basename == color_basename and test_ext.casefold() in valid_formats:
                            found = True
                            break
                    if not found:
                        missing_files.append(file_name)
            if missing_files:
                self.insert_progress(f"Error: Missing original images for: {', '.join(missing_files)}\n")
                return

        self.insert_progress(f"Decensoring {len(file_names)} image files\n")
        for n, file_name in enumerate(file_names, 1):
            color_file_path = os.path.join(input_color_dir, file_name)
            color_basename, color_ext = os.path.splitext(file_name)
            if os.path.isfile(color_file_path) and color_ext.casefold() in valid_formats:
                self.insert_progress(f"Decensoring image file: {file_name}\n")
                try:
                    colored_img = Image.open(color_file_path)
                except Exception as e:
                    self.insert_progress(f"Cannot identify image file ({color_file_path}): {e}\n")
                    self.files_removed.append((color_file_path, 3))
                    continue
                if self.is_mosaic:
                    ori_dir = self.decensor_input_original_path
                    test_file_names = os.listdir(ori_dir)
                    for test_file_name in test_file_names:
                        test_basename, test_ext = os.path.splitext(test_file_name)
                        if test_basename == color_basename and test_ext.casefold() in valid_formats:
                            ori_file_path = os.path.join(ori_dir, test_file_name)
                            ori_img = Image.open(ori_file_path)
                            self.decensor_image_variations(ori_img, colored_img, file_name)
                            break
                    else:
                        self.insert_progress(f"Original image not found for {color_file_path}\n")
                else:
                    self.decensor_image_variations(colored_img, colored_img, file_name)
            else:
                self.insert_progress(f"Image can't be found: {color_file_path}\n")
        self.insert_progress("Decensoring complete!\n")
        if self.files_removed:
            file.error_messages(None, self.files_removed)
        self.model = None
        torch.cuda.empty_cache()

    def clean_input_directories(self):
        allowed_extensions = {".png", ".jpg", ".jpeg"}
        self.insert_progress(f"Cleaning {self.decensor_input_path}...\n")
        for file_name in os.listdir(self.decensor_input_path):
            file_path = os.path.join(self.decensor_input_path, file_name)
            if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in allowed_extensions:
                os.remove(file_path)
        if self.is_mosaic:
            self.insert_progress(f"Cleaning {self.decensor_input_original_path}...\n")
            for file_name in os.listdir(self.decensor_input_original_path):
                file_path = os.path.join(self.decensor_input_original_path, file_name)
                if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in allowed_extensions:
                    os.remove(file_path)
        self.insert_progress("Done!\n")

    def do_post_jobs(self):
        if self.clean_up_input_dirs:
            self.clean_input_directories()

if __name__ == '__main__':
    decensor = Decensor()
    decensor.decensor_all_images_in_folder()
    decensor.do_post_jobs()