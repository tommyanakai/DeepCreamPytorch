# DeepCreamPyTorch
*Reviving Anime Artworks with PyTorch-Powered Deep Learning*


**DeepCreamPyTorch** is an unofficial PyTorch implementation of the original TensorFlow-based [DeepCreamPy](./README_old.md). This project enhances the original with a robust PyTorch backend, comprehensive training code, a user-friendly CLI, and critical bug fixes for improved performance and reliability.

DeepCreamPyTorch is a deep learning tool designed to seamlessly reconstruct parts of anime-style artworks. Simply mark the areas to restore with green using an image editor (e.g., GIMP, Photoshop), and let our neural network intelligently fill in the highlighted regions with plausible content.

<p align="center">
	<img src="https://github.com/Deepshift/DeepCreamPy/blob/master/readme_images/mermaid_collage.png" width="800" alt="Mermaid Reconstruction Example">
</p>

## ✨ Features
- **PyTorch Backend**: Rebuilt from the ground up with PyTorch for faster, more efficient processing.
- **Comprehensive Training Code**: Train your own models with included scripts and clear instructions.
- **CLI Interface**: Streamlined command-line interface for easy model execution and configuration.
- **Bug Fixes**: Addressed key issues from the original project for enhanced stability and output quality.

## 🚫 Limitations
DeepCreamPyTorch excels at reconstructing anime-style human-like figures with minor to moderate redactions. However, it may struggle with:
- Completely removed limbs or organs (e.g., arms, legs).
- Screentones (common in printed manga).
- Real-life photos, animated GIFs, or videos.


## 🚀 Getting Started
1. **Create Conda environment**:

```bash
conda env create -f environment_cpu.yml  # for cpu
conda env create -f environment_gpu.yml  # for gpu
```



2. **Run in GUI**:
```bash
python ./app.py
```
3. **Run in CLi**:
```bash
python main.py --decensor_input ./decensor_input --decensor_input_original ./decensor_input_original --decensor_output ./decensor_output --is_mosaic False
```
1. **Training (Not tested yet)**:
```bash
python train.py
```
