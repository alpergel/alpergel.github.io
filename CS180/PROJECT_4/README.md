<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">

<div style="max-width: 1100px; margin: 0 auto; padding: 12px 18px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; color: #0f172a; line-height: 1.6;">
<style>
/* Gallery appearance: rounded black frame; non-white inner background; images stay square */
figure { background: #f1f5f9; padding: 6px; border: 1px solid #000; border-radius: 10px; }
figure img { border-radius: 0 !important; border: none !important; box-shadow: none !important; display: block; }
/* Lightbox: subtle 1px border, no shadow */
#img-lightbox-img { border-radius: 0 !important; border: 1px solid #000 !important; box-shadow: none !important; }
</style>

<p align="center" style="margin: 0 0 10px;">
  <img src="https://img.shields.io/badge/CS180-Project%204-5B8DEF?style=for-the-badge" alt="CS180 Project 4 badge">
</p>

<h2 align="center" style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0.2rem 0 0.4rem; letter-spacing: 0.3px; background: linear-gradient(90deg, #5B8DEF, #A78BFA); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;">Alper Gel — Project 4</h2>

<hr style="border: none; border-top: 1px solid #e5e7eb; margin: 12px 0 16px;">


<h2 id="required-part-1">Part 0: Camera Calibration and 3D Scanning</h2>
<p style="margin: 0 0 10px; color: #334155;">
I utilized an ARUCO PDF generator tool to create an ARUCO tag of size 0.1 meters, and displayed that as close as possible to scale on my tablet. I captured around 30 images to calibrate with that and got the following calibration parameters:
<strong>Camera Matrix</strong>:
<table style="margin-bottom: 0.6em;">
  <tr>
    <td>5188.76</td>
    <td>0.00</td>
    <td>2068.00</td>
  </tr>
  <tr>
    <td>0.00</td>
    <td>5090.70</td>
    <td>1535.21</td>
  </tr>
  <tr>
    <td>0.00</td>
    <td>0.00</td>
    <td>1.00</td>
  </tr>
</table>

<strong>Distortion Coefficients</strong>:
<ul style="margin-top: 0;">
  <li>k₁ = 0.07885</li>
  <li>k₂ = 0.12868</li>
  <li>p₁ = -0.05880</li>
  <li>p₂ = -0.00659</li>
  <li>k₃ = -0.23006</li>
</ul>

Then, I placed a hat next to the same ARUCO tag setup, and captured around 35 images from all angles, attempting to maximize perspectives. Then, I utilized the calibration JSON generated earlier, and ran it through my PNP.py script in order to generate accurate poses for each image in the dataset. At the end of my PNP script, I utilized the provided viser boilerplate code in order to visualize the frustums. 
<p style="margin: 0 0 10px; color: #334155;">
<p style="margin: 32px 0;">
  <img src="assets/Part0/Screenshot 2025-11-14 200948.png" alt="Set 1 - Image 1" style="width: 40%; min-width: 120px; border-radius: 10px; border: 1.5px solid #e5e7eb; margin-right: 2%;">
  <img src="assets/Part0/Screenshot 2025-11-14 201026.png" alt="Set 1 - Image 2" style="width: 40%; min-width: 120px; border-radius: 10px; border: 1.5px solid #e5e7eb;">
  <br>
  <span style="font-size: 1.05rem; color: #64748b;">
    <b>Figure 1:</b> Viser Frustums of Custom Dataset From Afar and Upclose
  </span>
</p>


<h2>Part 1: Fit a Neural Field to a 2D Image </h2>
<p>For my 2D NeRF I had a sequential setup where I would first pass the input through my SinPosEnc module, which had a 2D input, and an output of input dim (2) + 2*(encoding frequency) * (input dim). That would pass to a linear layer of input dim equal to the SinPosEnc output, and layer width set by the hyperparameters. After that, there was a ReLU activation then 2 more linear layers with ReLU's in between. Finally there is a linear layer of input equivalent to layer width, and output 3, since we want to output RGB (3 channels), followed by a sigmoid activation. I used an LR of 1e-2 and an Adam optimizer. The default layer width is set to 256, and default PE frequency is set at 10.</p>

The following figure shows the training progression visualization for the provided test image for L = 10, Layer Width  = 256, Epochs = 5
<p style="margin: 32px 0; display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 16px;">
  <img src="assets/Part1/Fox_L_10_W_256/epoch_001.png" alt="Step 1" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Fox_L_10_W_256/epoch_002.png" alt="Step 2" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Fox_L_10_W_256/epoch_003.png" alt="Step 3" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Fox_L_10_W_256/epoch_004.png" alt="Step 4" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Fox_L_10_W_256/epoch_005.png" alt="Step 5" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">

</p>
<p style="text-align: center; color: #64748b; font-size: 1.08rem;">
  <b>Figure 2:</b> Comparison of model output at 5 training stages (left&nbsp;→&nbsp;right: earliest to latest).
</p>

<p style="margin: 32px 0; display: flex; flex-direction: column; align-items: center;">
  <img src="assets/Part1/Fox_L_10_W_256/fox_psnr_L_10_layer_256.png" alt="PSNR Curve" style="width: 60%; min-width: 280px; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <span style="color: #64748b; font-size: 1.08rem;">
    <b>Figure 3:</b> PSNR curve across each batch
  </span>
  <img src="assets/Part1/Fox_L_10_W_256/15677707699_d9d67acf9d_b-1.jpg" alt="Step 5" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb; margin-top: 24px;">
  <span style="color: #64748b; font-size: 1.08rem;">
    <b>Original Image Used for Training</b>
  </span>
</p>

The following figure shows the training progression visualization for my selected image for L = 10, Layer Width = 256, Epochs = 5
<p style="margin: 32px 0; display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 16px;">
  <img src="assets/Part1/Dog_L_10_W_256/epoch_001.png" alt="Step 1" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Dog_L_10_W_256/epoch_002.png" alt="Step 2" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Dog_L_10_W_256/epoch_003.png" alt="Step 3" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Dog_L_10_W_256/epoch_004.png" alt="Step 4" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <img src="assets/Part1/Dog_L_10_W_256/epoch_005.png" alt="Step 5" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb;">

</p>
<p style="text-align: center; color: #64748b; font-size: 1.08rem;">
  <b>Figure 2:</b> Comparison of model output at 5 training stages (left&nbsp;→&nbsp;right: earliest to latest).
</p>

<p style="margin: 32px 0; display: flex; flex-direction: column; align-items: center;">
  <img src="assets/Part1/Dog_L_10_W_256/psnr_dog.png" alt="PSNR Curve" style="width: 60%; min-width: 280px; border-radius: 12px; border: 1.5px solid #e5e7eb;">
  <span style="color: #64748b; font-size: 1.08rem;">
    <b>Figure 3:</b> PSNR curve across each batch
  </span>
  <img src="assets/Part1/Dog_L_10_W_256/dog2.jpg" alt="Step 5" style="width: 18%; border-radius: 12px; border: 1.5px solid #e5e7eb; margin-top: 24px;">
  <span style="color: #64748b; font-size: 1.08rem;">
    <b>Original Image Used for Training</b>
  </span>
</p>

For the Hyperparameter experimentation, I chose L values of 2 and 10, and Layer Width values of 256 and 512. The following figure shows the 2x2 grid of final result renderings (epoch 5) with the respective hyperparameters.
<p style="margin: 36px 0; display: flex; flex-direction: column; align-items: center;">
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px;">
    <div style="display: flex; flex-direction: column; align-items: center;">
      <img src="assets/Part1/Fox_L_2_W_256/epoch_005.png" alt="L=2, Width=256" style="width: 220px; border-radius: 10px; border: 1.5px solid #e5e7eb; margin-bottom: 6px;">
      <span style="color: #64748b; font-size: 1rem;">L = 2, Width = 256</span>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center;">
      <img src="assets/Part1/Fox_L_2_W_512/epoch_005.png" alt="L=2, Width=512" style="width: 220px; border-radius: 10px; border: 1.5px solid #e5e7eb; margin-bottom: 6px;">
      <span style="color: #64748b; font-size: 1rem;">L = 2, Width = 512</span>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center;">
      <img src="assets/Part1/Fox_L_10_W_256/epoch_005.png" alt="L=10, Width=256" style="width: 220px; border-radius: 10px; border: 1.5px solid #e5e7eb; margin-bottom: 6px;">
      <span style="color: #64748b; font-size: 1rem;">L = 10, Width = 256</span>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center;">
      <img src="assets/Part1/Fox_L_10_W_512/epoch_005.png" alt="L=10, Width=512" style="width: 220px; border-radius: 10px; border: 1.5px solid #e5e7eb; margin-bottom: 6px;">
      <span style="color: #64748b; font-size: 1rem;">L = 10, Width = 512</span>
    </div>
  </div>
  <span style="color: #64748b; font-size: 1.08rem; margin-top: 14px;">
    <b>Figure 4:</b> 2×2 grid of model result images for different combinations of PE frequency  (L) and layer width. Top row: L=2, bottom row: L=10. Left: Width=256, Right: Width=512.
  </span>
</p>

<h2>Part 2.1-2.5: Fit a Neural Radiance Field from Multi-view Images </h2>
<h2>Part 2.6: Training with Your Own Data</h2>

<h2>Extras Part 1: Optimizer Change</h2>

<h2>Extras Part 2: Monocular Depth Supervision</h2>
