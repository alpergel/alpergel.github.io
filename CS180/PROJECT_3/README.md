<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">

<div style="max-width: 1100px; margin: 0 auto; padding: 12px 18px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; color: #0f172a; line-height: 1.6;">
<style>
/* Gallery appearance: rounded black frame; non-white inner background; images stay square */
figure { background: #f1f5f9; padding: 6px; border: 1px solid #000; border-radius: 10px; }
figure img { border-radius: 0 !important; border: none !important; box-shadow: none !important; display: block; }
/* Lightbox: subtle 1px border, no shadow */
#img-lightbox-img { border-radius: 0 !important; border: 1px solid #000 !important; box-shadow: none !important; }
</style>


<h2 align="center" style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0.2rem 0 0.4rem; letter-spacing: 0.3px; background: linear-gradient(90deg, #5B8DEF, #A78BFA); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;">Alper Gel — Project 3</h2>


<h2 id="required-part-1">Section A</h2>
<h3>Part A.1: Shoot the Pictures</h3>
<p style="margin: 0 0 10px; color: #334155;">
The following images were captured on an iPhone 16 by starting on the center image, then while keeping my body and arms in place, slightly rotating the phone itself left and right from the center pose. Further, a leveling indicator was utilized to make sure each image had approx the sane COP.
<p style="margin: 0 0 10px; color: #334155;">
<p align="center" style="margin: 32px 0;">
  <img src="assets/A_1/CITRIS_Room/IMG_1534.jpg" alt="Set 1 - Image 1" style="width: 40%; min-width: 120px; border-radius: 10px; border: 1.5px solid #e5e7eb; margin-right: 2%;">
  <img src="assets/A_1/CITRIS_Room/IMG_1535.jpg" alt="Set 1 - Image 2" style="width: 40%; min-width: 120px; border-radius: 10px; border: 1.5px solid #e5e7eb;">
  <br>
  <span style="font-size: 1.05rem; color: #64748b;">
    <b>Set 1:</b> Two images from the first set.
  </span>
</p>

<p align="center" style="margin: 32px 0;">
  <img src="assets/A_1/Meeting_Room/IMG_1523.jpg" alt="Set 2 - Image 1" style="width: 40%; min-width: 120px; border-radius: 10px; border: 1.5px solid #e5e7eb; margin-right: 2%;">
  <img src="assets/A_1/Meeting_Room/IMG_1524.jpg" alt="Set 2 - Image 2" style="width: 40%; min-width: 120px; border-radius: 10px; border: 1.5px solid #e5e7eb;">
  <br>
  <span style="font-size: 1.05rem; color: #64748b;">
    <b>Set 2:</b> Two images from the second set.
  </span>
</p>

<h3>Part A.2: Recover Homographies</h3>

<p style="margin: 0 0 10px; color: #334155;">
The homography recovery process involves solving a system of linear equations. For each correspondence point (x, y) → (u, v), we set up the following system of equations:
</p>

<div style="background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6; margin: 20px 0; font-family: 'Courier New', monospace; font-size: 0.9rem;">

<h4 style="margin-top: 0; color: #1e40af;">System of Equations for Homography Recovery</h4>

<p>For n correspondence points, we have 2n equations plus 1 constraint equation:</p>

<p><strong>For each correspondence (xᵢ, yᵢ) → (uᵢ, vᵢ):</strong></p>

<p>Row 2i (u-equation):</p>
<p style="margin: 5px 0; padding-left: 20px;">
xᵢh₁ + yᵢh₂ + h₃ - uᵢxᵢh₇ - uᵢyᵢh₈ - uᵢh₉ = 0
</p>

<p>Row 2i+1 (v-equation):</p>
<p style="margin: 5px 0; padding-left: 20px;">
xᵢh₄ + yᵢh₅ + h₆ - vᵢxᵢh₇ - vᵢyᵢh₈ - vᵢh₉ = 0
</p>

<p><strong>Scale constraint:</strong></p>
<p style="margin: 5px 0; padding-left: 20px;">
h₉ = 1
</p>

<p><strong>Matrix form:</strong></p>
<p style="margin: 5px 0; padding-left: 20px;">
A · h = b
</p>

<p>Where:</p>
<ul style="margin: 10px 0; padding-left: 20px;">
<li><strong>A</strong> is a (2n+1) × 9 matrix</li>
<li><strong>h</strong> = [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉]ᵀ is the homography matrix flattened</li>
<li><strong>b</strong> = [0, 0, ..., 0, 1]ᵀ is the constraint vector</li>
</ul>

<p><strong>Structure of matrix A:</strong></p>
<p style="margin: 5px 0; padding-left: 20px;">
For each point i, rows 2i and 2i+1 are:</p>

<p style="margin: 5px 0; padding-left: 40px;">
Row 2i: [xᵢ, yᵢ, 1, 0, 0, 0, -uᵢxᵢ, -uᵢyᵢ, -uᵢ]</p>
<p style="margin: 5px 0; padding-left: 40px;">
Row 2i+1: [0, 0, 0, xᵢ, yᵢ, 1, -vᵢxᵢ, -vᵢyᵢ, -vᵢ]</p>

<p style="margin: 5px 0; padding-left: 20px;">
Final row (constraint): [0, 0, 0, 0, 0, 0, 0, 0, 1]</p>

</div>
In matrix notation, the homography vector <b>h</b> can be solved directly as <b>h = A<sup>-1</sup> · b</b> (when A is invertible). However, since we usually want MORE than 4 points, we set this problem up as a least squares problem using the Numpy lstsq() function. However, we can get an even more optimized solution by taking advantage of some linear algebra knowledge with the SVD decomposition. The rightmost singular vector provides us with H, which we then reshape into a 3 x 3 matrix. 

<br>
As I was seeing some numerical stability issues, I researched some ways to improve consistency and homography stability, and stumbled upon the concept of Hartley normalization, which allows you to improve the condition number of the matrix (high condition number high instability and likely no proper solution). With Hartley normalization, the identified correspondance points are recentered and rescaled so that the overal linear system is balanced. Math below:
<div style="background: #18181b; color: #f1f5f9; border-radius: 12px; padding: 24px; margin: 24px 0; font-size: 1.08rem;">
  <b>Hartley Normalization (Point Normalization for Homography)</b>
  <br><br>
  <p>
    Given a set of 2D points <span style="font-family: 'Latin Modern Math', serif;">{(x<sub>i</sub>, y<sub>i</sub>)}<sup>N</sup><sub>i=1</sub></span>, define the centroid:
  </p>
  <p style="margin-left: 24px;">
    <span style="font-size: 1.1em;">
      <span style="font-family: 'Latin Modern Math', serif;">
        \(\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i,\quad \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i\)
      </span>
    </span>
  </p>
  <p>
    Center the points:
  </p>
  <p style="margin-left: 24px;">
    <span style="font-family: 'Latin Modern Math', serif;">
      \(x'_i = x_i - \bar{x},\quad y'_i = y_i - \bar{y}\)
    </span>
  </p>
  <p>
    Compute the mean Euclidean distance of the centered points to the origin:
  </p>
  <p style="margin-left: 24px;">
    <span style="font-family: 'Latin Modern Math', serif;">
      \(\bar{d} = \frac{1}{N} \sum_{i=1}^N \sqrt{(x'_i)^2 + (y'_i)^2}\)
    </span>
  </p>
  <p>
    Choose the isotropic scale so that the <b>average distance becomes</b> \(\sqrt{2}\):
  </p>
  <p style="margin-left: 24px;">
    <span style="font-family: 'Latin Modern Math', serif;">
      \(s = \frac{\sqrt{2}}{\bar{d}}\)
    </span>
  </p>
  <p>
    The <b>normalization transform</b> (apply to homogeneous points \([x\ y\ 1]^T\)) is:
  </p>
  <p style="margin-left: 24px;">
    <span style="font-family: 'Latin Modern Math', serif;">
      \(T = \begin{bmatrix}
        s & 0 & -s\bar{x} \\
        0 & s & -s\bar{y} \\
        0 & 0 & 1
      \end{bmatrix}\)
    </span>
  </p>
</div>



<p align="center" style="margin: 32px 0;">
  <img src="assets/A_2/CITRIS_Room/correspondance.png" alt="Set 2 - Image 1" style="width: 80%; min-width: 400px; border-radius: 16px; border: 2.5px solid #e5e7eb; margin-right: 2%;">
  <br>
  <span style="font-size: 1.05rem; color: #64748b;">
    <b>Set 1:</b> Visualization of Correspondance Points.
  </span>
</p>
<p><strong>Set 1 Homography Matrix:</strong></p>
<pre><code>H = [[-1.50030413e-01 -4.10444775e-01  2.20066913e+03]
 [-3.61015852e-01 -4.24241792e-01  3.03414021e+03]
 [-1.03412057e-04 -1.59095473e-04  1.00000000e+00]]
</code></pre>
<p align="center" style="margin: 32px 0;">
  <img src="assets/A_2/Meeting_Room/correspondance.png" alt="Set 2 - Image 1" style="width: 80%; min-width: 400px; border-radius: 16px; border: 2.5px solid #e5e7eb; margin-right: 2%;">
  <br>
  <span style="font-size: 1.05rem; color: #64748b;">
    <b>Set 2:</b> Visualization of Correspondance Points.
  </span>
</p>
<p><strong>Set 2 Homography Matrix:</strong></p>
<pre><code>H = [[ 2.35072803e+00  4.96191253e-02 -5.44933219e+03]
 [ 8.53326606e-01  1.87472103e+00 -2.24930721e+03]
 [ 3.33232293e-04 -4.26128816e-05  1.00000000e+00]]
</code></pre>

<h3>Part A.3: Warp The Images</h3>

<p style="margin: 0 0 10px; color: #334155;">
Using the two image warping techniques we made (Nearest Neighbor and Bilinear Interpolation) we can now rectify rectangular objects via a combination of the following pipeline:

<ol style="margin: 0 0 10px 0; padding-left: 24px;">
  <li><strong>Input Image</strong></li>
  <li>Identify keypoints of the rectangular object</li>
  <li>Calculate the target rectangle size</li>
  <li>Compute the Homography Matrix</li>
  <li>Apply the inverse Homography Matrix and chosen warping method (Nearest Neighbor or Bilinear Interpolation) to output the rectified rectangular object (see below)</li>
</ol>
</p>
<p align="center" style="margin: 32px 0;">
  <img src="assets/A_3/CITRIS_Sign/Screenshot 2025-10-08 200557.png" alt="Rectified Output (Nearest Neighbor)" style="width: 60%; min-width: 420px; border-radius: 16px; border: 3px solid #e5e7eb; margin-right: 3%;">
  <br>
  <span style="font-size: 1.05rem; color: #64748b;">
    <b>Left:</b> Correspondance points bounding box. <b>Center:</b>Rectified output using Bilinear Interpolation. <b>Right:</b> Rectified output using Nearest Neighbor Warping.
  </span>
  <img src="assets/A_3/Poster/Screenshot 2025-10-08 200505.png" alt="Rectified Output (Bilinear Interpolation)" style="width: 60%; min-width: 420px; border-radius: 16px; border: 3px solid #e5e7eb;">
  <br>
  <span style="font-size: 1.05rem; color: #64748b;">
    <b>Left:</b> Correspondance points bounding box. <b>Center:</b>Rectified output using Bilinear Interpolation. <b>Right:</b> Rectified output using Nearest Neighbor Warping.
  </span>
</p>



<p style="margin: 0 0 10px; color: #334155;">
As is clear in the comparison images above, we can see that both images provide a successfully rectified rectangular object output, however, the bilinear interpolation offers higher visual quality. It is worthwhile to note that Nearest Neighbor approach is considerably faster than the Bilinear Interpolation approach. 

</p>

<h3>Part A.4: Blend the Images Into a Mosaic</h3>
For the panorama creation, I created a CLI that takes in a folder, the number of correspondances you want (default 4). The process then sorts the image files in the folder, selects the center index and starts to map all other images to that center image. It iterates through all the images that are not the center image, makes you provide a point correspondance between the image and the center image. Then the H matrix is calculated, which allows for the warp to be calculated. The warped image, validity mask, and origin of the new image are then appended to a list. Once all images in the folder have been warped properly, a global canvas for the pano is calculated, then the warped images are placed onto the canvas at their respective origin points. Finally we utilize feather blending to soften the edges between the placed images. I did put a laplacian stack blending implementation, but commented it out as it simply takes too long with more than 2+ images in a folder. For the same reason, I opted to use nearest neighbor warping for RAM and time efficiency during the panorama process. 

