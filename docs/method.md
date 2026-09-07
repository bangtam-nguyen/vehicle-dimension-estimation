# Method

Notes on the geometry behind the pipeline. For the full treatment see the thesis.

## 1. Footprint extraction

Instance segmentation returns a mask $M$ for each vehicle. Not all of it is useful for measurement: points on the roof, windshield, or upper body sit well above the road plane, and a planar homography maps only coplanar points correctly. Including them biases the result.

The footprint is built from the lower region of the mask:

1. Trim a fraction of the mask width from each side (`--trim_x_ratio`) to drop unreliable left and right extremes.
2. Keep a band of height `band_ratio × mask_height` measured from the bottom (`--band_ratio`, default 0.20).
3. For each image column, take the lowest point still inside the mask.
4. Smooth the resulting contour with a moving average (`--contour_smooth_k`).

The output is a point set $P = \{(x_i, y_i)\}_{i=1}^{N}$ approximating the vehicle's contact region with the road.

## 2. BEV homography

Assuming the observed road surface is approximately planar, the mapping from image plane to top-down plane is a homography $H \in \mathbb{R}^{3\times3}$. For a point in homogeneous coordinates:

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

$H$ is estimated from four point correspondences between the source image and the target BEV rectangle. Expanded:

$$u = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}, \qquad v = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}$$

The shared denominator is what encodes the perspective effect.

**Pixel-to-metre scale.** Two options. Declare it directly with `--scale_m_per_px`, or derive it from a pair of ground points whose real separation is known (`--scale_from_src_pair` with `--scale_pair_m`):

$$s = \frac{d_{\text{real}}}{d_{\text{BEV}}}$$

Scale error is linear in the final measurement, so this is the single most important parameter to get right.

## 3. PCA on the footprint

After warping, the footprint is a 2D point cloud $Q = \{(u_i, v_i)\}_{i=1}^{N}$ in the road plane. Centroid:

$$\mu = \frac{1}{N}\sum_{i=1}^{N} q_i$$

Covariance:

$$\Sigma = \frac{1}{N}\sum_{i=1}^{N}(q_i - \mu)(q_i - \mu)^{T}$$

Solving $\Sigma e_k = \lambda_k e_k$ with $\lambda_1 \geq \lambda_2$ gives $e_1$ as the dominant elongation direction of the footprint and $e_2$ as its orthogonal complement. Projecting each point:

$$\alpha_i = e_1^{T}(q_i - \mu), \qquad \beta_i = e_2^{T}(q_i - \mu)$$

Rather than taking raw extrema, which any single outlier point would dominate, the spans are taken between robust percentiles (`--trim_ex_low/high`, `--trim_ey_low/high`):

$$L_{px} = \text{span}(\alpha), \qquad W_{px} = \text{span}(\beta)$$

Converting to metres: $L = sL_{px}$, $W = sW_{px}$.

**Why this matters.** Measuring along $e_1$ and $e_2$ instead of the image axes makes the result independent of the vehicle's heading in the frame. This is the core difference from a bounding-box approach.

**When it degrades.** PCA describes the statistical distribution of the points, not the vehicle's true geometry. If the footprint is heavily occluded, asymmetrically distorted, or nearly isotropic, $e_1$ drifts away from the real longitudinal axis. The `pca_min_eig_ratio` parameter detects the isotropic case and falls back to the road axes; `pca_hybrid` mode blends both estimates.

## 4. Robust aggregation per track ID

A vehicle observed across $n_k$ frames yields $n_k$ measurements of the same physical quantity. Rather than selecting one frame, all valid observations are aggregated.

Frames are rejected when the minimum-area rectangle of the footprint is too skewed relative to the road axis (`--min_align`, default 0.92) — a strong signal that segmentation or the footprint is unreliable in that frame.

Surviving measurements are smoothed with a running median (`--smooth_window`), then reduced with the median, with MAD and IQR reported alongside as dispersion measures. The median is preferred over the mean for its resistance to the occasional badly broken mask. Tracks with fewer than `--summary_min_frames` valid frames are excluded from the final table.

The output for each vehicle is a single representative pair $(\hat{L}_k, \hat{W}_k)$ rather than a time series.
