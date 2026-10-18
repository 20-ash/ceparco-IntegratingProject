# Multi-Directional 5×5 Sobel Edge Detection

**Group 4**
- Copon, Matthew Joshua
- Lee, Ashley Fiona
- Ong, Eiress Bassey
- Xu, Kai Wen

## Description of the Project
Edge detection is a foundational step in remote sensing, medical image processing, and industrial detection. The Sobel operator is the most widely adapted method to find the approximate absolute gradient magnitude at each pixel due to its simplicity and efficiency. However, standard implementations rely on a 3x3 kernel considering only horizontal and vertical directions. 

Chang et al. (2023) expanded the Sobel operator to 5x5 size with four directional filters (0◦, 90◦, 45◦, 135◦) considering two additional diagonal directions. Results showed that compared to 3x3, the 5x5 operator is more insensitive to changes and less affected by noise since it takes more pixels or greater area into consideration. This helps provide clearer edge features which makes edge detection clearer and cleaner. Moreover, with the implementation of the four directional filter instead of two directional, more abundant textures were shown during edge detection. Their implementation, however, was optimized for NVIDIA GPUs using CUDA, which requires dedicated hardware and incurring data transfer overhead that limits deployment on standard systems.

This project addresses the gap by optimizing the four directional 5x5 Sobel operator for CPUs using **AVX2 SIMD** instructions. Through applying data-level parallelism, the group aims to achieve significant speedup compared to C implementation without requiring specialized hardware. Furthermore, the group will compare and evaluate the 5×5 versus 3×3 kernels, as well as two‑directional versus four‑directional implementations.

## Proposal Video
[Watch the project proposal video](https://drive.google.com/file/d/1-72TuUAJVy37nJjmVDsZhd3dUHP6azdz/view?usp=sharing)

## Proposed Algorithm
### 5x5
The filters of the four-directional 5x5 Sobel operator are defined as follows (Chang et al., 2023):

Horizontal Gradient (0◦)

$$
G_x =
\begin{bmatrix}
-1 & -2 & 0 & 2 & 1 \\
-4 & -8 & 0 & 8 & 4 \\
-6 & -12 & 0 & 12 & 6 \\
-4 & -8 & 0 & 8 & 4 \\
-1 & -2 & 0 & 2 & 1
\end{bmatrix} * I
$$

Vertical Gradient (90◦)

$$
G_y =
\begin{bmatrix}
-1 & -4 & -6 & -4 & -1 \\
-2 & -8 & -12 & -8 & -2 \\
0 & 0 & 0 & 0 & 0 \\
2 & 8 & 12 & 8 & 2 \\
1 & 4 & 6 & 4 & 1
\end{bmatrix} * I
$$

Diagonal Gradient (45◦)

$$
G_d =
\begin{bmatrix}
-6 & -4 & -1 & -2 & 0 \\
-4 & -12 & -8 & 0 & 2 \\
-1 & -8 & 0 & 8 & 1 \\
-2 & 0 & 8 & 12 & 4 \\
0 & 2 & 1 & 4 & 6
\end{bmatrix} * I
$$

Diagonal Gradient (135◦)

$$
G_{dt} =
\begin{bmatrix}
0 & -2 & -1 & -4 & -6 \\
2 & 0 & -8 & -12 & -4 \\
1 & 8 & 0 & -8 & -1 \\
4 & 12 & 8 & 0 & -2 \\
6 & 4 & 1 & 2 & 0
\end{bmatrix} * I
$$

Final results are aggregated as follows:

$$
G = \sqrt{G_x^2 + G_y^2 + G_d^2 + G_{dt}^2}
$$

#### Separate Convolution for G_x and G_y
G_x and G_y can be expressed as the outer product of two 1-D vectors:

$$
G_x =
\begin{bmatrix}
1 \\
4 \\
6 \\
4 \\
1
\end{bmatrix}
\times
\begin{bmatrix}
-1 & -2 & 0 & 2 & 1
\end{bmatrix}
$$

$$
G_y =
\begin{bmatrix}
-1 \\
-2 \\
0 \\
2 \\
1
\end{bmatrix}
\times
\begin{bmatrix}
1 & 4 & 6 & 4 & 1
\end{bmatrix}
$$

This means that each gradient can be computed first by applying the row vector horizontally across every row, then applying the column vector vertically to the intermediate results. The separable form produces uniform and repeating weight patterns which exposes the data‑level parallelism needed for SIMD execution. AVX2 can apply these identical weights to 8 or more output positions at once using 256‑bit YMM registers.

#### Sum-Difference Transformation for Diagonals
Gd and Gdt are not separable, so we apply the sum‑difference method from Chang et al. (2023):

$$
\begin{aligned}
S &= G_d + G_{dt} \\
D &= G_d - G_{dt} \\
G_d &= \frac{S + D}{2} \\
G_{dt} &= \frac{S - D}{2}
\end{aligned}
$$

Instead of computing two full 5x5 convolutions separately, the group will combine the kernels first, apply the combined forms (S and D), and then recover the original responses.

$$
S =
\begin{bmatrix}
-6 & -6 & -2 & -6 & -6 \\
-2 & -16 & -16 & -12 & -2 \\
0 & 0 & 0 & 0 & 0 \\
2 & 16 & 16 & 12 & 2 \\
6 & 6 & 2 & 6 & 6
\end{bmatrix}
$$

The middle row contains only zeros, so its contribution can be skipped entirely. Moreover, row 0 = -row 4 and row 1 = -row 3. This indicates that the group can just compute once and reuse with sign flip.

$$
D =
\begin{bmatrix}
-6 & -2 & 0 & 2 & 6 \\
-6 & -12 & 0 & 12 & 6 \\
-2 & -16 & 0 & 16 & 2 \\
-6 & -12 & 0 & 12 & 6 \\
-6 & -2 & 0 & 2 & 6
\end{bmatrix}
$$

The middle column contains only zeros, so its contribution can be skipped entirely. Moreover, col 0 = -col 4 and col 1 = -col 3. This indicates that the group can just compute once and reuse with sign flip.

### 3x3 for C implementation
The standard 3×3 two‑directional Sobel operator serves as our baseline for correctness and performance comparison. It uses only horizontal and vertical gradients:

Horizontal Gradient (0◦)

$$
G_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1 \\
\end{bmatrix} * I
$$

Vertical Gradient (90◦)

$$
G_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1 \\
\end{bmatrix} * I
$$

Final results are aggregated as follows:

$$
G = \sqrt{G_x^2 + G_y^2}
$$

## References
- Chang, Q., Li, X., Li, Y., & Miyazaki, J. (2023). Multi-directional Sobel operator kernel on GPUs. *Journal of Parallel and Distributed Computing, 177*, 160–170.
- Ben Fredj, H., Ltaif, M., Ammar, A., & Souani, C. (2017). Parallel implementation of Sobel filter using CUDA. *Proceedings of ICCAD '17*, 209–212.
- Shi, L., & Zhao, Y. (2023). Edge detection of high-resolution remote sensing image based on multi-directional improved Sobel operator. *IEEE Access, 11*, 135979–135993.

## Implementation of the Proposed Project
| Phase | Deliverables |
|---|---|
| **1. Basic Version** | Implement the algorithm in standard C and verify its correctness against OpenCV. |
| **2. AVX2 Optimization** | Use AVX2 intrinsics to improve performance across all four gradient calculations. |
| **3. Testing and Comparison** | Compare 5×5 4-Direction Sobel in standard C and SIMD. Measure execution time and speedup against the basic version using images of sizes 512×512, 1024×1024, and 2048×2048. |
| **4. Final Output** | Provide the working source code and a report of the performance results. |

### Implementation Versions

| Version |
|---|
| **1. 3×3 2-Direction Sobel (C)** |
| **2. 5×5 2-Direction Sobel (C)** | 
| **3. 5×5 4-Direction Sobel (C)** |
| **4. 5×5 4-Direction Sobel (AVX2/SIMD)** |

