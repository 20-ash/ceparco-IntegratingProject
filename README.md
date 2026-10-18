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

## Presentation pitch of the proposed project
[Watch the project proposal video](https://drive.google.com/file/d/1-72TuUAJVy37nJjmVDsZhd3dUHP6azdz/view?usp=sharing)

[PPT](https://canva.link/cnkb406mfp4stuf)

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

Chang et al. [1] combine the directions with the square root of the sum of squares. This project uses the sum of absolute values, as in the 3×3 formula above, so every step stays in integer arithmetic and the scalar, AVX2 and CUDA outputs can be compared exactly. A four-direction 5×5 operator needs about eight times the arithmetic of the two-direction 3×3 operator [1]. Shi and Zhao [3] go further to 16 directions; this project stops at four. Canny edge detection, which uses Sobel as its gradient step, is outside the scope of this project.

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

## Implementation of the Proposed Project
| Phase | Deliverables |
|---|---|
| **1. Basic Version** | Implement the algorithm in standard C and verify its correctness against OpenCV. |
| **2. AVX2 Optimization** | Use AVX2 intrinsics to improve performance across all four gradient calculations. |
| **3. Testing and Comparison** | Compare 5×5 4-Direction Sobel in standard C and SIMD. Measure execution time and speedup against the basic version using images of sizes 512×512, 1024×1024, and 2048×2048. |
| **4. Final Output** | Provide the working source code and a report of the performance results. |

### Inputs
 
- A PNG image, read with stb_image [9] and converted to 8-bit grayscale when it is loaded. The width and height come from the file.
- The operator to run: 3×3 two-direction, 5×5 two-direction or 5×5 four-direction in scalar C, or 5×5 four-direction in AVX2 or CUDA.
- Test images at 512×512, 1024×1024 and 2048×2048, the sizes used in [1].
stb_image and stb_image_write [9] are small single-header C libraries. stb_image reads a PNG file into an array of pixel values, one byte per pixel when loaded as grayscale. stb_image_write saves an array of pixel values back out as a PNG. This keeps the PNG format out of the program, so every version works on a plain array of bytes.
 
### Outputs
 
- The Sobel edge image (0–255), written as a PNG, from the scalar version of each operator, and from the AVX2 and CUDA versions of the 5×5 four-direction operator.
- A mismatch report: the number of differing pixels for the AVX2 and CUDA versions against the scalar version, and the first mismatch.
- Timing tables: scalar time for each operator and image size, and scalar, AVX2 and CUDA time and speedup for the 5×5 four-direction operator. For CUDA, the kernel time and the copy time are reported separately.

### Implementation Versions

| Version |
|---|
| **1. 3×3 2-Direction Sobel (C)** |
| **2. 5×5 2-Direction Sobel (C)** | 
| **3. 5×5 4-Direction Sobel (C)** |
| **4. 5×5 4-Direction Sobel (AVX2/SIMD)** |

## Related work
 
| Work | Operator and platform | What it reports | How this project differs |
| --- | --- | --- | --- |
| Chang et al. [1] | 5×5 four-direction, CUDA on a Jetson AGX Xavier and a GTX 1650Ti | 6.7× and 13× over OpenCV-GPU, and 1.6× to 1.9× over its own global memory kernel | Adds scalar C and AVX2 versions of the same operator and times them against CUDA on one machine. The paper gives equations but no source code, so the kernels are written from its equations |
| Ben Fredj et al. [2] | 3×3 two-direction, CUDA on a GeForce 920M against a C++ CPU version | GPU faster than CPU, copies to and from the GPU take 81% of the GPU time, best block size 32×32 | Adds the 5×5 operators and four directions, and compares against a vectorized CPU version, not only a scalar one |
| Shi and Zhao [3] | 16-direction 5×5 on satellite images, with denoising and thresholding | Edge accuracy (FOM and AUC), no execution times | Uses multiple directions only as motivation. Reports speed, not accuracy, and does not denoise or threshold |
| Gadler [4] | 3×3 two-direction in C, OpenCV and CUDA on an MX 150 and a GTX 1080 | CUDA computation about 1.9× to 2.3× faster than C, with file I/O taking most of the total time | Adds an AVX2 version, the 5×5 operators and four directions, and, like [4], keeps file I/O out of the computation time |
| Simd library [5] | SIMD-optimized image functions for x86 and ARM, including Sobel functions | Library functions | Hand-written AVX2 assembly of the 5×5 four-direction operator |
 
None of the sources above implements the four-direction 5×5 operator with SIMD instructions, as far as our search found. The AVX2 version of 5x5 four-directional Sobel is therefore the part of this project that differs most from existing work.
 
## Parallelization plan
 
Only the 5×5 four-direction operator is parallelized, with AVX2 and with CUDA. The 3×3 and 5×5 two-direction operators stay scalar and serve as baselines. In the four-direction operator, the four gradient filters (Gx, Gy, Gd and Gdt) and the combine and clamp step are parallelized. Each output pixel depends only on its own neighbourhood in the input, so no output depends on another output.
 
### AVX2 pattern, shown on the 3×3 operator
 
The 3×3 operator is not implemented in AVX2. It is shown here because the 5×5 four-direction code uses the same widen, add and pack pattern.
 
Each vector step handles 16 output pixels at once. Pixels are 8-bit, but the sums go above 255 and below 0 before the final clamp, so each pixel is widened to `int16`, and a 256-bit `ymm` register holds 16 of them. That is enough room for every 3×3 Sobel intermediate. The 256-bit integer instructions below need AVX2; the original AVX only widened floating-point operations to 256 bits. The instructions are described in [7].
 
| Step | Instruction | Why |
| --- | --- | --- |
| Load and widen 16 pixels | `vpmovzxbw ymm, m128` | Reads 16 `uint8` pixels and widens them to `int16` in one instruction |
| Left, centre and right neighbours | three loads at x−1, x, x+1 | Unaligned loads replace any shuffling |
| Weight of 2 | `vpaddw` of a value with itself | Doubles the middle row or column term |
| Gx and Gy sums | `vpaddw`, `vpsubw` | Gx and Gy stay within −1020 to 1020, so `int16` never overflows |
| Absolute value | `vpabsw` | Gives abs(Gx) and abs(Gy) |
| abs(Gx) + abs(Gy) | `vpaddw` | Largest value is 2040, still within `int16` |
| Clamp to 255 and narrow | `vpackuswb` | Saturates `int16` to 0–255 as it narrows to `uint8` |
| Restore pixel order | `vpermq` | The pack works inside each 128-bit lane, so the two halves come out interleaved |
 
### AVX2, 5×5 four-direction operator
 
The 5×5 weights also fit in `int16`. The largest single gradient is 12240 (48 × 255), and the sum of four absolute gradients is at most 48960, which fits in `uint16` but not in `int16`.
 
| Step | Instruction | Why |
| --- | --- | --- |
| Gx, horizontal pass on each row | four loads at x−2, x−1, x+1, x+2 (`vpmovzxbw`), `vpsubw`, `vpaddw` | Difference weights are −1 −2 0 2 1. The centre weight is 0, so only four loads are needed, and the weight of 2 is a `vpaddw` of a value with itself |
| Gy, horizontal pass on each row | five loads at x−2 to x+2, `vpaddw`, `vpsllw` | Smoothing weights are 1 4 6 4 1. The weight of 4 is a left shift by 2, and the weight of 6 is 4 times the value plus 2 times the value |
| Row buffer | row results stored as `int16` rows in memory | Each row is filtered once and reused by the five output rows that need it |
| Gx and Gy, vertical pass | `vpsubw`, `vpaddw`, `vpsllw` over five row results | Gx smooths down the rows with 1 4 6 4 1. Gy takes the difference −1 −2 0 2 1 down the rows |
| Gd and Gdt | weighted taps from the matrices above, zero weights skipped | Computed directly from the weights with no row reuse, because they do not split into two 1D passes [1] |
| Absolute value and sum | `vpabsw`, `vpaddw` | The sum of four values is kept as `uint16` |
| Clamp to 255 and narrow | `vpminuw` with 255, then `vpackuswb` | Clamp first, because `vpackuswb` reads its input as signed and a sum above 32767 would become 0 |
| Restore pixel order | `vpermq` | Same reason as in the 3×3 table |
 
### CUDA, 5×5 four-direction operator
 
- **Mapping.** One thread computes one output pixel, in 2D blocks [8]. The block shape is chosen with a short test, as in [2].
- **Memory (required kernel).** A GPU has a large, slow global memory, and a small, fast shared memory for each block of threads. In the required kernel, each thread reads its own 25 pixels from global memory. Neighbouring threads need mostly the same pixels, so each pixel is requested by up to 25 different threads. This is the global memory kernel of [1]. The output is a separate buffer.
- **Optional kernel.** Each block first copies a tile of the image (its own pixels plus a 2-pixel margin) into shared memory once, and the threads read from that tile. It could be faster, but [1] found shared memory was not always faster than global memory, so it is a second kernel to compare, only if time remains.
- **Arithmetic.** The same integer sum of absolute values as the scalar version, so the output is identical.
- **Flow.** Copy the image to the GPU, run the kernel, copy the result back [2].
- **Timing.** The kernel and each copy are timed with CUDA events. A throwaway CUDA call runs first so driver start-up is not counted [4].
### Borders, tail pixels and buffers
 
- **Borders.** The first version leaves the output border at 0 (skip), and the scalar reference does the same so the two compare exactly. The border is 1 pixel wide for the 3×3 operator and 2 pixels wide for the 5×5 operators. Zero, replicate and reflect are options to add later.
- **Tail pixels.** When the width is not a multiple of 16, the last partial block is finished by the scalar loop. A vector block is used only when its 16 outputs and their 2-pixel margins all lie inside the row.
- **Buffers.** Input and output are separate arrays, so a row is never overwritten while a neighbouring row still needs it.

### Not parallelized
 
- The 3×3 and 5×5 two-direction operators, which stay scalar.
- Reading and writing the PNG, and the grayscale conversion done when the image is loaded.
- The border pixels, which stay 0, and the tail of each row, which the scalar loop finishes.
- The copies between host and GPU memory, which are timed but not accelerated.
- The verification and timing code.

## Target platform
 
- **Processor:** an x86-64 CPU with AVX2. The exact model is recorded with the benchmark results.
- **GPU:** an NVIDIA GeForce RTX GPU. The exact model is recorded with the benchmark results.
- **System and tools:** Windows 11, with Visual Studio for the C code, NASM for the assembly and the CUDA Toolkit (nvcc) for the GPU code.
- **Languages:** C for PNG reading and writing (stb_image and stb_image_write), the scalar baselines, verification and timing; NASM x86-64 assembly for the AVX2 version; CUDA C++ for the GPU kernel.
## Verification
 
The AVX2 version and the CUDA version are checked against the scalar C version of the 5×5 four-direction operator on the same input, and the outputs must be identical byte for byte, because all of them use integer arithmetic only.
 
- **Compare each gradient on its own.** Keep Gx, Gy, Gd and Gdt as separate buffers in the test build and compare each one before the combined result, so a mismatch points at one filter.
- **Check the scalar versions against OpenCV.** The scalar 3×3 and 5×5 two-direction outputs are compared with OpenCV `Sobel` (kernel sizes 3 and 5) [6], on interior pixels only, because OpenCV pads the border and this project leaves it at 0. OpenCV `Sobel` only computes the horizontal and vertical derivatives, so Gd and Gdt are checked with `filter2D` using the same kernel values.
- **Report the difference.** Count the differing pixels and print the first mismatch with its x, y and both values.
- **Use an 8×8 hand-checked image first.** Its Sobel output can be worked out on paper using the letter-labelled 3×3 method above, and one interior pixel of each 5×5 operator can be worked out with the matrices above.
- **Cover the awkward sizes.** Include widths that are not a multiple of 16, a width under 16, a very small image such as 3×3, and a 5×5 image, which has a single output pixel for the 5×5 operators, so tail handling and border policy are both exercised.
- **Look at the pictures.** Write each result as an image file and view it next to the input, since a plausible-looking number check can still hide a flipped sign or shifted row.

## Benchmarks
 
The scalar version of each operator is timed on the same image. The 5×5 four-direction operator is also timed for AVX2 and CUDA. Speedup = scalar time / AVX2 time, and scalar time / CUDA time for CUDA. Only the processing is timed, not loading or saving the image, because file I/O took most of the total time in [4] and would hide the speedup. For CUDA, the kernel time is reported on its own and together with the copies to and from the GPU, because the copies can take longer than the kernel [1], [2]. Each time is the average of repeated runs. The scalar table also shows how much the larger window and the two extra directions add to the 3×3 baseline, which [1] estimates at about eight times.
 
Results (fill in for each image size tested):
 
Scalar C, all three operators:
 
| Operator | Scalar time (ms) | Time relative to 3×3 two-direction |
| --- | --- | --- |
| 3×3 two-direction |  | 1 |
| 5×5 two-direction |  |  |
| 5×5 four-direction |  |  |
 
5×5 four-direction operator, scalar against AVX2 and CUDA:
 
| Version | Time (ms) | Speedup over scalar |
| --- | --- | --- |
| Scalar C |  | 1 |
| AVX2 |  |  |
| CUDA kernel |  |  |
| CUDA copies to and from the GPU |  | - |
| CUDA kernel and copies |  |  |
 
## Build and run
 
 
## Roadmap
 
Milestones follow the course schedule: proposal in Week 6, mid-project update demo in Week 10 and complete submission in Week 13.
 
### Mid-project milestone (Week 10)
 
- [ ] Read and write PNG images in C (stb_image and stb_image_write)
- [ ] Scalar Sobel in `main.c`
- [ ] Build the 8×8 hand-checked test image and check the scalar Sobel against the paper result
- [ ] Scalar 3x3, 5×5 two-direction and 5×5 four-direction, checked against OpenCV and the hand-checked values
- [ ] CUDA global memory kernel for the 5×5 four-direction operator, matched to the scalar version
- [ ] AVX2 Gx and Gy for the 5×5 operator, matched to the scalar Gx and Gy
### Final submission (Week 13)
 
- [ ] AVX2 Gd and Gdt computed directly, and the combined 5×5 four-direction output, matched to the scalar version
- [ ] Benchmark on all test images and fill in the results tables
- [ ] Optional: a shared memory CUDA kernel
- [ ] Write up the results and what limited the speedup

## References
 
[1] Q. Chang, X. Li, Y. Li and J. Miyazaki, "Multi-directional Sobel operator kernel on GPUs," *Journal of Parallel and Distributed Computing*, vol. 177, pp. 160–170, 2023, doi: 10.1016/j.jpdc.2023.03.004. Also available as arXiv:2305.00515.
 
[2] H. Ben Fredj, M. Ltaif, A. Ammar and C. Souani, "Parallel implementation of Sobel filter using CUDA," in *Proc. ICCAD'17*, Hammamet, Tunisia, Jan. 19–21, 2017, pp. 209–212.
 
[3] L. Shi and Y. Zhao, "Edge detection of high-resolution remote sensing image based on multi-directional improved Sobel operator," *IEEE Access*, vol. 11, pp. 135979–135993, 2023, doi: 10.1109/ACCESS.2023.3338355.
 
[4] D. Gadler, "A comparison of the Sobel filter in C, OpenCV and CUDA," course report, Programming Tools for Parallel and Distributed Systems, University of Pisa, 2019. Code: https://github.com/DanyEle/Sobel_Filter
 
[5] ermig1979, "Simd Library," GitHub repository. https://github.com/ermig1979/Simd (accessed Oct. 2026).
 
[6] OpenCV, `Sobel` and `filter2D` functions, imgproc module documentation. https://docs.opencv.org (accessed Oct. 2026).
 
[7] Intel, "Intel Intrinsics Guide." https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html (accessed Oct. 2026).
 
[8] NVIDIA, "CUDA C++ Programming Guide." https://docs.nvidia.com/cuda/cuda-c-programming-guide/ (accessed Oct. 2026).
 
[9] S. Barrett, "stb_image and stb_image_write," GitHub repository. https://github.com/nothings/stb
 
[10] I. Sobel and G. Feldman, "A 3x3 isotropic gradient operator for image processing," talk at the Stanford Artificial Intelligence Project, 1968, as cited in [1].
 
