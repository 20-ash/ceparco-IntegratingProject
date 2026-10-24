# Multi-Directional 5×5 Sobel Edge Detection with AVX2

**Group 4**
- Copon, Matthew Joshua
- Lee, Ashley Fiona
- Ong, Eiress Bassey
- Xu, Kai Wen

## Abstract

The Sobel operator is one of the most popular edge detection operators, with applications in remote sensing, medical image processing and industrial detection [1]. It finds the approximate absolute gradient magnitude at each pixel [1]. Standard implementations use a 3×3 kernel that considers only horizontal and vertical directions. Chang et al. [1] expanded the operator to a 5×5 size with four directional filters (0°, 90°, 45°, 135°), adding two diagonal directions. They state that the 5×5 operator is less affected by noise than the 3×3 operator and that four directions show more texture than two, which they compare visually in their Fig. 1. Their implementation targets NVIDIA GPUs using CUDA, which requires dedicated hardware.

This project implements the four-direction 5×5 Sobel operator in both <mark> 3×3 and 5×5 sizes, each as three kernels: a C base kernel, a C optimized kernel and an AVX2 kernel. The AVX2 SIMD instructions, which need no specialized hardware. The C base kernel computes every filter directly from its weights and is the golden reference for error checking. The C optimized kernel and the AVX2 kernel use the separable form for the horizontal and vertical filters and the sum-difference transformation of [1] for the diagonal filters. Through data-level parallelism, the group aims to achieve speedup over the C optimized kernel (the SIMD speedup, same algorithm) and the C base kernel (the total speedup). The group will also compare the 5×5 and 3×3 kernels, and the two-directional and four-directional implementations, in the C base kernel. Every optimized kernel is checked against the C base kernel gradient by gradient, before any absolute value or sum, and the base kernel is checked against hand-worked values and OpenCV.</mark> Execution time and speedup are measured at 512×512, 1024×1024 and 2048×2048.

## Presentation pitch of the proposed project
[Watch the project proposal video](https://drive.google.com/file/d/1-72TuUAJVy37nJjmVDsZhd3dUHP6azdz/view?usp=sharing)

[PPT](https://canva.link/cnkb406mfp4stuf)

## Proposed Algorithm
### 5×5
The filters of the four-directional 5×5 Sobel operator are defined as follows [1]. The paper names the two diagonal filters Gd and Gdt without saying which is the 45° one and which is the 135° one, so they are not labelled with angles here:

Horizontal Gradient (0°)

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

Vertical Gradient (90°)

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

Diagonal Gradient (Gd)

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

Diagonal Gradient (Gdt)

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

<mark>
$$
G = |G_x| + |G_y| + |G_d| + |G_{dt} 
$$
</mark>

The 5×5 two-direction operator uses only $G_x$ and $G_y$, with <mark> $G = G = |G_x| + |G_y|$.</mark>

Chang et al. [1] combine the directions with the square root of the sum of squares, $G = \sqrt{G_x^2 + G_y^2 + G_d^2 + G_{dt}^2}$. This project uses the sum of absolute values instead. The absolute value is needed because a gradient can be negative, and opposite signs would otherwise cancel each other. A sum is used instead of a square root so every step stays in integer arithmetic, the scalar and AVX2 outputs can be compared exactly, and AVX2 can process 16 pixels per step without a floating-point conversion. The sum only approximates the true gradient magnitude, so its values are larger than the square root version and vary somewhat with edge direction. This is acceptable because the project compares speed and agreement between versions, not edge quality. A four-direction 5×5 operator needs about eight times the arithmetic of the two-direction 3×3 operator [1]. Shi and Zhao [2] go further to 16 directions; this project stops at four. Canny edge detection, which uses Sobel as its gradient step, is outside the scope of this project.

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

This means that each gradient can be computed first by applying the row vector horizontally across every row, then applying the column vector vertically to the intermediate results. The separable form produces uniform and repeating weight patterns which exposes the data-level parallelism needed for SIMD execution. AVX2 can apply these identical weights to 16 output positions at once, because each pixel is widened to 16 bits and a 256-bit YMM register holds 16 of them.

#### Sum-Difference Transformation for Diagonals
Gd and Gdt are not separable, so the project applies the sum-difference method from [1] in the AVX2 version. Instead of filtering with Gd and Gdt, the image is filtered with their sum and difference kernels:

$$
\begin{aligned}
S &= G_d + G_{dt} \\
D &= G_d - G_{dt}
\end{aligned}
$$

The kernel that produces S is:

$$
S =
\begin{bmatrix}
-6 & -6 & -2 & -6 & -6 \\
-2 & -12 & -16 & -12 & -2 \\
0 & 0 & 0 & 0 & 0 \\
2 & 12 & 16 & 12 & 2 \\
6 & 6 & 2 & 6 & 6
\end{bmatrix}
$$

The middle row contains only zeros, so its contribution is skipped entirely. Moreover, row 0 = -row 4 and row 1 = -row 3, so each of the two distinct rows is filtered horizontally once and reused with a sign flip. Writing A for the horizontal result of row 0's weights (-6 -6 -2 -6 -6) and B for the horizontal result of row 1's weights (-2 -12 -16 -12 -2), the response at row y is

$$
S(y) = A(y-2) + B(y-1) - B(y+1) - A(y+2)
$$

The kernel that produces D is:

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

The middle column contains only zeros, so its contribution is skipped entirely. Moreover, col 0 = -col 4 and col 1 = -col 3. D is the transpose of S, so the same weights are applied vertically first. Writing V0 for the vertical result of column 0's weights (-6 -6 -2 -6 -6) and V1 for the vertical result of column 1's weights (-2 -12 -16 -12 -2), the response at column x is

$$
D(x) = V0(x-2) - V0(x+2) + V1(x-1) - V1(x+1)
$$

S and D are still two 5×5 filters, but each needs only two distinct 1-D weight sets and four additions or subtractions in its second pass, instead of a full 25-tap sum. The original responses are recovered as

$$
G_d = \frac{S + D}{2}, \qquad G_{dt} = \frac{S - D}{2}
$$

Every entry of S and D is even, so S + D and S − D are even and the division by 2 is exact in integer arithmetic.

**Combining without recovering Gd and Gdt.** For any two numbers a and b, |a| + |b| = max(|a + b|, |a − b|). Applied to the diagonal responses, this gives

$$
|G_d| + |G_{dt}| = \max(|S|,\ |D|)
$$

so the final result can be written, with no halving step, as

$$
G = \min(255,\ |G_x| + |G_y| + \max(|S|,\ |D|))
$$

The AVX2 version uses this form. Gd and Gdt are still recovered with the formulas above in the test build, so each can be compared with the scalar version on its own (see Verification).

The scalar C version computes Gd and Gdt directly from their 25 weights. This keeps the scalar version an independent reference, so a match between AVX2 and scalar also checks the sum-difference derivation.

### 3×3 for C implementation
The standard 3×3 two-directional Sobel operator serves as our baseline for correctness and performance comparison. It uses only horizontal and vertical gradients:

Horizontal Gradient (0°)

$$
G_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1 \\
\end{bmatrix} * I
$$

Vertical Gradient (90°)

$$
G_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1 \\
\end{bmatrix} * I
$$

Final results are aggregated as follows, and capped at 255:

$$
G = \min(255,\ |G_x| + |G_y|)
$$

## Implementation of the Proposed Project
| Phase | Deliverables |
|---|---|
| **1. Basic Version** | Implement the 3×3 two-direction, 5×5 two-direction and 5×5 four-direction operators in standard C, with Gd and Gdt computed directly from their weights. Verify the horizontal and vertical gradients against OpenCV `Sobel` and the diagonal gradients against OpenCV `filter2D`. |
| **2. AVX2 Optimization** | Write the 5×5 four-direction operator in AVX2 assembly (NASM): separable Gx and Gy, and S and D with the sum-difference transformation for the diagonals. |
| **3. Testing and Comparison** | Compare the 5×5 four-direction operator in standard C and AVX2, and compare the 3×3 two-direction, 5×5 two-direction and 5×5 four-direction operators in standard C. Measure execution time and speedup against the basic version using images of sizes 512×512, 1024×1024, and 2048×2048. |
| **4. Final Output** | Provide the working source code and a report of the performance results. |

### Inputs
 
- A PNG image, read with stb_image [7] and converted to 8-bit grayscale when it is loaded. The width and height come from the file.
- The operator to run: 3×3 two-direction, 5×5 two-direction or 5×5 four-direction in scalar C, or 5×5 four-direction in AVX2.
- Test images at 512×512, 1024×1024 and 2048×2048, the sizes used in [1].

stb_image and stb_image_write [7] are small single-header C libraries. stb_image reads a PNG file into an array of pixel values, one byte per pixel when loaded as grayscale. stb_image_write saves an array of pixel values back out as a PNG. This keeps the PNG format out of the program, so every version works on a plain array of bytes.
 
### Outputs
 
- The Sobel edge image (0–255), written as a PNG, from the scalar version of each operator and from the AVX2 version of the 5×5 four-direction operator.
- A mismatch report: the number of differing pixels for the AVX2 version against the scalar version, and the first mismatch.
- Timing tables: scalar time for each operator and image size, and scalar and AVX2 time and speedup for the 5×5 four-direction operator.

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
| Chang et al. [1] | 5×5 four-direction, CUDA on a Jetson AGX Xavier and a GTX 1650Ti | 6.7× and 13× over OpenCV-GPU, and 1.6× to 1.9× over its own global memory kernel | Targets CPU SIMD instead of a GPU, and applies the same sum-difference idea to the diagonal filters with 16-bit AVX2 lanes. The paper gives equations but no source code, so the AVX2 routines are written from its equations |
| Shi and Zhao [2] | 16-direction 5×5 on satellite images, with denoising and thresholding | Edge accuracy (MAE, MSE, NMSE, SNR, PSNR, FOM and AUC), no execution times | Uses multiple directions only as motivation. Reports speed, not accuracy, and does not denoise or threshold |
| Gadler [3] | 3×3 two-direction in C, OpenCV and CUDA on an MX 150 and a GTX 1080 | Computation time of C and CUDA versions, with file I/O taking most of the total time | Adds an AVX2 version, the 5×5 operators and four directions, and, like [3], keeps file I/O out of the computation time |
| Simd library [4] | SIMD-optimized image functions for x86 and ARM, including Sobel functions | Library functions | Hand-written AVX2 assembly of the 5×5 four-direction operator |
 
None of the sources above implements the four-direction 5×5 operator with SIMD instructions, as far as our search found. The AVX2 version of 5×5 four-directional Sobel is therefore the part of this project that differs most from existing work.
 
## Parallelization plan
 
Only the 5×5 four-direction operator is parallelized, with AVX2. The 3×3 and 5×5 two-direction operators stay scalar and serve as baselines. In the four-direction operator, the four filters (Gx, Gy, S and D) and the combine and clamp step are parallelized. Each output pixel depends only on its own neighbourhood in the input, so no output depends on another output.
 
### AVX2 pattern, shown on the 3×3 operator
 
The 3×3 operator is not implemented in AVX2. It is shown here because the 5×5 four-direction code uses the same widen, add and pack pattern.
 
Each vector step handles 16 output pixels at once. Pixels are 8-bit, but the sums go above 255 and below 0 before the final clamp, so each pixel is widened to `int16`, and a 256-bit `ymm` register holds 16 of them. That is enough room for every 3×3 Sobel intermediate. The 256-bit integer instructions below need AVX2; the original AVX only widened floating-point operations to 256 bits. The instructions are described in [6].
 
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
 
The 5×5 weights also fit in `int16`. The largest single gradient is 12240 (48 × 255) for Gx, Gy, Gd and Gdt. The largest S or D response is 17850 (70 × 255, where 70 is the sum of the positive weights), which also fits. The sum |Gx| + |Gy| + max(|S|, |D|) is at most 42330, which fits in `uint16` but not in `int16`. Intermediate sums may wrap in `int16` without harm, because two's complement addition is exact modulo 2^16 and every final value fits.
 
| Step | Instruction | Why |
| --- | --- | --- |
| Gx, horizontal pass on each row | four loads at x−2, x−1, x+1, x+2 (`vpmovzxbw`), `vpsubw`, `vpaddw` | Difference weights are −1 −2 0 2 1. The centre weight is 0, so only four loads are needed, and the weight of 2 is a `vpaddw` of a value with itself |
| Gy, horizontal pass on each row | five loads at x−2 to x+2, `vpaddw`, `vpsllw` | Smoothing weights are 1 4 6 4 1. The weight of 4 is a left shift by 2, and the weight of 6 is 4 times the value plus 2 times the value |
| S, horizontal pass on each row (A and B) | five loads at x−2 to x+2, `vpaddw` of the symmetric pairs, `vpmullw` with broadcast constants (or shift-add) | A uses weights −6 −6 −2 −6 −6 and B uses −2 −12 −16 −12 −2. Both are symmetric, so the four outer pixels are summed first and then scaled. The two results are stored as separate `int16` row buffers |
| Row buffers | Gx, Gy, A and B results stored as `int16` rows in memory | Each row is filtered once and reused by the five output rows that need it |
| Gx and Gy, vertical pass | `vpsubw`, `vpaddw`, `vpsllw` over five row results | Gx smooths down the rows with 1 4 6 4 1. Gy takes the difference −1 −2 0 2 1 down the rows |
| S, vertical pass | `vpaddw`, `vpsubw` over four row-buffer reads | S(y) = A(y−2) + B(y−1) − B(y+1) − A(y+2). The middle row is skipped |
| D, vertical pass on each input row (V0 and V1) | five row loads, `vpaddw` of the symmetric pairs, `vpmullw` with broadcast constants (or shift-add) | D is the transpose of S, so the same weights are applied down the rows first. V0 and V1 are stored as `int16` row buffers |
| D, horizontal pass | four unaligned loads of V0 at x−2, x+2 and V1 at x−1, x+1, then `vpsubw`, `vpaddw` | D(x) = V0(x−2) − V0(x+2) + V1(x−1) − V1(x+1). The middle column is skipped |
| Absolute value | `vpabsw` on Gx, Gy, S and D | Gives |Gx|, |Gy|, |S| and |D| |
| Diagonal term | `vpmaxsw` of |S| and |D| | |Gd| + |Gdt| = max(|S|, |D|), so no halving is needed. Both inputs are non-negative and at most 17850, so a signed maximum is safe |
| Sum | `vpaddw` | |Gx| + |Gy| + the diagonal term, kept as `uint16` |
| Clamp to 255 and narrow | `vpminuw` with 255, then `vpackuswb` | Clamp first, because `vpackuswb` reads its input as signed and a sum above 32767 would become 0 |
| Restore pixel order | `vpermq` | Same reason as in the 3×3 table |
| Test build only: recover Gd and Gdt | `vpaddw`, `vpsubw`, then `vpsraw` by 1 | (S + D) / 2 and (S − D) / 2 are exact because S and D are even. Used to compare each diagonal with the scalar version |

**Calling convention.** On Windows x64, `xmm6` to `xmm15` are callee-saved. The assembly routine saves and restores any of them it uses, and executes `vzeroupper` before returning.

### Borders, tail pixels and buffers
 
- **Borders.** The first version leaves the output border at 0 (skip), and the scalar reference does the same so the two compare exactly. The border is 1 pixel wide for the 3×3 operator and 2 pixels wide for the 5×5 operators. Zero, replicate and reflect are options to add later.
- **Tail pixels.** When the width is not a multiple of 16, the last partial block is finished by the scalar loop. A vector block is used only when its 16 outputs and their 2-pixel margins all lie inside the row.
- **Buffers.** Input and output are separate arrays, so a row is never overwritten while a neighbouring row still needs it. The row buffers for Gx, Gy, A, B, V0 and V1 are separate `int16` arrays.

### Not parallelized
 
- The 3×3 and 5×5 two-direction operators, which stay scalar.
- Reading and writing the PNG, and the grayscale conversion done when the image is loaded.
- The border pixels, which stay 0, and the tail of each row, which the scalar loop finishes.
- The verification and timing code.

## Target platform
 
- **Processor:** an x86-64 CPU with AVX2. The exact model is recorded with the benchmark results.
- **System and tools:** Windows 11, with Visual Studio for the C code and NASM for the assembly.
- **Languages:** C for PNG reading and writing (stb_image and stb_image_write), the scalar baselines, verification and timing; NASM x86-64 assembly for the AVX2 version.

## Verification
 
The AVX2 version is checked against the scalar C version of the 5×5 four-direction operator on the same input, and the outputs must be identical byte for byte, because both use integer arithmetic only.
 
- **Compare each gradient on its own.** Keep Gx, Gy, S, D, Gd and Gdt as separate buffers in the test build and compare each one before the combined result, so a mismatch points at one filter. The scalar S and D buffers are computed as Gd + Gdt and Gd − Gdt from the direct scalar diagonals. The AVX2 Gd and Gdt are recovered from its S and D.
- **Check the scalar versions against OpenCV.** The scalar 3×3 and 5×5 two-direction outputs are compared with OpenCV `Sobel` (kernel sizes 3 and 5) [5], on interior pixels only, because OpenCV pads the border and this project leaves it at 0. OpenCV `Sobel` only computes the horizontal and vertical derivatives, so Gd and Gdt are checked with `filter2D` using the same kernel values.
- **Report the difference.** Count the differing pixels and print the first mismatch with its x, y and both values.
- **Use an 8×8 hand-checked image first.** Its output can be worked out on paper for the 3×3 operator with the 3×3 matrices above, and for one interior pixel of each 5×5 operator with the 5×5 matrices above, including S and D at that pixel.
- **Cover the awkward sizes.** Include widths that are not a multiple of 16, a width under 16, a very small image such as 3×3, and a 5×5 image, which has a single output pixel for the 5×5 operators, so tail handling and border policy are both exercised.
- **Look at the pictures.** Write each result as an image file and view it next to the input, since a plausible-looking number check can still hide a flipped sign or shifted row.

## Benchmarks
 
The scalar version of each operator is timed on the same image. The 5×5 four-direction operator is also timed for AVX2. Speedup = scalar time / AVX2 time. Only the processing is timed, not loading or saving the image, because file I/O took most of the total time in [3] and would hide the speedup. Each time is the average of repeated runs. The scalar C baseline is built in the Visual Studio Release configuration, and its compiler flags are recorded with the benchmark results. The compiler's auto-vectorization report is checked, so the baseline is stated as scalar or auto-vectorized. The scalar table also shows how much the larger window and the two extra directions add to the 3×3 baseline, which [1] estimates at about eight times.

The scalar baseline computes the diagonals directly, while AVX2 uses the sum-difference method, so the AVX2 speedup combines the gain from SIMD and the gain from the cheaper diagonal computation.
 
Results (fill in for each image size tested):
 
Scalar C, all three operators:
 
| Operator | Scalar time (ms) | Time relative to 3×3 two-direction |
| --- | --- | --- |
| 3×3 two-direction |  | 1 |
| 5×5 two-direction |  |  |
| 5×5 four-direction |  |  |
 
5×5 four-direction operator, scalar against AVX2:
 
| Version | Time (ms) | Speedup over scalar |
| --- | --- | --- |
| Scalar C |  | 1 |
| AVX2 |  |  |
 
## Roadmap
 
Milestones follow the course schedule: proposal in Week 6, mid-project update demo in Week 10 and complete submission in Week 13.
 
### Mid-project milestone (Week 10)
 
- [ ] Read and write PNG images in C (stb_image and stb_image_write)
- [ ] Build the 8×8 hand-checked test image and check the scalar Sobel against the hand-worked result
- [ ] Scalar 3×3, 5×5 two-direction and 5×5 four-direction (diagonals computed directly), checked against OpenCV and the hand-checked values
- [ ] AVX2 Gx and Gy for the 5×5 operator, matched to the scalar Gx and Gy

### Final submission (Week 13)
 
- [ ] AVX2 S and D with the sum-difference transformation, with Gd and Gdt recovered in the test build and matched to the scalar Gd and Gdt
- [ ] AVX2 combined 5×5 four-direction output, matched to the scalar version byte for byte
- [ ] Benchmark on all test images and fill in the results tables
- [ ] Write up the results and what limited the speedup

## AI Usage
 
The group used Claude AI, by Anthropic during the proposal stage for the following:
 
- **Revising this README.** Claude AI helped reword and restructure sections, make the notation and references consistent, and update the document when the project scope changed throughout revisions.
- **Double-checking against the papers.** The AI was given source papers, which it used to check the kernel values, the numbers attributed to each paper, and the claims in the related work table. It also checked the overflow bounds and the sum-difference formulas, including a numerical check of the S and D kernels.
- **Suggesting syntax for the parallelism plan.** Claude AI suggested AVX2 instructions and the order of steps for the plan above (for example, `vpmovzxbw`, `vpabsw`, `vpminuw` before `vpackuswb`, and `vpermq`). These suggestions were based on Intel Intrinsics Guide [6] and will be tested against the scalar version before being used.

## References
 
[1] Q. Chang, X. Li, Y. Li and J. Miyazaki, "Multi-directional Sobel operator kernel on GPUs," *Journal of Parallel and Distributed Computing*, vol. 177, pp. 160–170, 2023, doi: 10.1016/j.jpdc.2023.03.004. Also available as arXiv:2305.00515.
 
[2] L. Shi and Y. Zhao, "Edge detection of high-resolution remote sensing image based on multi-directional improved Sobel operator," *IEEE Access*, vol. 11, pp. 135979–135993, 2023, doi: 10.1109/ACCESS.2023.3338355.
 
[3] D. Gadler, "A comparison of the Sobel filter in C, OpenCV and CUDA," course report, Programming Tools for Parallel and Distributed Systems, University of Pisa, 2019. Code: https://github.com/DanyEle/Sobel_Filter
 
[4] ermig1979, "Simd Library," GitHub repository. https://github.com/ermig1979/Simd (accessed Oct. 2026).
 
[5] OpenCV, `Sobel` and `filter2D` functions, imgproc module documentation. https://docs.opencv.org (accessed Oct. 2026).
 
[6] Intel, "Intel Intrinsics Guide." https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html (accessed Oct. 2026).
 
[7] S. Barrett, "stb_image and stb_image_write," GitHub repository. https://github.com/nothings/stb (accessed Oct. 2026).
