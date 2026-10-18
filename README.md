# Multi-Directional 5×5 Sobel Edge Detection

**Group 4**
- Copon, Matthew Joshua
- Lee, Ashley Fiona
- Ong, Eiress Bassey
- Xu, Kai Wen

## Description of the Project
Edge detection is a foundational step in remote sensing, medical image processing, and industrial detection. The Sobel operator is the most widely adapted method to find the approximate absolute gradient magnitude at each pixel due to its simplicity and efficiency. However, standard implementations rely on a 3x3 kernel considering on horizontal and vertical directions. 

Cheng et al. (2023) expanded the Sobel operator to 5x5 size with four directional filters (0◦, 90◦, 45◦, 135◦) considering two additional diagonal directions. Results showed that compared to 3x3, the 5x5 operator is more insensitive to changes and less affected by noise since it takes more pixels or greater area into consideration. This helps provide clearer edge features which makes edge detection clearer and cleaner. Moreover, with the implementation of the four directional filter instead of two directional, more abundant textures were shown during edge detection. Their implementation, however, was optimized for NVIDIA GPUs using CUDA, which requires dedicated hardware and incurring data transfer overhead that limits deployment on standard systems.

This project addresses the gap by optimizing the four directional 5x5 Sobel operator for CPUs using AVX2 SIMD instructions. Through applying data-level parallelism, the group sees the vision to achieve significant speedup without requiring specialized hardware.

## Proposed Algorithm
## References
- Chang, Q., Li, X., Li, Y., & Miyazaki, J. (2023). Multi-directional Sobel operator kernel on GPUs. *Journal of Parallel and Distributed Computing, 177*, 160–170.
- Ben Fredj, H., Ltaif, M., Ammar, A., & Souani, C. (2017). Parallel implementation of Sobel filter using CUDA. *Proceedings of ICCAD '17*, 209–212.
- Shi, L., & Zhao, Y. (2023). Edge detection of high-resolution remote sensing image based on multi-directional improved Sobel operator. *IEEE Access, 11*, 135979–135993.
## Implementation of the Proposed Project
## Presentation Pitch of the Proposed Project
