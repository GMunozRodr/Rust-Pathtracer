

# Rust Path Tracer

A path tracer I wrote in Rust to explore acceleration structures and sampling techniques. The rendering itself is standard Monte Carlo with MIS, the interesting parts are in the optimizations.

![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange)

<img width="1920" height="1080" alt="Bistro" src="https://github.com/user-attachments/assets/75255e26-7401-4e9d-841f-0d640157d8fb" />

Source: [Amazon Lumberyard Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro). Converted to GLB using [Blender](https://www.blender.org/).

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/da869fa8-d294-488d-b195-591325b3c7fd" />

Source: Custom. Done before triangle and glTF support was implemented.

## Implemented Features

### 8-Wide BVH with AVX2

I have done acceleration structures of different types before, so this time I wanted to try something more dedicated this time. The result was a wide BVH that used SIMD to speed up the process. The idea is to test 8 child nodes at once using SIMD instead of traversing them one by one. The construction works in two steps: first build a normal binary BVH using surface area heuristic, then collapse groups of nodes into 8-wide nodes by repeatedly picking the largest children to expand. The nodes are laid out to be 64 bytes so they align with cache lines. There's a fallback using the `wide` crate for non-x86. Overall, I could maybe have done better, but this felt like a nice excuse to use SIMD intrinsics.

Resources:
- [Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs](https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs) — Ylitie, Karras, Laine
- [On Quality Metrics of Bounding Volume Hierarchies](https://meistdan.github.io/publications/bvh_star/paper.pdf) — Aila et al.

### Two-Level Acceleration (TLAS/BLAS)

This is how the Vulkan API handles instancing, so I wanted to imitate it. Each mesh gets its own acceleration structure (BLAS), and those are placed into a top-level structure (TLAS) with transforms.

Resources:
- [Vulkan Ray Tracing](https://www.khronos.org/blog/ray-tracing-in-vulkan) — Khronos Group
- [Introduction to Real-Time Ray Tracing with Vulkan](https://developer.nvidia.com/blog/vulkan-raytracing/) — NVIDIA

### Adaptive Sampling

Not all pixels need the same number of samples. I track the running variance for each pixel and focus samples on the noisy ones. Once a pixel's error estimate drops below a threshold, it stops getting new samples. This can dramatically increase the speed of sampling in some cases like scenes where a lot of rays hit the sky directly.

Resources:
- [Adaptive Sampling and Reconstruction using Greedy Error Minimization](https://www.cs.umd.edu/~zwicker/publications/AdaptiveSamplingGreedyError-SIGA11.pdf) — Rousselle et al.
- [A Hierarchical Automatic Stopping Condition for Monte Carlo Global Illumination](https://jo.dreggn.org/home/2009_stopping.pdf) — Dammertz et al.

### Environment Map Importance Sampling

When sampling the environment for lighting, you want to send more rays toward the bright parts of the HDR map. I precompute a 2D probability distribution from the image luminance and sample from it directly. The tricky part is accounting for how pixels near the poles of the sphere cover less area than pixels at the equator. Luckily, this technique is very well documented online.

### Blue Noise Generation

Random sampling tends to clump, which shows up as noise patterns in the image. Blue noise spreads samples more evenly. I implemented the void-and-cluster algorithm to generate the noise textures at startup, then combine them with low-discrepancy sequences across frames. The first few samples look noticeably better than with pure random. My technique is rather simple and there are much better blue noise resources out there that I could have used, but I wanted to at least try implementing something myself as an educational challenge.

Resources:
- [The void-and-cluster method for dither array generation](https://cv.ulichney.com/papers/1993-void-cluster.pdf) — Ulichney
- [Blue-noise Dithered Sampling](https://blogs.autodesk.com/media-and-entertainment/2024/01/04/autodesk-arnold-research-papers/#bluenoise-dithered-sampling) — Georgiev, Fajardo

### Random Walk Subsurface Scattering

For translucent materials like skin or wax, light doesn't just bounce off the surface — it scatters around inside. I simulate this by random-walking through the volume, with different scatter distances for red, green, and blue (since red light penetrates deeper in skin, for example). This was the lighting feature that took me the most to implement, and I have to admit I am not too happy with the result, but it will do for now.

### glTF Loader

I went with glTF over FBX because it packs textures and materials into a single file.

### Tonemapping

A few different tonemappers to convert HDR to displayable colors:
- **Reinhard**
- **ACES**
- **AgX**

## Build & Run

```bash
cargo build --release
cargo run --release -- path/to/model.glb
cargo run --release -- --benchmark path/to/model.glb
```

Controls: WASD + mouse to move, scroll for speed, `A` toggles adaptive sampling, `T` cycles tonemapping, `[`/`]` adjusts exposure.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `glam` | Vector/matrix math |
| `rayon` | Parallel rendering |
| `gltf` | Scene loading with PBR extensions |
| `image` | PNG/EXR I/O |
| `oidn` | Intel Open Image Denoiser |
| `minifb` | Preview window |
| `wide` | Portable SIMD fallback |

