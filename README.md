# Diff-Mamba for Indian Pines

Diff-Mamba is a two-stage hyperspectral image classification workflow. First, a diffusion-based U-shaped Vision Transformer (UViT) denoiser is trained without class labels and used as a frozen feature extractor. Then, a supervised classifier combines the extracted diffusion features with raw spectral patches. Its Mamba and center pixel-driven token generator (CPSTG) components aggregate local spatial context to predict a land-cover class for each pixel.

This repository applies the workflow to the Indian Pines hyperspectral dataset, using 200 spectral bands and 16 foreground classes.

## Architecture

```text
Hyperspectral cube
       │
       ├── Random 64×64 crops ──► DDPM + 3D UViT denoiser
       │                                  │
       │                         Decoder features at 6 timesteps
       │                                  │
       └── 9×9 spectral patches ──────────┤
                                          ▼
                       ┌────────────────────────────────┐
                       │ Spectral branch: CNN +         │
                       │ Transformer                    │
                       ├────────────────────────────────┤
                       │ Diffusion branch: projection + │
                       │ positional convolution +       │
                       │ Mamba + CPSTG                  │
                       └───────────────┬────────────────┘
                                       ▼
                              MLP classification head
                                       ▼
                              16 class logits
```

### Diffusion model

The denoiser is trained with a denoising diffusion probabilistic model (DDPM) objective. Noise is added to unlabeled image crops, and the model learns to predict that noise using mean squared error. The diffusion process uses 500 steps and a linear beta schedule from `1e-4` to `2e-2`.

### UViT denoiser

The UViT accepts inputs shaped `(B, 1, 200, 64, 64)` and predicts noise with the same shape. A 3D convolution first extracts joint spectral-spatial features. The result is rearranged and processed by a 2D convolution, then combined with a timestep embedding and a depthwise-convolution positional encoding.

The transformer encoder-decoder uses a hidden width of 800 (`4 × 200`) and eight attention heads. With the default depth of six, the implementation has three encoder blocks, one middle block, and three decoder blocks. Decoder blocks add matching encoder skip features. During feature extraction, the output from each of the three decoder blocks can be retained.

### Diffusion feature extraction

The trained UViT is frozen and applied to the scene at timesteps `5, 10, 50, 100, 200, and 400`. Features from all three decoder stages are saved as spatial maps aligned to the 145×145 scene. Each feature map has 800 channels and is stored in float16 format.

### Classifier

For each labeled pixel, the classifier receives aligned 9×9 patches from the raw spectral cube and the concatenated diffusion feature maps.

- **Spectral branch:** A 1×1 convolution projects the spectral input to 128 channels. The 81 spatial tokens are processed by a one-layer Transformer encoder and mean pooled.
- **Diffusion branch:** A 1×1 convolution projects the concatenated diffusion features to 128 channels. A depthwise 3×3 convolution adds local positional information. The resulting 81-token sequence is processed by one residual Mamba block with state size 16.
- **CPSTG:** The center pixel token guides aggregation of the Mamba sequence. Tokens are weighted using their sequence distance and feature distance from the center token.
- **Classification head:** The CPSTG semantic feature, mean-pooled Mamba feature, and mean-pooled spectral Transformer feature are concatenated and passed through an MLP to produce 16 class logits.

## Dataset and evaluation

The configured dataset is `data/Indian_30_split.mat`. It contains the hyperspectral input cube and separate training (`TR`) and test (`TE`) labels. Background label 0 is excluded from supervised classification; labels 1–16 correspond to the 16 foreground classes.

The image is normalized band by band to `[-1, 1]`. This preprocessing uses the full unlabeled image cube. The diffusion stage trains on randomly sampled 64×64 crops. The classifier uses labeled training coordinates, while the test coordinates are reserved for final evaluation.

Evaluation reports overall accuracy (OA), average per-class accuracy (AA), Cohen’s kappa, per-class accuracy, and a confusion matrix. The full-scene classification map shows predicted classes and is not a ground-truth map.

## Default configuration

| Setting | Default |
| --- | ---: |
| Spectral bands | 200 |
| Foreground classes | 16 |
| Diffusion steps | 500 |
| Diffusion patch size | 64×64 |
| Diffusion training epochs | 20,000 |
| Diffusion batch size | 2 |
| Random crops per diffusion epoch | 512 |
| UViT hidden width / attention heads | 800 / 8 |
| UViT encoder / middle / decoder blocks | 3 / 1 / 3 |
| Feature timesteps | 5, 10, 50, 100, 200, 400 |
| Classifier patch size | 9×9 |
| Classifier hidden width | 128 |
| Mamba layers / state size | 1 / 16 |
| Classifier training epochs | 30 |
| Random seed | 2025 |

## Implementation scope

This repository adapts the described Diff-Mamba workflow to the supplied Indian Pines dataset and split. In particular, the Mamba sequence is formed from each supervised 9×9 local patch, rather than from a global scan of the entire scene. Some classifier integration details are implementation choices where a complete classifier implementation was not present in the supplied source. The implementation should therefore not be interpreted as a verified exact reproduction of every paper detail.
