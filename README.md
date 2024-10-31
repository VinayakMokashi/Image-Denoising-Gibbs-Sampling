# Image Denoising using Gibbs Sampling:
This repository implements image denoising using Gibbs Sampling, a Markov Chain Monte Carlo (MCMC) technique. The objective is to restore a noisy image by leveraging probabilistic inference through a pairwise Markov Random Field (MRF) model, effectively reducing noise and enhancing image clarity.

## Project Overview:
Image denoising is a key step in image processing, helping to remove noise and restore images closer to their original quality. This project applies Gibbs Sampling for Maximum a Posteriori (MAP) estimation, optimizing the likelihood of each pixel's state based on its neighbors.

## Key Features:
__Gibbs Sampling (MCMC)__: Generates a Markov chain of sample states for denoising.
__Pairwise MRF Model:__ Establishes probabilistic dependencies between neighboring pixels, allowing local optimization.
__Energy Minimization:__ Uses energy functions to guide sampling, ensuring convergence toward a noise-free image.

## Project Structure:
Implementation of Gibbs Sampling for probabilistic inference on noisy images.
Image Loading and Preprocessing: Grayscale conversion and noise addition.
Visualization: Displays the denoised image and plots energy convergence.

## Results:
The denoising process shows marked improvement in image quality, with Gibbs Sampling effectively reducing noise. Energy convergence graphs demonstrate the stability and effectiveness of the optimization process.

### Image denoising.ipynb is the code file and Noisy.png and Denoised.png are the before and after results using the code.
