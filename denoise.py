"""
Image denoising with Gibbs sampling on an Ising / pairwise-MRF model.

The hidden (clean) image Y and observed (noisy) image X are binary, with
pixels in {-1, +1}. The model energy is

    E(Y, X) = -eta * sum_i  Y_i X_i              (data / observation term)
              -beta * sum_<i,j> Y_i Y_j          (smoothness term, 4-neighbours)

The posterior P(Y_i = +1 | neighbours, X_i) = sigmoid(2 * w), where
    w = eta * X_i + beta * (sum of Y over the 4 neighbours).

We draw samples from the posterior with Gibbs sampling and estimate each
pixel by its posterior mean (marginal MAP). The sampler uses a vectorised
red/black ("checkerboard") sweep: same-colour pixels are conditionally
independent given the other colour, so a whole colour is updated in one
NumPy operation. This is hundreds of times faster than per-pixel Python loops.

Usage examples (see README.md for more):

    # Denoise an already-noisy image
    python denoise.py denoise Noisy.png --energy-plot --compare

    # Corrupt any clean image so you can test the denoiser on it
    python denoise.py noise my_clean_logo.png --prob 0.1 --output my_noisy.png
    python denoise.py denoise my_noisy.png --compare

    # Run the bundled E=MC^2 example end to end
    python denoise.py demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Image I/O and binarisation
# --------------------------------------------------------------------------- #
def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's method: pick the [0,1] threshold that maximises between-class
    variance. Robust default for bilevel images of any brightness/contrast."""
    hist, edges = np.histogram(gray, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return 0.5
    cum_mean = np.cumsum(hist * centers)
    mean_bg = np.where(weight_bg > 0, cum_mean / np.maximum(weight_bg, 1), 0.0)
    total_mean = cum_mean[-1]
    mean_fg = np.where(weight_fg > 0, (total_mean - cum_mean) / np.maximum(weight_fg, 1), 0.0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return float(centers[int(np.argmax(between))])


def load_binary_image(path: str, threshold="auto", invert: bool = False) -> np.ndarray:
    """Load any image and convert it to a binary {-1, +1} array.

    Handles RGB, RGBA, palette, and grayscale inputs of any bit depth by
    letting Pillow do the grayscale conversion (mode 'L', 0-255), which the
    original ``np.dot(img[..., :3], ...)`` code could not do safely.

    threshold : float in [0, 1], or "auto" for Otsu's method.
    invert    : swap foreground/background if your image is light-on-dark.
    Returns +1 for bright pixels, -1 for dark pixels.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input image not found: {path}")
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0

    if isinstance(threshold, str):
        if threshold.lower() == "auto":
            thr = otsu_threshold(gray)
        else:
            thr = float(threshold)
    else:
        thr = float(threshold)

    binary = np.where(gray > thr, 1, -1).astype(np.int8)
    if invert:
        binary = (-binary).astype(np.int8)
    return binary


def binary_to_uint8(binary: np.ndarray) -> np.ndarray:
    """Map {-1, +1} -> {0, 255} (dark, bright) for saving/visualisation."""
    return ((binary.astype(np.int16) + 1) // 2 * 255).astype(np.uint8)


def save_binary_image(binary: np.ndarray, path: str) -> None:
    """Save a {-1, +1} array as a real image at its native resolution."""
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(binary_to_uint8(binary), mode="L").save(path)


# --------------------------------------------------------------------------- #
# Noise model (for generating test inputs from clean images)
# --------------------------------------------------------------------------- #
def add_salt_pepper(binary: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Flip each binary pixel independently with probability ``prob``."""
    flip = rng.random(binary.shape) < prob
    return np.where(flip, -binary, binary).astype(np.int8)


# --------------------------------------------------------------------------- #
# Vectorised Gibbs sampler
# --------------------------------------------------------------------------- #
def _neighbour_sum(Y: np.ndarray) -> np.ndarray:
    """Sum of the 4 nearest neighbours for every pixel.

    Out-of-bounds neighbours count as 0 (equivalent to a zero-padded border).
    Implemented with in-place shifted adds to avoid allocating a padded copy
    on every sweep, which dominates the runtime on large images.
    """
    s = np.zeros(Y.shape, dtype=np.float64)
    s[1:, :] += Y[:-1, :]   # neighbour above
    s[:-1, :] += Y[1:, :]   # neighbour below
    s[:, 1:] += Y[:, :-1]   # neighbour left
    s[:, :-1] += Y[:, 1:]   # neighbour right
    return s


def _energy(Y: np.ndarray, X: np.ndarray, eta: float, beta: float) -> float:
    """Total Ising energy of configuration Y given observation X."""
    data = np.sum(Y * X)
    smooth = np.sum(Y[:-1, :] * Y[1:, :]) + np.sum(Y[:, :-1] * Y[:, 1:])
    return float(-eta * data - beta * smooth)


def gibbs_denoise(
    X: np.ndarray,
    eta: float = 1.0,
    beta: float = 1.0,
    burn_in: int = 100,
    samples: int = 1000,
    seed: int | None = 0,
    verbose: bool = True,
):
    """Run red/black Gibbs sampling and return (denoised, posterior, energies).

    denoised  : {-1, +1} array from thresholding the posterior mean at 0.5
    posterior : per-pixel P(Y_i = +1) estimated from the post-burn-in samples
    energies  : list of (step, energy, phase) with phase in {"B", "S"}
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.int8)
    Y = rng.choice(np.array([1, -1], dtype=np.int8), size=X.shape)

    # Checkerboard colour of each pixel; same-colour pixels are independent
    # given the opposite colour, so each colour is updated in one shot.
    rows, cols = np.indices(X.shape)
    color = ((rows + cols) % 2).astype(np.int8)

    posterior = np.zeros(X.shape, dtype=np.float64)
    energies: list[tuple[int, float, str]] = []
    total_steps = burn_in + samples
    Xf = X.astype(np.float64)

    for step in range(total_steps):
        for c in (0, 1):
            w = eta * Xf + beta * _neighbour_sum(Y)
            prob = 1.0 / (1.0 + np.exp(-2.0 * w))
            draw = np.where(rng.random(X.shape) < prob, 1, -1).astype(np.int8)
            mask = color == c
            Y[mask] = draw[mask]

        phase = "B" if step < burn_in else "S"
        if phase == "S":
            posterior += Y == 1
        energies.append((step, _energy(Y, X, eta, beta), phase))

        if verbose and (step % 50 == 0 or step == total_steps - 1):
            print(f"  step {step + 1:4d}/{total_steps}  energy={energies[-1][1]:.0f}", flush=True)

    posterior /= max(samples, 1)
    denoised = np.where(posterior > 0.5, 1, -1).astype(np.int8)
    return denoised, posterior, energies


# --------------------------------------------------------------------------- #
# Plotting (matplotlib imported lazily so the core tool runs without it)
# --------------------------------------------------------------------------- #
def plot_energy(energies, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array([e[0] for e in energies])
    vals = np.array([e[1] for e in energies])
    phases = np.array([e[2] for e in energies])
    burn = phases == "B"
    samp = phases == "S"

    plt.figure(figsize=(8, 5))
    if burn.any():
        plt.plot(steps[burn], vals[burn], "r", label="burn-in")
    if samp.any():
        plt.plot(steps[samp], vals[samp], "b", label="sampling")
    plt.title("Energy convergence")
    plt.xlabel("iteration")
    plt.ylabel("energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_comparison(noisy: np.ndarray, denoised: np.ndarray, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(binary_to_uint8(noisy), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Noisy input")
    axes[0].axis("off")
    axes[1].imshow(binary_to_uint8(denoised), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Denoised")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_out(input_path: str, suffix: str) -> str:
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_{suffix}.png"


def cmd_denoise(args: argparse.Namespace) -> None:
    X = load_binary_image(args.input, threshold=args.threshold, invert=args.invert)
    print(f"Loaded '{args.input}'  shape={X.shape}  "
          f"(eta={args.eta}, beta={args.beta}, burn_in={args.burn_in}, samples={args.samples})")

    t0 = time.time()
    denoised, _, energies = gibbs_denoise(
        X, eta=args.eta, beta=args.beta, burn_in=args.burn_in,
        samples=args.samples, seed=args.seed, verbose=not args.quiet,
    )
    print(f"Done in {time.time() - t0:.1f}s")

    out = args.output or _default_out(args.input, "denoised")
    save_binary_image(denoised, out)
    print(f"Saved denoised image -> {out}")

    if args.energy_plot:
        ep = _default_out(out, "energy")
        plot_energy(energies, ep)
        print(f"Saved energy plot     -> {ep}")
    if args.compare:
        cp = _default_out(out, "comparison")
        plot_comparison(X, denoised, cp)
        print(f"Saved comparison      -> {cp}")


def cmd_noise(args: argparse.Namespace) -> None:
    X = load_binary_image(args.input, threshold=args.threshold, invert=args.invert)
    rng = np.random.default_rng(args.seed)
    noisy = add_salt_pepper(X, args.prob, rng)
    out = args.output or _default_out(args.input, "noisy")
    save_binary_image(noisy, out)
    frac = float(np.mean(noisy != X))
    print(f"Added salt-and-pepper noise (p={args.prob}, actual flipped={frac:.3f})")
    print(f"Saved noisy image -> {out}")


def cmd_demo(args: argparse.Namespace) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    noisy_path = os.path.join(here, "Noisy.png")
    if not os.path.isfile(noisy_path):
        sys.exit(f"Bundled demo image not found: {noisy_path}")
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    X = load_binary_image(noisy_path, threshold="auto")
    print(f"Demo: denoising bundled Noisy.png  shape={X.shape}")
    denoised, _, energies = gibbs_denoise(X, verbose=True)

    out = os.path.join(out_dir, "demo_denoised.png")
    save_binary_image(denoised, out)
    plot_energy(energies, os.path.join(out_dir, "demo_energy.png"))
    plot_comparison(X, denoised, os.path.join(out_dir, "demo_comparison.png"))
    print(f"Demo outputs written to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Binary image denoising via Gibbs sampling on an Ising/MRF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("input", help="Path to the input image")
        sp.add_argument("-o", "--output", default=None, help="Output path (default: alongside input)")
        sp.add_argument("--threshold", default="auto",
                        help="Binarisation threshold in [0,1], or 'auto' for Otsu")
        sp.add_argument("--invert", action="store_true",
                        help="Invert foreground/background (use for light-on-dark images)")
        sp.add_argument("--seed", type=int, default=0, help="Random seed (use -1 for nondeterministic)")

    d = sub.add_parser("denoise", help="Denoise a noisy image")
    add_common(d)
    d.add_argument("--eta", type=float, default=1.0, help="Data term weight (trust in observation)")
    d.add_argument("--beta", type=float, default=1.0, help="Smoothness weight (neighbour agreement)")
    d.add_argument("--burn-in", type=int, default=100, help="Burn-in sweeps (discarded)")
    d.add_argument("--samples", type=int, default=1000, help="Posterior sweeps (averaged)")
    d.add_argument("--energy-plot", action="store_true", help="Also save the energy convergence plot")
    d.add_argument("--compare", action="store_true", help="Also save a noisy-vs-denoised comparison")
    d.add_argument("--quiet", action="store_true", help="Suppress per-step progress")
    d.set_defaults(func=cmd_denoise)

    n = sub.add_parser("noise", help="Add salt-and-pepper noise to a clean image")
    add_common(n)
    n.add_argument("--prob", type=float, default=0.1, help="Per-pixel flip probability")
    n.set_defaults(func=cmd_noise)

    demo = sub.add_parser("demo", help="Run the bundled E=MC^2 example end to end")
    demo.set_defaults(func=cmd_demo)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "seed", 0) == -1:
        args.seed = None
    try:
        args.func(args)
    except FileNotFoundError as e:
        sys.exit(
            f"Error: {e}\n"
            "Hint: pass the path to a real image file. To try the bundled "
            "example, run:  python denoise.py denoise Noisy.png --compare"
        )
    except (OSError, ValueError) as e:
        sys.exit(f"Error: could not process image: {e}")


if __name__ == "__main__":
    main()
