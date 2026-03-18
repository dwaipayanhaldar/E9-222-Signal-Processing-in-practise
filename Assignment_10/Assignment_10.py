#!/usr/bin/env python3
"""
STUDENT LAB VERSION (SKELETON)

Image deblurring with Hessian mixed spectral regularization, solved by ADMM.

Students will implement ONLY the proximal mapping for the mixed penalty:
    R(M) = rw*||M||_* + (1-rw)*||M||_F
applied per-pixel to 2x2 Hessian matrices.

Everything else (FFT-based x-update, Hessian operator, ADMM loop) is provided.

---------------------------------------------------------------------------
TASK:
  Implement prox_mixed_nuclear_frob(Z, tau, rw)

  Input:
    Z   : array of shape (..., 2, 2) containing a batch of 2x2 matrices
    tau : positive scalar (tau = lambda / rho)
    rw  : scalar in [0,1] giving weight between nuclear (rw) and Frobenius (1-rw)

  Output:
    Z_prox : array with same shape as Z.

Guidance:
  1) For each 2x2 matrix Z_i, compute SVD: Z_i = U_i diag(s_i) V_i^T
  2) Let s_i be the vector of singular values (length 2).
  3) The prox reduces to a proximal problem in the singular values:
         min_{t >= 0} 0.5||t - s||_2^2 + tau*( rw||t||_1 + (1-rw)||t||_2 )
     This is the sparse-group-lasso prox on s.
  4) The solution is known in closed form:
       - First apply soft-thresholding to singular values (L1 part)
       - Then apply vector shrinkage (L2 part)
     Reconstruct using the same U,V.

Notes:
  - Use broadcasting to avoid loops over pixels if you can.
  - Start with a loop implementation for clarity, then vectorize if time permits.
---------------------------------------------------------------------------

Run demo:
  python lab_deblur_hessian_mixed_skeleton.py --demo

"""

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Utilities -------------------------

def psf2otf(psf: np.ndarray, out_shape):
    psf = np.asarray(psf, dtype=np.float64)
    H, W = out_shape
    otf = np.zeros((H, W), dtype=np.float64)
    kh, kw = psf.shape
    otf[:kh, :kw] = psf

    cy = kh // 2
    cx = kw // 2
    otf = np.roll(otf, -cy, axis=0)
    otf = np.roll(otf, -cx, axis=1)

    return np.fft.fft2(otf)


def gaussian_kernel(size: int, sigma: float):
    ax = np.arange(size) - (size // 2)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def normalize01(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def rmse(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)))


# ------------------------- Hessian operator (periodic) -------------------------

def hessian_filters():
    dxx = np.array([[1.0, -2.0, 1.0]])
    dyy = dxx.T
    dxy = 0.25 * np.array([[1.0, 0.0, -1.0],
                           [0.0, 0.0,  0.0],
                           [-1.0, 0.0, 1.0]])
    return dxx, dyy, dxy


def apply_hessian_fft(x_fft, OTF_dxx, OTF_dyy, OTF_dxy):
    xx = np.fft.ifft2(OTF_dxx * x_fft).real
    yy = np.fft.ifft2(OTF_dyy * x_fft).real
    xy = np.fft.ifft2(OTF_dxy * x_fft).real

    H, W = xx.shape
    Hx = np.zeros((H, W, 2, 2), dtype=np.float64)
    Hx[..., 0, 0] = xx
    Hx[..., 1, 1] = yy
    Hx[..., 0, 1] = xy
    Hx[..., 1, 0] = xy
    return Hx


def hessian_adjoint_fft(Z_minus_U, OTF_dxx, OTF_dyy, OTF_dxy):
    Z11 = Z_minus_U[..., 0, 0]
    Z22 = Z_minus_U[..., 1, 1]
    Z12 = Z_minus_U[..., 0, 1]
    Z21 = Z_minus_U[..., 1, 0]

    F = (
        np.conj(OTF_dxx) * np.fft.fft2(Z11)
        + np.conj(OTF_dyy) * np.fft.fft2(Z22)
        + np.conj(OTF_dxy) * np.fft.fft2(Z12)
        + np.conj(OTF_dxy) * np.fft.fft2(Z21)
    )
    return F


# ------------------------- STUDENT TASK: Prox operator -------------------------

def prox_mixed_nuclear_frob(Z, tau, rw):
    """
    TODO (STUDENTS): implement the proximal mapping

        prox_{tau * [ rw*||.||_* + (1-rw)*||.||_F ]}(Z)

    where Z is a batch of 2x2 matrices with shape (...,2,2).

    REQUIRED behavior checks:
      - if tau <= 0: return Z unchanged
      - rw=1: should reduce to nuclear prox (singular value soft-threshold)
      - rw=0: should reduce to Frobenius prox (matrix shrink by Frobenius norm)
      - output shape == input shape

    Implementation hint:
      1) U, s, Vt = svd(Z)
         s has shape (...,2) (two singular values)
      2) Build t (shrunk singular values) by applying:
            (a) elementwise soft-threshold on s by tau*rw
            (b) vector shrink of the resulting 2-vector by tau*(1-rw)
      3) Return U diag(t) Vt.
         Use broadcasting: (U * t[...,None,:]) @ Vt
    """
    # --------- START OF STUDENT CODE AREA ---------
    # raise NotImplementedError("Students must implement prox_mixed_nuclear_frob")
    if tau <= 0:
        return Z
    
    alpha = tau*rw
    beta = tau*(1-rw)

    U,s,Vt = np.linalg.svd(Z)

    u = np.maximum(s-alpha, 0)
    norm_u = np.linalg.norm(u, axis = -1, keepdims= True)

    t = np.maximum((1-(beta/(norm_u+1e-12))), 0)*u

    Ut = U*t[...,None,:]
    Z = Ut @ Vt

    return Z

    # --------- END OF STUDENT CODE AREA ---------


# ------------------------- ADMM Solver -------------------------

@dataclass
class ADMMOptions:
    lam: float = 0.02
    rho: float = 2.0
    rw: float = 0.7
    max_iter: int = 200
    abs_tol: float = 1e-4
    rel_tol: float = 1e-3
    verbose: bool = True


def deblur_hessian_mixed_admm(b, psf, opts: ADMMOptions):
    b = np.asarray(b, dtype=np.float64)
    H, W = b.shape

    OTF_K = psf2otf(psf, (H, W))
    dxx, dyy, dxy = hessian_filters()
    OTF_dxx = psf2otf(dxx, (H, W))
    OTF_dyy = psf2otf(dyy, (H, W))
    OTF_dxy = psf2otf(dxy, (H, W))

    denom = (np.abs(OTF_K) ** 2
             + opts.rho * (np.abs(OTF_dxx) ** 2 + np.abs(OTF_dyy) ** 2 + 2.0 * (np.abs(OTF_dxy) ** 2))
             + 1e-12)

    Fb = np.fft.fft2(b)
    rhs_data = np.conj(OTF_K) * Fb

    x = b.copy()
    x_fft = np.fft.fft2(x)
    Z = apply_hessian_fft(x_fft, OTF_dxx, OTF_dyy, OTF_dxy)
    U = np.zeros_like(Z)

    lam, rho, rw = float(opts.lam), float(opts.rho), float(opts.rw)

    for it in range(1, opts.max_iter + 1):
        # ---- x-update (FFT closed-form solve) ----
        ZmU = Z - U
        rhs_reg = rho * hessian_adjoint_fft(ZmU, OTF_dxx, OTF_dyy, OTF_dxy)
        x_fft = (rhs_data + rhs_reg) / denom
        x = np.fft.ifft2(x_fft).real

        # ---- Hessian of x ----
        Hx = apply_hessian_fft(x_fft, OTF_dxx, OTF_dyy, OTF_dxy)

        # ---- Z-update (THIS CALL USES STUDENT-PROVIDED PROX) ----
        V = Hx + U
        Z_old = Z
        Z = prox_mixed_nuclear_frob(V, tau=lam / rho, rw=rw)

        # ---- dual update ----
        U = U + (Hx - Z)

        # ---- stopping criteria ----
        r = Hx - Z
        s = rho * (Z - Z_old)

        r_norm = float(np.sqrt(np.sum(r * r)))
        s_norm = float(np.sqrt(np.sum(s * s)))

        Hx_norm = float(np.sqrt(np.sum(Hx * Hx)))
        Z_norm = float(np.sqrt(np.sum(Z * Z)))
        U_norm = float(np.sqrt(np.sum(U * U)))

        eps_pri = math.sqrt(r.size) * opts.abs_tol + opts.rel_tol * max(Hx_norm, Z_norm)
        eps_dual = math.sqrt(U.size) * opts.abs_tol + opts.rel_tol * (rho * U_norm)

        if opts.verbose and (it == 1 or it % 10 == 0 or it == opts.max_iter):
            print(f"it {it:4d} | r {r_norm:.3e} (<= {eps_pri:.3e}) "
                  f"| s {s_norm:.3e} (<= {eps_dual:.3e}) | rho {rho:.2e} | rw {rw:.2f}")

        # if r_norm <= eps_pri and s_norm <= eps_dual:
        #     break

    return x


# ------------------------- Demo / Self-test scaffold -------------------------

# def synthetic_test(opts: ADMMOptions):
#     H, W = 256, 256

#     x0 = np.zeros((H, W), dtype=np.float64)
#     x0[40:216, 60:80] = 1.0
#     x0[120:140, 80:220] = 1.0
#     yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
#     x0 += 0.4 * np.exp(-((xx - 180) ** 2 + (yy - 70) ** 2) / (2 * 18 ** 2))
#     x0 = normalize01(x0)

#     psf = gaussian_kernel(size=17, sigma=3.0)

#     OTF_K = psf2otf(psf, (H, W))
#     Fx0 = np.fft.fft2(x0)
#     b = np.fft.ifft2(OTF_K * Fx0).real

#     rng = np.random.default_rng(0)
#     sigma_n = 0.01
#     b_noisy = np.clip(b + sigma_n * rng.standard_normal(b.shape), 0.0, 1.0)

#     xhat = deblur_hessian_mixed_admm(b_noisy, psf, opts)
#     xhat_clip = np.clip(xhat, 0.0, 1.0)

#     print("\nRMSE:")
#     print("  blurred/noisy vs truth :", rmse(b_noisy, x0))
#     print("  restored vs truth      :", rmse(xhat_clip, x0))

def real_test(opts: ADMMOptions):
    H, W = 256, 256

    x0 = plt.imread('Test_image_2.jpeg')
    x0 = x0[:,:,0]
    x0 = normalize01(x0)
    psf = gaussian_kernel(size=17, sigma=3.0)

    OTF_K = psf2otf(psf, (H, W))
    Fx0 = np.fft.fft2(x0)
    b = np.fft.ifft2(OTF_K * Fx0).real

    rng = np.random.default_rng(0)
    sigma_n = 0.01
    b_noisy = np.clip(b + sigma_n * rng.standard_normal(b.shape), 0.0, 1.0)

    xhat = deblur_hessian_mixed_admm(b_noisy, psf, opts)
    xhat_clip = np.clip(xhat, 0.0, 1.0)

    print("\nRMSE:")
    print("  blurred/noisy vs truth :", rmse(b_noisy, x0))
    print("  restored vs truth      :", rmse(xhat_clip, x0))

    plt.figure(figsize = (14,8))
    plt.subplot(1,2,1)
    plt.imshow(b_noisy, cmap = 'gray')
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(xhat_clip, cmap = 'gray')
    plt.title('Reconstructed Image')
    plt.savefig(f'Reconstructed_Image2_lambda{opts.lam}_rw{opts.rw}.png')
    plt.show()

    return rmse(xhat_clip, x0)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--lam", type=float, default=0.02)
#     ap.add_argument("--rho", type=float, default=2.0)
#     ap.add_argument("--rw", type=float, default=0.7)
#     ap.add_argument("--max_iter", type=int, default=200)
#     ap.add_argument("--quiet", action="store_true")
#     ap.add_argument("--demo", action="store_true")
#     args = ap.parse_args()

#     opts = ADMMOptions(
#         lam=args.lam,
#         rho=args.rho,
#         rw=args.rw,
#         max_iter=args.max_iter,
#         verbose=not args.quiet,
#     )

#     if args.demo:
#         synthetic_test(opts)
#         return

#     print("Run:\n"
#           "  python lab_deblur_hessian_mixed_skeleton.py --demo\n",
#           file=sys.stderr)

def run_real(laambda, rw):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=laambda)
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--rw", type=float, default=rw)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()

    opts = ADMMOptions(
        lam=args.lam,
        rho=args.rho,
        rw=args.rw,
        max_iter=args.max_iter,
        verbose=not args.quiet,
    )

    rmse = real_test(opts)
    return rmse


if __name__ == "__main__":
    rmse_track = []
    lam_space = [0.0001,0.005,0.01,0.015, 0.02]
    for lam in lam_space:
        rmse_track.append(run_real(lam, 1))
    print(f"The RMSEs are {rmse_track}")
    