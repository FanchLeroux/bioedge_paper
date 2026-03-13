# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 09:28:01 2026

@author: fleroux
"""

# %%

import pathlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from tqdm import tqdm

from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.BioEdge import BioEdge
from OOPAO.Pyramid import Pyramid
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from OOPAO.MisRegistration import MisRegistration
from OOPAO.tools.displayTools import displayMap, display_wfs_signals, interactive_show

from aobench.simulation.oopao.closed_loop import close_the_loop
from aobench.super_resolution import bin_array
from aobench.pattern import get_circular_pupil
from aobench.analysis.sensitivity import (
    compute_photon_noise_sensitivity,
    compute_readout_noise_sensitivity,
)
from aobench.config import Config

config = Config()


# %% functions definitions


def check_wfs_pupils(valid_pixel_map, wfs, n_it=3, correct=False):
    from OOPAO.tools.tools import centroid

    plt.close("all")
    xs = wfs.sx
    ys = wfs.sy
    if correct is False:
        for i in range(4):
            I = wfs.grabFullQuadrant(i + 1, valid_pixel_map)
            xc = I.shape[0] // 2
            [x, y] = np.asarray(centroid(I, threshold=0.3))
            I_ = np.abs(wfs.grabFullQuadrant(i + 1))
            I_ /= I_.max()
            [x_, y_] = np.asarray(centroid(I_, threshold=0.3))

            plt.figure(1)
            plt.subplot(2, 2, i + 1)
            plt.imshow(I - I_)
            plt.plot(x, y, "+", markersize=20)
            plt.plot(x_, y_, "+", markersize=20)
            plt.axis("off")
            xs[i] += x - x_
            ys[i] += y_ - y
            plt.title("Quadrant #" + str(i))
            plt.draw()

    else:
        for i_it in range(n_it):
            wfs.apply_shift_wfs(sx=xs, sy=ys)

            for i in range(4):
                I = wfs.grabFullQuadrant(i + 1, valid_pixel_map)
                xc = I.shape[0] // 2
                [x, y] = np.asarray(centroid(I, threshold=0.3))
                I_ = np.abs(wfs.grabFullQuadrant(i + 1))
                I_ /= I_.max()
                [x_, y_] = np.asarray(centroid(I_, threshold=0.3))
                plt.figure(1)
                plt.subplot(n_it, 4, 4 * i_it + i + 1)
                plt.imshow(I - I_)
                plt.plot(x, y, "+", markersize=20)
                plt.plot(x_, y_, "+", markersize=20)
                plt.title(
                    "Step "
                    + str(i_it)
                    + " -- ["
                    + str(np.round(x - x_, 1))
                    + ","
                    + str(np.round(y - y_, 1))
                    + "]"
                )
                plt.axis("off")
                xs[i] += x - x_
                ys[i] += y_ - y
                plt.draw()
                plt.pause(0.0001)
        return


# %% Parameters

# optical setup
wavelength = 635e-9  # [m]
modulation = 2  # radius or half grey width [lambda/D]

# linearity tests
n_modes = [0, 2, 10, 50, 100, 300]
amplitudes_rad = np.linspace(-4, 4, 100)
amplitudes_meter = wavelength / (2 * np.pi) * amplitudes_rad

# %% Import closed loop data

dirc = (
    pathlib.Path(__file__).parents[1]
    / "data"
    / "20260312_bioedge_bench_results_for_jdd"
)

utc_closed_loop = "utc_2026-03-12_10-22-24"
utc_closed_loop = "utc_2026-03-12_13-57-13"
utc_closed_loop = "utc_2026-03-12_14-16-36"
utc_closed_loop = "utc_2026-03-12_14-21-11"

closed_loop_filename = utc_closed_loop + "_closed_loop.h5"

# %%

with h5py.File(dirc / "closed_loop" / closed_loop_filename) as f:
    closed_loop_grp = f["closed_loop_grp"]
    reconstructor_grp = f["reconstructor_grp"]
    interaction_matrix_grp = reconstructor_grp["interaction_matrix_grp"]
    interaction_matrix_exp = interaction_matrix_grp["interaction_matrix"][...]
    strokes_exp = interaction_matrix_grp["interaction_matrix"].attrs["strokes"]
    valid_pixels = interaction_matrix_grp["valid_pixels"][...]
    modal_basis = interaction_matrix_grp["modal_basis"][...]
    turbulence = closed_loop_grp["turbulence"][...] * 2 * np.pi / 255  # [rad]
    wfs_frames_exp = closed_loop_grp["wfs_frames"][...]
    total_exp = closed_loop_grp["total"][...]
    residual_exp = closed_loop_grp["residual"][...]
    frequency = closed_loop_grp["turbulence"].attrs["frequency"]
    r0 = closed_loop_grp["turbulence"].attrs["r0"]
    delay = closed_loop_grp.attrs["delay"]
    loop_gain = closed_loop_grp.attrs["gain"][()]
    reference_intensities = interaction_matrix_grp["reference_intensities"][...]
    focal_plane_images = closed_loop_grp["focal_plane_images"][...]
    focal_plane_images_open_loop = f["open_loop_grp"]["focal_plane_images"][...]
    reference_psf = f["closed_loop_grp"]["reference_psf"][...]

# reference_psf = np.load("reference_psf.npy")

# %% Plot focal plane images

linthresh = 1e-1

vmin = min(
    focal_plane_images.min(), focal_plane_images_open_loop.min(), reference_psf.min()
)
vmax = max(
    focal_plane_images.max(), focal_plane_images_open_loop.max(), reference_psf.max()
)

nrows, ncols, figsize = 1, 3, 5

axs: tuple[plt.Axes, ...]
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(figsize * ncols, figsize * nrows),
    constrained_layout=True,
)
axs[0].imshow(
    focal_plane_images_open_loop.mean(axis=0),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
)
axs[0].set_title("Open loop focal plane image")
axs[1].imshow(
    focal_plane_images.mean(axis=0),
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
)
axs[1].set_title("Closed loop focal plane image")
im = axs[2].imshow(
    reference_psf,
    norm=mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
    cmap="inferno",
)
axs[2].set_title("Reference PSF")
fig.colorbar(im, ax=axs, aspect=60, shrink=0.91, location="bottom").ax.tick_params(
    axis="x", direction="out", size=2
)

fig.suptitle(f"{utc_closed_loop}")

fig.savefig(utc_closed_loop + "_focal_plane_images_comparison.pdf")

# %% Compute sensitivities

photon_noise_sensitivity_exp = compute_photon_noise_sensitivity(
    interaction_matrix_exp, reference_intensities
)
readout_noise_sensitivity_exp = compute_readout_noise_sensitivity(
    interaction_matrix_exp,
    n_subapertures=interaction_matrix_exp.shape[0] // 4,  # 4 pupils
)

# %% Plot sensitivities

nrows, ncols, figsize = 1, 2, 5
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(figsize * ncols, figsize * nrows)
)
axs = np.atleast_1d(axs).ravel()

axs[0].plot(photon_noise_sensitivity_exp, label="exp")
axs[0].axhline(2**0.5, color="k", linestyle="--", label="$\sqrt{2}$")
axs[0].set_xlabel("Mode")
axs[0].set_ylabel("$S_{\gamma}$")
axs[0].set_title("Photon noise Sensitivity")
axs[0].legend()

axs[1].plot(readout_noise_sensitivity_exp, label="exp")
axs[1].set_xlabel("Mode")
axs[1].set_ylabel("$S_{RON}$")
axs[1].set_title("Readout noise Sensitivity")
axs[1].legend()

fig.suptitle(f"{utc_closed_loop}")
fig.savefig(utc_closed_loop + "_sensitivities.pdf")

# %% check modal basis std

pupil = get_circular_pupil(modal_basis.shape[1])
print((modal_basis[:, pupil].std(axis=1)))

# %% Extract simualtions parameters

# if valid_pixels.shape[0] < valid_pixels.shape[1]:
#     npx = (valid_pixels.shape[1] - valid_pixels.shape[0]) // 2
#     suplement = (valid_pixels.shape[1] - valid_pixels.shape[0]) % 2
#     valid_pixels_sim = np.pad(valid_pixels, ((npx, npx + suplement), (0, 0)))

# if valid_pixels.shape[0] > valid_pixels.shape[1]:
#     npx = (valid_pixels.shape[0] - valid_pixels.shape[1]) // 2
#     suplement = (valid_pixels.shape[0] - valid_pixels.shape[1]) % 2
#     valid_pixels_sim = np.pad(valid_pixels, ((0, 0), (npx, npx + suplement)))

telescope_resolution = modal_basis.shape[1]
wfs_resolution = (
    valid_pixels.sum(axis=0).max() // 2
)  # 2 pupils along each axis of the 2D plane
wfs_resolution += wfs_resolution % 2  # force to be even

wfs_sim_cam_resolution = wfs_resolution * 4

if valid_pixels.shape[0] < wfs_sim_cam_resolution:
    npx = (wfs_sim_cam_resolution - valid_pixels.shape[0]) // 2
    suplement = (wfs_sim_cam_resolution - valid_pixels.shape[0]) % 2
    valid_pixels_sim = np.pad(valid_pixels, ((npx, npx + suplement), (0, 0)))

if valid_pixels.shape[1] < wfs_sim_cam_resolution:
    npx = (wfs_sim_cam_resolution - valid_pixels.shape[1]) // 2
    suplement = (wfs_sim_cam_resolution - valid_pixels.shape[1]) % 2
    valid_pixels_sim = np.pad(valid_pixels_sim, ((0, 0), (npx, npx + suplement)))

# %% build numerical twin

src = Source("R", magnitude=0)

# consider a slightly padded pupil (avoid edge effects for the numerical twin)
n_pixel_padded = 3
tel = Telescope(resolution=wfs_resolution - 2 * n_pixel_padded, diameter=2)

tel.pad(n_pixel_padded)

# apply mis-registrations to the modal DM
mis_reg = MisRegistration()
mis_reg.rotationAngle = 270 + 0.5
mis_reg.radialScaling = -0.05
mis_reg.tangentialScaling = -0.05
mis_reg.shiftX = 0.0 * tel.D / wfs_resolution
mis_reg.shiftY = 0 * tel.D / wfs_resolution

# interpolate and flip the data
modal_basis_sim = interpolate_cube(
    modal_basis,
    pixel_size_in=tel.D / telescope_resolution,
    pixel_size_out=tel.D / wfs_resolution,
    resolution_out=wfs_resolution,
    shape_out=[wfs_resolution, wfs_resolution],
    mis_registration=mis_reg,
    fliplr=True,
    flipud=False,
)

turbulence_sim = interpolate_cube(
    turbulence,
    pixel_size_in=tel.D / telescope_resolution,
    pixel_size_out=tel.D / wfs_resolution,
    resolution_out=wfs_resolution,
    shape_out=[wfs_resolution, wfs_resolution],
    mis_registration=mis_reg,
    fliplr=True,
    flipud=False,
)

src * tel
dm = DeformableMirror(
    tel,
    nSubap=modal_basis_sim.shape[1],
    modes=modal_basis_sim.reshape((modal_basis_sim.shape[0], -1)).T,
)

# check the modes
displayMap(dm.modes, axis=1, norma=True)


# %% Bi- O Edge numerical twin

wfs = BioEdge(
    nSubap=wfs_resolution,
    telescope=tel,
    modulation=0,
    lightRatio=0,  # no need to use it since we input the valid pixel map
    n_pix_edge=0,
    n_pix_separation=0,
    postProcessing="fullFrame_sum_flux",  # normalisation based on the flux on the detector (same as what is done on the bench)
    grey_width=modulation,
    userValidSignal=valid_pixels_sim,  # input user-define valid pixel map
    quadrants_numbering=[
        0,
        1,
        3,
        2,
    ],  # new feature to re-order the quadrants to match the experimental display
)

check_wfs_pupils(valid_pixels_sim, wfs, correct=True, n_it=2)

# %% compute simulated interaction matrix - bioedge

stroke_rad = 0.1  # [rad]
stroke_m = stroke_rad * src.wavelength / (2 * np.pi)  # [modal_basis_sim]

sim_calib = InteractionMatrix(
    ngs=src,
    tel=tel,
    wfs=wfs,
    dm=dm,
    M2C=np.identity(
        dm.coefs.shape[0]
    ),  # full M2C => identity matrix since we consider a modal DM
    stroke=stroke_m,
    invert=False,
    display=True,
    single_pass=False,
)

interaction_matrix_sim = sim_calib.D * src.wavelength / (2 * np.pi)

# %% visual comparison

modes = [0, 1, 2, 5, 50, 150, 341]

support_exp = np.full(valid_pixels.shape, np.nan)
support_sim = np.full(valid_pixels_sim.shape, np.nan)

empty_rows_exp = ~np.any(valid_pixels, axis=1)
empty_cols_exp = ~np.any(valid_pixels, axis=0)
empty_rows_sim = ~np.any(valid_pixels_sim, axis=1)
empty_cols_sim = ~np.any(valid_pixels_sim, axis=0)

fig, axs = plt.subplots(
    nrows=len(modes),
    ncols=3,  # mode label + exp + sim
    figsize=(5, 2.3 * len(modes)),
    constrained_layout=True,
)

# ---- column titles (once) ----
axs[0, 1].set_title("experimental\n", fontsize=11)
axs[0, 2].set_title("simulation\n", fontsize=11)

for i, mode in enumerate(modes):

    support_exp[valid_pixels] = interaction_matrix_exp[:, mode]
    support_sim[valid_pixels_sim] = interaction_matrix_sim[:, mode]

    vmin = support_exp[valid_pixels].min()
    vmax = support_exp[valid_pixels].max()

    # ---- mode index (text only) ----
    axs[i, 0].axis("off")
    axs[i, 0].text(
        0.5,
        0.5,
        f"KL mode {mode}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=axs[i, 0].transAxes,
    )

    # ---- experimental ----
    im_exp = axs[i, 1].imshow(
        np.delete(
            np.delete(support_exp, empty_rows_exp, axis=0), empty_cols_exp, axis=1
        ),
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 1].axis("off")

    # ---- simulation ----
    im_sim = axs[i, 2].imshow(
        np.delete(
            np.delete(support_sim, empty_rows_sim, axis=0), empty_cols_sim, axis=1
        ),
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 2].axis("off")

    fig.colorbar(
        im_exp, ax=[axs[i, 1], axs[i, 2]], aspect=30, shrink=0.95, location="bottom"
    ).ax.tick_params(axis="x", direction="out", size=2)

fig.suptitle(
    f"{utc_closed_loop}\n\nComparison of simulated and experimental\ninteraction matrices\n",
    fontsize=12,
    fontweight="bold",
)

plt.savefig(
    f"{utc_closed_loop}_interaction_matrix_visual.pdf",
)
plt.show()


# %% SVD

u_exp, s_exp, vt_exp = np.linalg.svd(interaction_matrix_exp, full_matrices=False)
u_sim, s_sim, vt_sim = np.linalg.svd(interaction_matrix_sim, full_matrices=False)

# %%

plt.figure()
plt.plot(s_exp, label="experimental")
plt.plot(s_sim, label="simulated")
plt.yscale("log")
plt.legend()
plt.xlabel("# eigen mode")
plt.ylabel("eigenvalue")
plt.title(f"{utc_closed_loop}\nInteraction matrix SVD")
plt.show()


# %% closed loop simulation

reconstructor_sim = np.linalg.pinv(sim_calib.D)
dm.coefs = 0
closed_loop_sim = close_the_loop(
    src,
    tel,
    dm,
    wfs,
    modal_basis=modal_basis_sim,
    turbulence_phase_screens=turbulence_sim,
    valid_pixels=valid_pixels_sim,
    reconstructor=reconstructor_sim,
    loop_gain=loop_gain,
    delay=delay,
    n_iter=total_exp.shape[0],
    reference_intensities=wfs.referenceSignal,
    display=False,
)

# %% figures closed loop

plt.figure()
plt.plot(total_exp, label="exp_total")
plt.plot(residual_exp, label="exp_residual")
plt.plot(closed_loop_sim.total, label="sim_total")
plt.plot(closed_loop_sim.residual, label="sim_residual")
plt.legend()
plt.xlabel("# iter")
plt.ylabel("residual phase std [rad]")
plt.title(
    utc_closed_loop + f"\nclosed loop with Bi-O-Edge\n"
    f"r0: {(100*r0):.0f} cm - gain: {loop_gain:.1f} - frequency: {frequency:.0f} Hz"
)

plt.figure()
plt.plot(np.exp(-(residual_exp**2)), label="experimental")
plt.plot(np.exp(-((closed_loop_sim.residual) ** 2)), label="simulation")
plt.legend()
plt.xlabel("# iter")
plt.ylabel("Strehl Ratio (Marechal)")
plt.title(
    utc_closed_loop + f"\nclosed loop with Bi-O-Edge\n"
    f"r0: {(100*r0):.0f} cm - gain: {loop_gain:.1f} - frequency: {frequency:.0f} Hz"
)
plt.show()

# %% linearity tests

# Simulation Bi-O-Edge

reconstructed_amplitudes_rad_bioedge_sim = np.full(
    (
        len(n_modes),
        amplitudes_meter.shape[0],
        reconstructor_sim.shape[0],
    ),
    np.nan,
)


reconstructed_amplitudes_rad_pyramid = np.full(
    (
        len(n_modes),
        amplitudes_meter.shape[0],
        reconstructor_sim.shape[0],
    ),
    np.nan,
)

for mode_index, n_mode in enumerate(n_modes):
    for amplitude_index, amplitude_meter in tqdm(enumerate(amplitudes_meter)):
        coefs = np.zeros(dm.nValidAct)
        coefs[n_mode] = amplitude_meter
        dm.coefs = coefs
        src**tel * dm * wfs
        src**tel * dm * pyramid
        reconstructed_amplitudes_rad_bioedge_sim[mode_index, amplitude_index, :] = (
            2 * np.pi / src.wavelength * (reconstructor_sim @ wfs.signal)
        )
        reconstructed_amplitudes_rad_pyramid[mode_index, amplitude_index, :] = (
            2 * np.pi / src.wavelength * (reconstructor_pyramid @ pyramid.signal)
        )

# %%

import math

# Define grid size
n_cols = 3  # you can change this
n_rows = math.ceil(len(n_modes) / n_cols)

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows))
axs = axs.flatten()  # flatten to 1D array for easy indexing

for mode_index, n_mode in enumerate(n_modes):
    axs[mode_index].plot(
        amplitudes_rad,
        reconstructed_amplitudes_rad_bioedge_sim[mode_index, :, n_mode],
        label="linearity curve bioedge sim",
    )
    axs[mode_index].plot(
        amplitudes_rad,
        reconstructed_amplitudes_rad_pyramid[mode_index, :, n_mode],
        label="linearity curve pyramid sim",
    )
    axs[mode_index].plot(
        amplitudes_rad,
        amplitudes_rad,
        label="y=x",
    )

    axs[mode_index].set_title(f"Linearity curve KL mode {n_mode}")
    axs[mode_index].set_xlabel("Input [rad]")
    axs[mode_index].set_ylabel("Output [rad]")
    axs[mode_index].legend()

# Turn off any unused subplots
for ax in axs[len(n_modes) :]:
    ax.axis("off")

plt.tight_layout()
plt.show()


# %%
