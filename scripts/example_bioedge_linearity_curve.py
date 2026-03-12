# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:37:17 2025

@author: fleroux
"""

# %%

from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

from tqdm import tqdm

from aobench.storage.containers import LinearityMeasurements, GlobalResult
from aobench.storage.io import load_experiment

from aobench.config import Config

# %% Load configuration currently defined in config/config.py


config = Config()

# %%

dirc = Path(__file__).parent

# %% Import experimental linearity results

utc = "utc_2026-01-23_11-55-57"

path = config.get_timestamped_dir(
    config.paths.raw_linearity_measurements_dir,
    utc=utc,
) / config.get_timestamped_filename(
    config.output_filenames.raw_linearity_measures_filename,
    utc=utc,
)

linearity_experiment: GlobalResult = load_experiment(path, lazy=True)

linearity_mesurements: LinearityMeasurements = (
    linearity_experiment.linearity_measurements
)

# %% define functions


# %% Define simulations parameters


# initialize the dictionary
param = {}

# fill the dictionary
# ------------------ ATMOSPHERE ----------------- #

param["r0"] = config.parameters.turbulence.r0  # [m] value of r0 in the visibile
param["L0"] = config.parameters.turbulence.L0  # [m] value of L0 in the visibile
param["fractionnal_r0"] = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
param["wind_speed"] = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of  layers
param["wind_direction"] = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
param["altitude"] = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers
param["seeds"] = range(1)

# ------------------- TELESCOPE ------------------ #

param["diameter"] = 8  # [m] telescope diameter
param["n_subaperture"] = 20  # number of WFS subaperture along the
# telescope diameter
# [pixel] sampling of the WFS subapertures
param["n_pixel_per_subaperture"] = 8
# in telescope pupil space
param["resolution"] = (
    param["n_subaperture"] * param["n_pixel_per_subaperture"]
)  # resolution of the telescope driven by
# the WFS
param["size_subaperture"] = (
    param["diameter"] / param["n_subaperture"]
)  # [m] size of a subaperture projected in M1 space
param["sampling_time"] = 1 / 1000  # [s] loop sampling time
param["centralObstruction"] = 0  # central obstruction in percentage
# of the diameter

# ---------------------- NGS ---------------------- #

param["magnitude"] = 0  # magnitude of the guide star

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
param["optical_band"] = "R4"  # optical band of the guide star

# ------------------------ DM --------------------- #

param["n_actuator"] = 2 * param["n_subaperture"]  # number of actuators

# ----------------------- WFS ---------------------- #

param["modulation"] = 2.5  # [lambda/D] modulation radius or grey width
param["grey_length"] = param["modulation"]  # [lambda/D] grey length in case of
# small grey bioedge WFS
param["n_pix_separation"] = 10  # [pixel] separation ratio between the pupils
param["psf_centering"] = False  # centering of the FFT and of the mask on
# the 4 central pixels
param["light_threshold"] = 0.3  # light threshold to select the valid pixels
param["post_processing"] = "fullFrame"  # post-processing of the WFS signals
# ('slopesMaps' or 'fullFrame')
param["detector_photon_noise"] = False
param["detector_read_out_noise"] = 0.0  # e- RMS

# super resolution
param["sr_amplitude"] = 0.25  # [pixel] super resolution shifts amplitude

# [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)
param["pupil_shift_bioedge"] = [
    [
        param["sr_amplitude"],
        -param["sr_amplitude"],
        param["sr_amplitude"],
        -param["sr_amplitude"],
    ],
    [
        param["sr_amplitude"],
        -param["sr_amplitude"],
        -param["sr_amplitude"],
        param["sr_amplitude"],
    ],
]

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

param["modal_basis"] = "KL"
# [m] actuator stroke for interaction matrix computation
param["stroke"] = 1e-9
param["single_pass"] = False  # push-pull or push only for the calibration
param["compute_M2C_Folder"] = str(Path(__file__).parent)

# ----------------------- RECONSTRUCTION ------------------------ #

param["n_modes_to_show_lse"] = 310
param["n_modes_to_show_lse_sr"] = 900

# --------------------- FILENAME -------------------- #

# name of the system
param["filename"] = (
    "_"
    + param["optical_band"]
    + "_band_"
    + str(param["n_subaperture"])
    + "x"
    + str(param["n_subaperture"])
    + "_"
    + param["modal_basis"]
    + "_basis"
)

# %% Build objects

# % -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(
    resolution=param["resolution"],  # [pixel] resolution of
    # the telescope
    diameter=param["diameter"],
)  # [m] telescope diameter

# % -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(
    optBand=param["optical_band"],  # Source optical band
    # (see photometry.py)
    magnitude=param["magnitude"],
)  # Source Magnitude

# % -----------------------    ATMOSPHERE   ----------------------------

# coupling telescope and source is mandatory to generate Atmosphere object
ngs * tel

# create the Atmosphere object
atm = Atmosphere(
    telescope=tel,  # Telescope
    r0=param["r0"],  # Fried Parameter [m]
    L0=param["L0"],  # Outer Scale [m]
    # Cn2 Profile (percentage)
    fractionalR0=param["fractionnal_r0"],
    windSpeed=param["wind_speed"],  # [m.s-1] wind speed of layers
    # [degrees] wind direction
    windDirection=param["wind_direction"],
    # of layers
    altitude=param["altitude"],
)  # [m] altitude of layers

# %% -------------------------     DM   ----------------------------------

dm = DeformableMirror(tel, nSubap=param["n_actuator"])

# %% ------------------------- MODAL BASIS -------------------------------

if param["modal_basis"] == "KL":

    M2C_KL_full, HHt, PSD_atm, df = compute_M2C(
        telescope=tel,
        atmosphere=atm,
        deformableMirror=dm,
        param=param,
        nameFolder=param["compute_M2C_Folder"],
        remove_piston=False,
        HHtName="KL_covariance_matrix",
        baseName="KL_basis",
        mem_available=6.1e9,
        minimF=False,
        nmo=None,
        ortho_spm=True,
        SZ=np.int64(2 * tel.OPD.shape[0]),
        nZer=3,
        NDIVL=1,
        lim_inversion=1e-16,
        returnHHt_PSD_df=True,
    )

    M2C = M2C_KL_full[:, 1:]  # remove piston

    dm.coefs = np.zeros(dm.nValidAct)  # reset dm.OPD

elif param["modal_basis"] == "poke":
    M2C = np.identity(dm.nValidAct)

# %% ----------------------- Grey Bi-O-Edge ---------------------------- #

# grey bioedge - fullFrame
gbioedge_full_frame = BioEdge(
    nSubap=param["n_subaperture"],
    telescope=tel,
    modulation=0.0,
    grey_width=param["modulation"],
    lightRatio=param["light_threshold"],
    n_pix_separation=param["n_pix_separation"],
    postProcessing="fullFrame",
)

# %% Calibration

calib_full_frame = InteractionMatrix(
    ngs,
    tel,
    dm,
    gbioedge_full_frame,
    M2C=M2C,
    stroke=param["stroke"],
    single_pass=param["single_pass"],
    noise="off",
    display=True,
)

# %% LSE Reconstructor computation

reconstructor_lse_full_frame = np.linalg.pinv(
    calib_full_frame.D[:, : param["n_modes_to_show_lse"]]
)

# %% Linearity measurement - Parameters

n_modes = linearity_mesurements.modes_numbers
amplitudes_rad = np.asarray(linearity_mesurements.injected_amplitudes)
amplitudes_meter = config.parameters.ao.wavelength / (2 * np.pi) * amplitudes_rad

# %% Memory allocation

reconstructed_amplitudes_rad_full_frame = np.full(
    (
        n_modes.shape[0],
        amplitudes_meter.shape[0],
        reconstructor_lse_full_frame.shape[0],
    ),
    np.nan,
)
# %%

mode_index = 0
for n_mode in n_modes:

    amplitude_index = 0
    for amplitude_meter in tqdm(amplitudes_meter):
        tel.resetOPD()
        dm.coefs = amplitude_meter * M2C[:, n_mode]
        ngs * tel * dm * gbioedge_full_frame

        reconstructed_amplitudes_rad_full_frame[mode_index, amplitude_index, :] = (
            2
            * np.pi
            / ngs.wavelength
            * (reconstructor_lse_full_frame @ gbioedge_full_frame.signal)
        )
        amplitude_index += 1
    mode_index += 1

# %% Plots - Simulation

mode_index = 0
for n_mode in n_modes:

    plt.figure()

    plt.plot(
        amplitudes_rad,
        reconstructed_amplitudes_rad_full_frame[mode_index, :, n_mode],
        label="linearity curve full_frame",
    )

    plt.plot(
        amplitudes_rad,
        amplitudes_rad,
        label="y=x",
    )

    plt.title(f"Linearity curve KL mode {n_mode}")
    plt.xlabel("Input [rad]")
    plt.ylabel("Output [rad]")
    plt.legend()

    plt.title(f"Linearity curve KL mode {n_mode}")
    plt.xlabel("Input [nm]")
    plt.ylabel("Output [nm]")
    plt.legend()

    mode_index += 1

# %% Plots - Experiment VS simulation

mode_index = 0
for n_mode in n_modes:

    plt.figure()

    plt.plot(
        amplitudes_rad,
        reconstructed_amplitudes_rad_full_frame[mode_index, :, n_mode],
        "-*",
        label="linearity curve simulation",
    )

    plt.plot(
        amplitudes_rad,
        np.asarray(
            linearity_mesurements.reconstructed_amplitudes[mode_index, :, n_mode]
        ),
        "-+",
        label="linearity curve experiment",
    )

    plt.plot(
        amplitudes_rad,
        amplitudes_rad,
        "k",
        label="y=x",
    )

    plt.title(f"Linearity curve KL mode {n_mode}")
    plt.xlabel("Input [rad]")
    plt.ylabel("Output [rad]")
    plt.legend()

    mode_index += 1

# %% Plots - Experiment VS simulation - All modes on the same figure

plt.figure()

plt.plot(
    amplitudes_rad,
    amplitudes_rad,
    "k",
    label="y=x",
)

mode_index = 0
for n_mode in n_modes:

    (line_sim,) = plt.plot(
        amplitudes_rad,
        reconstructed_amplitudes_rad_full_frame[mode_index, :, n_mode],
        "--*",
        label=f"linearity curve simulation KL mode {n_mode}",
    )

    plt.plot(
        amplitudes_rad,
        np.asarray(
            linearity_mesurements.reconstructed_amplitudes[mode_index, :, n_mode]
        )
        - linearity_mesurements.reconstructed_amplitudes[mode_index, 23, n_mode],
        "-+",
        color=line_sim.get_color(),
        label=f"linearity curve experiment KL mode {n_mode}",
    )

    mode_index += 1

plt.title("Linearity curves - Bias corrected")
plt.xlabel("Input [rad]")
plt.ylabel("Output [rad]")
plt.legend()

# %%

experimental_interaction_matrix = np.asarray(
    linearity_experiment.interaction_matrix.raw_interaction_matrix
)[:, linearity_experiment.reconstructor.valid_pixels].T

experimental_sensitivities = np.diag(
    experimental_interaction_matrix.T @ experimental_interaction_matrix
)

# %% simulated sensibilities

simulated_sensibilities = np.diag(calib_full_frame.D.T @ calib_full_frame.D)

# %%

plt.figure()
plt.plot(experimental_sensitivities)

plt.figure()
plt.plot(simulated_sensibilities[:342])

# %%

plt.figure()
plt.plot(
    experimental_sensitivities / experimental_sensitivities.max(),
    "-",
    label="experimental_sensitivities",
)
plt.plot(
    simulated_sensibilities[:342] / simulated_sensibilities[:342].max(),
    "-",
    label="simulated_sensitivities",
)

plt.legend()


# %%

simulated_imat = np.zeros(
    (calib_full_frame.D.shape[1],) + gbioedge_full_frame.validSignal.shape
)

simulated_imat[:, gbioedge_full_frame.validSignal] = calib_full_frame.D.T

# %% Display Imat

n_modes = [0, 2, 10, 50, 150, 300]

fig, axs = plt.subplots(
    nrows=len(n_modes),
    ncols=3,
    figsize=(6, 1.8 * len(n_modes)),
)

# Tight horizontal spacing
fig.subplots_adjust(
    wspace=0.5,  # ← smaller = closer columns (try 0.01–0.05)
    hspace=0.15,
)


# Column titles
axs[0, 0].set_title("Calibration modes", fontsize=12, pad=12)
axs[0, 1].set_title("Simulated\nreduced intensities", fontsize=12, pad=12)
axs[0, 2].set_title("Experimental\nreduced intensities", fontsize=12, pad=12)

for row, n_mode in enumerate(n_modes):

    ax = axs[row, 0]
    ax.imshow(
        linearity_experiment.interaction_matrix.modal_basis[n_mode, ...],
        aspect="equal",  # preserve aspect ratio
    )
    ax.set_axis_off()

    # Row title (explicit text, survives axis off)
    ax.text(
        -0.15,
        0.5,
        f"KL {n_mode}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="medium",
    )

    ax = axs[row, 1]
    ax.imshow(
        np.flip(simulated_imat[n_mode, ...], axis=1).T,
        aspect="equal",  # preserve aspect ratio
    )
    ax.set_axis_off()

    ax = axs[row, 2]
    ax.imshow(
        linearity_experiment.interaction_matrix.raw_interaction_matrix[n_mode, ...],
        aspect="equal",  # preserve aspect ratio
    )
    ax.set_axis_off()

plt.show()


# %%

D_exp = linearity_experiment.interaction_matrix.raw_interaction_matrix[...][
    :, linearity_experiment.reconstructor.valid_pixels[...]
].T

U_exp, s_exp, VT_exp = np.linalg.svd(D_exp, full_matrices=False)

# %%

U_sim, s_sim, VT_sim = np.linalg.svd(
    calib_full_frame.D[:, : D_exp.shape[1]], full_matrices=False
)

# %%

cond_exp = s_exp.max() / s_exp.min()
cond_sim = s_sim.max() / s_sim.min()

plt.figure()
plt.plot(
    s_exp / s_exp.max(),
    label=f"experimental - cond = {cond_exp:.2f}",
)
plt.plot(
    s_sim / s_sim.max(),
    label=f"simulated - cond = {cond_sim:.2f}",
)
plt.yscale("log")
plt.xlabel("# eigen mode")
plt.ylabel("eigen value")
plt.legend(loc="lower left")
plt.title("SVD normalized by max")

plt.show()

# %%
