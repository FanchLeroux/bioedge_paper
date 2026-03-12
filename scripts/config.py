# %%

from dataclasses import dataclass, field

from pathlib import Path

import numpy as np


@dataclass
class Paths:

    root_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root_dir / "data"

    sdk_dir: Path = root_dir / "src" / "aobench" / "third_party" / "sdk"
    slm_sdk_dir: Path = sdk_dir / "meadowlark" / "slm1920"
    thorcam_sdk_dir: Path = sdk_dir / "thorlabs" / "zelux_camera"

    slm_lut_dir: Path = data_dir / "slm" / "lut"
    slm_wfc_dir: Path = data_dir / "slm" / "wfc"

    modal_basis_dir: Path = data_dir / "modal_basis"
    turbulence_dir: Path = data_dir / "turbulence"

    lut_measurements_dir: Path = data_dir / "slm" / "lut"
    dark_dir: Path = data_dir / "dark"
    interaction_matrix_dir: Path = data_dir / "interaction_matrix"
    reconstructor_dir: Path = data_dir / "reconstructor"
    linearity_dir: Path = data_dir / "linearity"
    closed_loop_dir: Path = data_dir / "closed_loop"
    measure_mask_dir: Path = data_dir / "measure_mask"


############## Hardware ##############


@dataclass
class Source:
    wavelength: float = 635e-9  # [m]


@dataclass
class Slm:

    lut_filename: str = "slm_at_635.lut"
    wfc_filename: str | None = "utc_2026-03-12_10-40-43_slm_wfc.h5"

    center: np.ndarray = field(
        default_factory=lambda: np.array((500, 1055))
    )  # [slm px]
    pupil_diameter: int | None = 660  # 800  # [slm px]
    tilt_offset_amplitude: float | None = None  # [rad RMS]
    tilt_offset_angle: float | None = 0.0  # [deg]
    sleep_time: float = 1.0  # [s]


@dataclass
class Orca:
    exposure_time: float = 5e-3  # [s]
    roi: tuple[int, int, int, int] = field(
        default_factory=lambda: (410, 1970, 310, 1650)
    )  # full sensor is (0, 2048, 0, 2048) | (x0, x1, y0, y1)
    binning: int = 4  # max 4
    serial_number: str = "S/N: 002369"
    n_frames_avg_dark: int = 500
    dark_filename: str | None = "utc_2026-03-12_13-27-12_dark_orca.h5"
    x: int | None = None
    y: int | None = None


@dataclass
class Thorcam:
    serial_number: str | None = None
    exposure_time: float = 1e-3  # [s]
    roi: tuple[int, int, int, int] | None = field(
        default_factory=lambda: (600, 400, 1000, 800)
    )  # full sensor is (0, 1440, 0, 1080) | (x0, y0, x1, y1)
    binning: int = 1  # max 16 | print(thorcam.camera.biny_range)
    n_frames_avg_dark: int = Orca().n_frames_avg_dark
    dark_filename: str | None = "utc_2026-03-12_13-27-25_dark_thorcam.h5"


@dataclass
class Hardware:
    source: Source = field(default_factory=Source)
    slm: Slm = field(default_factory=Slm)
    orca: Orca = field(default_factory=Orca)
    thorcam: Thorcam = field(default_factory=Thorcam)


########################################################################################


@dataclass
class ModalBasis:
    filename: str = "KL_modal_basis.h5"  # filename for saving
    n_subapertures: int = 20
    n_pixels_edge: int = (
        10  # number of extra pixels to compute high resolution modes and get sharp edges
    )


@dataclass
class Turbulence:
    filename: str = "turbulence.h5"  # filename for saving
    seed: int = 5
    r0: float = 0.05  # [m]
    L0: float = 30.0  # [m]
    frequency: float = 500.0  # [Hz]
    n_screens: int = 500
    pupil_diameter: float = 2.0  # [m]
    n_pixels_edge: int = 10
    fractionnal_r0: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.1, 0.1, 0.25, 0.1])
    )
    wind_speed: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                5.0,
                4.0,
                8.0,
                10.0,
                2.0,
            ]
        )
    )  # [m.s-1] wind speed of  layers
    wind_direction: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0,
                72,
                144,
                216,
                288,
            ]
        )
    )  # [degrees] wind direction of layers
    altitude: np.ndarray = field(
        default_factory=lambda: np.array([0, 1000, 5000, 10000, 12000])
    )  # [m] altitude of layers


@dataclass
class InteractionMatrix:
    filename: str = "interaction_matrix.h5"  # filename for saving
    modal_basis_filename: str = "utc_2026-03-11_15-44-39_KL_modal_basis.h5"
    n_modes: int = 342
    modes_limits: np.ndarray = field(
        default_factory=lambda: np.array([2, 25])
    )  # define low, middl and high order modes to adapt stroke and n_frames_avg
    strokes: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.1, 0.05])
    )  # [rad]
    n_frames_avg: np.ndarray = field(default_factory=lambda: np.array([50, 20, 10]))


@dataclass
class Reconstructor:
    filename: str = "reconstructor.h5"
    interaction_matrix_filename: str = "utc_2026-03-12_13-36-14_interaction_matrix.h5"
    n_modes: int = 342


@dataclass
class FlattenWavefront:
    filename: str = "slm_wfc.h5"
    number_of_modes: tuple = field(default_factory=lambda: (3, 25, 50, 150, 342))


@dataclass
class AOLoop:
    filename: str = "closed_loop.h5"
    reconstructor_filename: str = "utc_2026-03-12_13-37-21_reconstructor.h5"
    turbulence_filename: str = "utc_2026-03-12_14-27-46_turbulence.h5"
    n_controlled_modes: int = 342
    gain: float = 0.9
    leaky_gain: float = 1.0
    delay: int = 2  # [frames]
    n_iter: int = 100
    n_frames_avg: int = 1


@dataclass
class Linearity:

    filename: str = "linearity.h5"

    reconstructor_filename: str = AOLoop().reconstructor_filename

    n_frames_avg: np.ndarray = field(default_factory=lambda: np.array([50, 20, 10]))

    modes_limits: np.ndarray = field(default_factory=lambda: np.array([2, 25]))

    modes_numbers: np.ndarray = field(
        default_factory=lambda: np.array([0, 1, 2, 10, 50, 150, 300])
    )

    amplitudes_rad: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.0,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
            ]
        )
    )

    strokes: np.ndarray = field(init=False)

    def __post_init__(self):
        self.strokes = np.concatenate(
            (-np.flip(self.amplitudes_rad), self.amplitudes_rad[1:])
        )


@dataclass
class MeasureMask:
    filename: str = "grey_width_measure.h5"
    amplitudes_rad: np.ndarray = field(
        default_factory=lambda: 7
        * np.sign(np.linspace(-1, 1, 20))
        * np.linspace(-1, 1, 20) ** 2
    )
    tilt_angle = 0  # [deg]


@dataclass
class SuperResolutions:
    shifts: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [
                    0.25,
                    -0.25,
                    0.25,
                    -0.25,
                ],
                [
                    0.25,
                    -0.25,
                    -0.25,
                    0.25,
                ],
            ]
        )
    )


@dataclass
class Parameters:

    modal_basis: ModalBasis = field(default_factory=ModalBasis)
    turbulence: Turbulence = field(default_factory=Turbulence)
    hardware: Hardware = field(default_factory=Hardware)
    interaction_matrix: InteractionMatrix = field(default_factory=InteractionMatrix)
    reconstructor: Reconstructor = field(default_factory=Reconstructor)
    flatten_wavefront: FlattenWavefront = field(default_factory=FlattenWavefront)
    linearity: Linearity = field(default_factory=Linearity)
    ao_loop: AOLoop = field(default_factory=AOLoop)
    measure_mask: MeasureMask = field(default_factory=MeasureMask)


@dataclass
class Config:

    paths: Paths = field(default_factory=Paths)
    parameters: Parameters = field(default_factory=Parameters)
