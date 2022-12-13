import numpy as np
from dataclasses import dataclass


@dataclass(repr=True)
class Params:

    NAME: str = "gabriel"

    # window
    WINDOW_WIDTH: int = 300
    WINDOW_HEIGHT: int = WINDOW_WIDTH
    APERTURE_RADIUS: int = WINDOW_HEIGHT // 2
    WINDOW_NAME: str = "Random Dot Kinematogram"

    # time:
    TICK_RATE: int = 60  # frames
    TIME_FIX: int = 20
    TIME_ISI: int = 30
    TIME_RDK: int = 300  # stimulus length
    TIME_ITI: int = 10

    # dot parameters
    N_DOTS: int = (
        int(np.pi * APERTURE_RADIUS**2) // 250
    )  # max num of simultaneously displayed dots
    DOT_SIZE: int = 3  # size in pixels
    DOT_SPEED: int = 4  # speed in pixels per frame
    DOT_ANGLES: None = None
    N_TRIALS_PER_BATCH: int = 5
    N_BATCH: int = 5
    DOT_REPETITIONS: int = 1  # how many repetitions of same trials?
    # : int DOT_ANGLES = list(np.arange(-180, 180, 10))  # motion directions

    # DOT_COHERENCE = [0.3]  # motion coherence of all dots except subset (between 0 and 1)
    # DOT_COHERENCE.append(1 - 2 * DOT_COHERENCE[0])
    # SUBSET_RATIO = DOT_COHERENCE[0]  # ratio of dots moving temporally coherently

    # DOT_COHERENCE: tuple = ((0.1, 0.5), (0.5, 0.9))  # coherences or range of coherences
    # SUBSET_RATIO: float = (0.01, 0.3)  # ratio or range of ratio

    DOT_COHERENCE: tuple = ((0.1, 0.2, 0.3, 0.4, 0.5), (0.5, 0.6, 0.7, 0.8, 0.9))
    SUBSET_RATIO: tuple = (0.01, 0.05, 0.1, 0.2, 0.3)
    TEMPORALLY_COHERENT: bool = True

    COMOD: bool = False
    N_ANGLES: int = 2

    # Diffusion angle parameter
    DIFFUSION_SCALE: int = 0
    DIFFUSE_SUBSET: bool = False

    # aperture parameters
    APERTURE_WIDTH: int = 4  # line width in pixels

    # fixation parameters
    FIX_SIZE: tuple = (5, 5)  # width and height of fix cross
    FIX_WIDTH: int = 2  # line width

    # colors
    COL_BLACK: tuple = (0, 0, 0)
    COL_WHITE: tuple = (255, 255, 255)
    COL_BLUE: tuple = (30, 144, 255)
    COL_RED: tuple = (178, 34, 34)
    DOT_COLOR: tuple = COL_WHITE
    WINDOW_COLOUR: tuple = COL_BLACK
    APERTURE_COLOR: tuple = COL_WHITE
    FIX_COLOR: tuple = COL_WHITE


def get_random_params(params):

    coherences = []
    for c in params.DOT_COHERENCE:
        r = np.random.rand()
        try:
            iter(c)
            if len(c) == 2:
                coherences.append((c[1] - c[0]) * r + c[0])
            else:
                coherences.append(np.random.choice(c))
        except TypeError:
            coherences.append(r * c)

    s = params.SUBSET_RATIO
    r = np.random.rand()
    try:
        iter(s)
        if len(s) == 2:
            subset_ratio = (s[1] - s[0]) * r + s[0]
        else:
            subset_ratio = np.random.choice(s)
    except TypeError:
        subset_ratio = r * s

    return coherences, subset_ratio
