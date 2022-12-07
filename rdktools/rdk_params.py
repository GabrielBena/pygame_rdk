import numpy as np
from dataclasses import dataclass


@dataclass(repr=True)
class Params:

    # basic parameters
    COL_BLACK: tuple = (0, 0, 0)
    COL_WHITE: tuple = (255, 255, 255)

    WINDOW_WIDTH: int = 350
    WINDOW_HEIGHT: int = WINDOW_WIDTH
    APERTURE_RADIUS: int = WINDOW_HEIGHT // 2
    WINDOW_NAME: str = "Random Dot Kinematogram"
    WINDOW_COLOUR: tuple = COL_BLACK

    # time:
    TICK_RATE: int = 60
    TIME_FIX: int = 20  # frames
    TIME_ISI: int = 30
    TIME_RDK: int = 300
    TIME_ITI: int = 10

    # dot parameters
    N_DOTS: int = (
        int(np.pi * APERTURE_RADIUS**2) // 250
    )  # max num of simultaneously displayed dots
    DOT_SIZE: int = 3  # size in pixels
    DOT_SPEED: int = 4  # speed in pixels per frame
    DOT_ANGLES: int = None
    N_TRIALS: int = 5
    # : int DOT_ANGLES = list(np.arange(-180, 180, 10))  # motion directions
    DOT_REPETITIONS: int = 1  # how many repetitions of same trials?

    # DOT_COHERENCE = [0.3]  # motion coherence of all dots except subset (between 0 and 1)
    # DOT_COHERENCE.append(1 - 2 * DOT_COHERENCE[0])
    # SUBSET_RATIO = DOT_COHERENCE[0]  # ratio of dots moving temporally coherently

    DOT_COHERENCE: tuple = (0.2, 0.8)
    SUBSET_RATIO: float = 0.1

    DOT_COLOR: tuple = COL_WHITE

    # aperture parameters
    APERTURE_WIDTH: int = 4  # line width in pixels
    APERTURE_COLOR: int = COL_WHITE

    # fixation parameters
    FIX_SIZE: tuple = (5, 5)  # width and height of fix cross
    FIX_COLOR: tuple = COL_WHITE
    FIX_WIDTH: int = 2  # line width
