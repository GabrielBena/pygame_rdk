# random dot motion kinematogram
# timo flesch, 2019

import pygame
from pygame.locals import *
import pygame_gui
from datetime import datetime

import random
import numpy as np
from rdktools.rdk_params import Params
import pandas as pd

from rdktools.rdk_stimuli import RDK, Fixation, BlankScreen, ResultPrompt
from PIL import Image
import os, shutil
import dataclasses
import yaml


def other_angle(angle):
    other = np.random.randint(0, 360)
    if np.abs(other - angle) > 10:
        return other
    else:
        return other_angle(angle)


def set_trials(n_reps=10, angles=[0, 90, 135], n_trials=None, shuff=True):
    """creates vector of all motion directions"""
    all_trials = []
    assert not (
        (angles is None) and (n_trials is None)
    ), "Provide angles or n_trials for random draw"

    if angles is None:
        angles = np.random.randint(0, 360, n_trials)

    other_angles = map(other_angle, angles)

    for subset_angle, global_angle in zip(angles, other_angles):

        all_trials.append([[global_angle, subset_angle] for _ in range(n_reps)])

    all_trials = np.concatenate(all_trials, axis=0)
    if shuff:
        idxs = np.arange(len(all_trials))
        random.shuffle(idxs)
        all_trials = all_trials[idxs]

    return all_trials


class TrialSequence(object):
    """defines the sequence of events within a trial.
    returns a n-D matrix with all displayed frames as greyscale images (2D)
    """

    def __init__(self, params=Params()):

        pygame.init()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(
            (params.WINDOW_WIDTH, params.WINDOW_HEIGHT)
        )
        pygame.display.set_caption(params.WINDOW_NAME)
        self.display.fill(params.COL_BLACK)
        self.rdk = RDK(self.display, params)
        self.fix = Fixation(self.display, params)
        self.iti = BlankScreen(self.display, params)
        self.result_screen = ResultPrompt(self.display, self.fix.centre)

    def run(self, angles=(90, 0)):
        frames_fix = self.fix.show()
        self.rdk.new_sample(angles)
        frames_rdk, chosen_angle = self.rdk.show()
        frames_iti = self.iti.show()

        if chosen_angle is None:
            chosen_angle = self.result_screen.show()

        return (
            np.concatenate((frames_fix, frames_rdk, frames_iti), axis=0),
            chosen_angle,
        )


class Experiment(object):
    def __init__(self, save_gif=True, save_data=True, params=Params()) -> None:

        self.trials = set_trials(
            n_reps=params.DOT_REPETITIONS,
            angles=params.DOT_ANGLES,
            n_trials=params.N_TRIALS,
        )

        self.trial_seq = TrialSequence(params)

        self.save_gif = save_gif
        self.save_data = save_data

        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y-%H-%M")

        try:
            date_time = now.strftime("%d-%m-%Y-%H-%M")
            self.save_path = f"experiments/{date_time}"
            os.mkdir(self.save_path)
        except FileExistsError:
            date_time = now.strftime("%d-%m-%Y-%H-%M-%S")
            self.save_path = f"experiments/{date_time}"
            os.mkdir(self.save_path)

        params_dict = dataclasses.asdict(params)
        with open(f"{self.save_path}/params.yml", "w") as out_file:
            yaml.dump(params_dict, out_file, default_flow_style=False)

        if save_gif:
            os.mkdir(self.save_path + "/gifs")
        if save_data:
            os.mkdir(self.save_path + "/data")

        self.angles = []
        self.results = []

    def run(self):

        try:
            for ii, angles in enumerate(self.trials):

                frames, chosen_angle = self.trial_seq.run(angles)
                self.angles.append(angles)
                res = np.minimum(
                    np.abs(angles[1] - chosen_angle),
                    np.abs(angles[1] - 360 - chosen_angle) % 180,
                )
                self.results.append([chosen_angle, res])

                imgs = [Image.fromarray(f.T * 255) for f in frames]
                if self.save_gif:
                    imgs[0].save(
                        f"{self.save_path}/gifs/{ii}_{angles}.gif",
                        save_all=True,
                        append_images=imgs[1:],
                        loop=0,
                        duration=1.5,
                    )

                if self.save_data:
                    np.save(f"{self.save_path}/data/{ii}_{angles}", frames)

            self.angles = np.array(self.angles)
            self.results = np.array(self.results)

            pygame.quit()
            self.save_results()

        except KeyboardInterrupt:

            self.angles = np.array(self.angles)
            self.results = np.array(self.results)

            pygame.quit()
            self.save_results()

    def save_results(self):

        result_dict = {
            "global_bias": [],
            "temp_coherent_bias": [],
            "chosen_angle": [],
            "absolute error": [],
        }
        for angles, (chosen_angle, error) in zip(self.angles, self.results):

            result_dict["global_bias"].append(angles[0])
            result_dict["temp_coherent_bias"].append(angles[1])
            result_dict["chosen_angle"].append(chosen_angle)
            result_dict["absolute error"].append(error)

        results = pd.DataFrame.from_dict(result_dict)
        results.to_csv(f"{self.save_path}/results.csv")
