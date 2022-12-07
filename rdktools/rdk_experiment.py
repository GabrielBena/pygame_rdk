# random dot motion kinematogram
# timo flesch, 2019

import pygame
from pygame.locals import *
import pygame_gui
import ptext

from datetime import datetime

import random
import numpy as np
import pandas as pd

from rdktools.rdk_stimuli import RDK, Fixation, BlankScreen, ResultPrompt
from PIL import Image
import os, shutil
from pathlib import Path
import dataclasses
import pyaml

from copy import deepcopy


def other_angle(angle):
    other = np.random.randint(0, 360)
    if np.abs(other - angle) > 36:
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

    def __init__(self, display, params):

        self.clock = pygame.time.Clock()
        self.display = display
        pygame.display.set_caption(params.WINDOW_NAME)
        self.display.fill(params.COL_BLACK)
        self.rdk = RDK(self.display, params)
        self.fix = Fixation(self.display, params)
        self.iti = BlankScreen(self.display, params)
        self.result_screen = ResultPrompt(self.display, self.fix.centre)

    def run(self, angles=(90, 0)):
        frames_fix = self.fix.show()
        self.rdk.new_sample(angles)
        frames_rdk, chosen_angle, decision_time = self.rdk.show()
        frames_iti = self.iti.show()

        if chosen_angle is None:
            chosen_angle = self.result_screen.show()

        return (
            np.concatenate((frames_fix, frames_rdk, frames_iti), axis=0),
            chosen_angle,
            decision_time,
        )


class Experiment(object):
    def __init__(self, params, randomize=True, save_gif=True, save_data=True) -> None:

        pygame.init()

        # params.NAME = input("Name : ")
        self.params = params
        self.save_gif = save_gif
        self.save_data = save_data
        self.randomize = randomize

        try:
            os.mkdir("experiments")
        except FileExistsError:
            pass

        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y-%H-%M-%S")
        self.save_path = f"experiments/{params.NAME}/{date_time}"
        path = Path(self.save_path)
        path.mkdir(exist_ok=True, parents=True)

        params_dict = dataclasses.asdict(params)
        with open(f"{self.save_path}/params.yml", "w") as out_file:
            pyaml.dump(params_dict, out_file)

        if save_gif:
            os.mkdir(self.save_path + "/gifs")
        if save_data:
            os.mkdir(self.save_path + "/data")

        self.angles = []
        self.results = []
        self.coherences = []
        self.subset_ratio = []

        win_width = params.WINDOW_WIDTH
        win_height = params.WINDOW_HEIGHT
        self.centre = [win_height // 2, win_width // 2]

    def text_prompt(self, center, display, batch=None):

        # create a text surface object,
        # on which text is drawn on it.
        txt = "RDK : Get Ready"
        if batch is not None:
            txt += f"\n Batch {batch} / {self.params.N_BATCH}"

        run = True
        # infinite loop
        while run:

            # completely fill the surface object
            # with white color
            display.fill(self.params.COL_BLACK)

            # copying the text surface object
            # to the display surface object
            # at the center coordinate.
            ptext.draw(txt, center=self.centre, color=self.params.COL_WHITE)

            # iterate over the list of Event objects
            # that was returned by pygame.event.get() method.
            for event in pygame.event.get():

                # if event object type is QUIT
                # then quitting the pygame
                # and program both.
                if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                    run = False

                # Draws the surface object to the screen.
                pygame.display.update()

    def get_random_params(self):

        coherences = []
        for c in self.params.DOT_COHERENCE:
            r = np.random.rand()
            if isinstance(c, tuple):
                coherences.append((c[1] - c[0]) * r + c[0])
            else:
                coherences.append(r * c)

        s = self.params.SUBSET_RATIO
        if isinstance(s, tuple):
            subset_ratio = (s[1] - s[0]) * r + s[0]
        else:
            subset_ratio = r * s

        return coherences, subset_ratio

    def run_batch(self, params=None, batch_idx=0):
        if params is None:
            params = self.params

        self.trials = set_trials(
            n_reps=params.DOT_REPETITIONS,
            angles=params.DOT_ANGLES,
            n_trials=params.N_TRIALS_PER_BATCH,
        )

        self.trial_seq = TrialSequence(self.display, params)

        for ii, angles in enumerate(self.trials):

            frames, chosen_angle, decison_time = self.trial_seq.run(angles)
            self.results.append([chosen_angle])
            self.angles.append(angles)
            for angle in angles:
                res = np.minimum(
                    np.abs(angle - chosen_angle),
                    np.abs(angle - 360 - chosen_angle) % 180,
                )
                self.results[-1].append(res)

            self.results[-1].append(decison_time)
            self.coherences.append(params.DOT_COHERENCE)
            self.subset_ratio.append(params.SUBSET_RATIO)

            imgs = [Image.fromarray(f.T * 255) for f in frames]
            if self.save_gif:
                imgs[0].save(
                    f"{self.save_path}/gifs/{batch_idx}_{ii}_{angles}.gif",
                    save_all=True,
                    append_images=imgs[1:],
                    loop=0,
                    duration=1.5,
                )

            if self.save_data:
                np.save(f"{self.save_path}/data/{ii}_{angles}", frames)

    def run(self):

        self.display = pygame.display.set_mode(
            (self.params.WINDOW_WIDTH, self.params.WINDOW_HEIGHT)
        )

        for batch in range(self.params.N_BATCH):

            params = deepcopy(self.params)
            if self.randomize:
                coherences, subset_ratio = self.get_random_params()

            params.DOT_COHERENCE = coherences
            params.SUBSET_RATIO = subset_ratio

            try:
                self.text_prompt(self.centre, self.display, batch)
                self.run_batch(params, batch)
            except KeyboardInterrupt:
                break

        pygame.quit()
        self.save_results()

    def save_results(self):

        self.angles = np.array(self.angles)
        self.results = np.array(self.results)

        result_dict = {
            "global_direction": [],
            "subset_direction": [],
            "global_coherence": [],
            "subset_coherence": [],
            "subset_ratio": [],
            "chosen_angle": [],
            "absolute error_global": [],
            "absolute error_subset": [],
            "decision_time": [],
        }
        for angles, (chosen_angle, error1, error2, d_time), (c1, c2), s_r in zip(
            self.angles, self.results, self.coherences, self.subset_ratio
        ):

            result_dict["global_direction"].append(angles[0])
            result_dict["subset_direction"].append(angles[1])
            result_dict["chosen_angle"].append(chosen_angle)
            result_dict["absolute error_global"].append(error1)
            result_dict["absolute error_subset"].append(error2)
            result_dict["decision_time"].append(d_time)
            result_dict["global_coherence"].append(c1)
            result_dict["subset_coherence"].append(c2)
            result_dict["subset_ratio"].append(s_r)

        results = pd.DataFrame.from_dict(result_dict)

        try:
            existing_results = pd.read_csv(f"{self.save_path}/results.csv")
            results = pd.concat([existing_results, results])
        except FileNotFoundError:
            pass

        self.results_pd = results
        results.to_csv(f"{self.save_path}/results.csv")
