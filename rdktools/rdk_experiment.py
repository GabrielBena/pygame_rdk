# random dot motion kinematogram
# timo flesch, 2019

import pygame
from pygame.locals import *
import pygame_gui
from ptext import ptext


from datetime import datetime
import random
import numpy as np
import pandas as pd

from PIL import Image
import os, shutil
from pathlib import Path
import dataclasses
from dacite import from_dict
import pyaml
import yaml
import matplotlib.pyplot
from copy import deepcopy

from rdktools.rdk_stimuli import RDK, Fixation, BlankScreen, ResultPrompt
from rdktools.rdk_params import Params, get_random_params
from heatmap.plot import compute_and_plot_heatmap, compute_and_plot_colormesh


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
        frames_iti = self.fix.show()  # self.iti.show()

        for f in [frames_rdk, frames_iti]:
            if isinstance(f, str):
                if f == "exit":
                    return "exit", 0, 0

        if chosen_angle is None:
            chosen_angle = self.result_screen.show()

        return (
            np.concatenate((frames_fix, frames_rdk, frames_iti), axis=0),
            chosen_angle,
            decision_time,
        )


class Experiment(object):
    def __init__(
        self,
        params,
        path=None,
        randomize=True,
        save_gif=True,
        save_data=True,
    ) -> None:

        if params is None:
            assert path is not None
            self.load_exp(path)

        else:
            self.params = params

            try:
                os.mkdir("experiments")
            except FileExistsError:
                pass
            if path is None:
                now = datetime.now()
                date_time = now.strftime("%d-%m-%Y-%H-%M-%S")
                self.save_path = f"experiments/{self.params.NAME}/{date_time}"
                path = Path(self.save_path)
                path.mkdir(exist_ok=True, parents=True)
            else:
                self.save_path = path

            params_dict = dataclasses.asdict(self.params)
            with open(f"{self.save_path}/params.yml", "w") as out_file:
                pyaml.dump(params_dict, out_file)

            if save_gif:
                os.mkdir(self.save_path + "/gifs")
            if save_data:
                os.mkdir(self.save_path + "/data")

        pygame.init()

        # self.params.NAME = input("Name : ")

        self.save_gif = save_gif
        self.save_data = save_data
        self.randomize = randomize

        self.angles = []
        self.results = []
        self.coherences = []
        self.subset_ratio = []

        win_width = self.params.WINDOW_WIDTH
        win_height = self.params.WINDOW_HEIGHT
        self.centre = [win_height // 2, win_width // 2]

    def text_prompt(self, center, display, batch=None):

        # create a text surface object,
        # on which text is drawn on it.

        if batch is not None:
            txt = f"RDK : \n Batch {batch} / {self.params.N_BATCH}"
        else:
            txt = "RDK : Get Ready \n Choose a direction at any point during trial \n by clicking with the mouse"

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
            if isinstance(frames, str):
                if frames == "exit":
                    return "exit"

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

        self.text_prompt(self.centre, self.display, None)
        for batch in range(self.params.N_BATCH):

            params = deepcopy(self.params)
            if self.randomize:
                coherences, subset_ratio = get_random_params(params)
                # print(coherences, subset_ratio)
                params.DOT_COHERENCE = coherences
                params.SUBSET_RATIO = subset_ratio

            try:
                self.text_prompt(self.centre, self.display, batch + 1)
                r = self.run_batch(params, batch)
                if r == "exit":
                    raise KeyboardInterrupt
                self.save_results()

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
            "absolute_error_global": [],
            "absolute_error_subset": [],
            "decision_time": [],
        }
        for angles, (chosen_angle, error1, error2, d_time), (c1, c2), s_r in zip(
            self.angles, self.results, self.coherences, self.subset_ratio
        ):

            result_dict["global_direction"].append(angles[0])
            result_dict["subset_direction"].append(angles[1])
            result_dict["chosen_angle"].append(chosen_angle)
            result_dict["absolute_error_global"].append(error1)
            result_dict["absolute_error_subset"].append(error2)
            result_dict["decision_time"].append(d_time)
            result_dict["global_coherence"].append(c1)
            result_dict["subset_coherence"].append(c2)
            result_dict["subset_ratio"].append(s_r)

        results = pd.DataFrame.from_dict(result_dict)

        try:
            existing_results = pd.read_csv(f"{self.save_path}/results.csv", index_col=0)
            results = pd.concat([existing_results, results])
        except FileNotFoundError:
            pass

        self.results_pd = results
        results.to_csv(f"{self.save_path}/results.csv")

        self.angles = []
        self.results = []
        self.coherences = []
        self.subset_ratio = []

    def plot_results(
        self,
        type="scipy",
        metric="absolute_error_subset",
        resolution=100,
        smoothness=3,
        use_ratio=False,
    ):

        values = self.results_pd[["subset_coherence", "subset_ratio", metric]].values.T

        y_label = "subset_proportion"

        if use_ratio:
            coherence_ratio = (
                self.results_pd["subset_coherence"].values
                / self.results_pd["global_coherence"].values
            )
            values = (coherence_ratio, *values[1:])

            x_label = "coherence_ratio"
        else:
            x_label = "subset_coherence"

        if type == "scipy":
            *_, (fig, ax), cbar = compute_and_plot_colormesh(
                values, resolution=resolution
            )
        else:
            *_, (fig, ax), cbar = compute_and_plot_heatmap(
                values, resolution=resolution, smoothness=smoothness, random=False
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        cbar.set_label(metric)

        # fig.suptitle(f"{metric} as function of {x_label} and {y_label}")

    def load_exp(self, path):
        with open(path + "/params.yml", "r") as file:
            params = yaml.safe_load(file)
        params = {n: tuple(c) if type(c) is list else c for n, c in params.items()}
        self.params = from_dict(data_class=Params, data=params)

        self.results_pd = pd.read_csv(path + "/results.csv")
        self.results = self.results_pd.values
        self.save_path = path
