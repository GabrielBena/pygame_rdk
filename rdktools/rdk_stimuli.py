# random dot motion kinematogram
# timo flesch, 2019


import pygame
from pygame.locals import *
import random
import numpy as np
from math import cos, sin, atan2, radians, degrees
import matplotlib.pyplot as plt
import time

# from array2gif import write_gif

from rdktools.rdk_params import Params
from rdktools.rdk_helper import polar2cartesian, cartesian2polar, get_mouse_angle


class Fixation(object):
    """implements a fixation cross"""

    def __init__(self, display, params):

        win_width = params.WINDOW_WIDTH
        win_height = params.WINDOW_HEIGHT
        f_col = params.FIX_COLOR
        f_width = params.FIX_WIDTH
        f_size = params.FIX_SIZE
        f_duration = params.TIME_FIX
        a_col = params.APERTURE_COLOR
        a_size = params.APERTURE_RADIUS
        a_width = params.APERTURE_WIDTH
        win_colour = params.WINDOW_COLOUR
        tick_rate = params.TICK_RATE

        self.display = display
        self.win_width = win_width
        self.win_height = win_height
        self.win_colour = win_colour
        self.centre = [self.win_width // 2, self.win_height // 2]
        self.f_col = f_col
        self.f_width = f_width
        self.f_size = f_size
        self.f_duration = f_duration
        self.f_duration_init = f_duration
        self.a_col = a_col
        self.a_size = a_size
        self.a_width = a_width
        self.tick_rate = tick_rate
        self.clock = pygame.time.Clock()

    def draw(self):
        # draw aperture and and fixation cross
        self.fix_x = pygame.draw.line(
            self.display,
            self.f_col,
            [self.centre[0] - self.f_size[0], self.centre[1]],
            [self.centre[0] + self.f_size[0], self.centre[1]],
            self.f_width,
        )

        self.fix_y = pygame.draw.line(
            self.display,
            self.f_col,
            [self.centre[0], self.centre[1] - self.f_size[1]],
            [self.centre[0], self.centre[1] + self.f_size[1]],
            self.f_width,
        )

        self.aperture = pygame.draw.circle(
            self.display, self.a_col, self.centre, self.a_size, self.a_width
        )

    def show(self):
        pygame.display.flip()
        allframes = np.zeros((self.f_duration, self.win_width, self.win_height))
        ii = 0
        while self.f_duration > 0:
            event = pygame.event.get()
            self.clock.tick(self.tick_rate)
            self.draw()
            pygame.display.flip()
            frame = self.collect_frame()
            frame = frame.astype("float32")
            frame *= 255.0 / frame.max()
            allframes[ii, :, :] = frame
            ii += 1
            self.f_duration -= 1
        self.f_duration = self.f_duration_init
        # self.hide()
        return allframes

    def wait(self):
        time.sleep(self.f_duration)

    def hide(self):
        self.display.fill(self.win_colour)
        pygame.display.update()

    def collect_frame(self):
        string_image = pygame.image.tostring(self.display, "RGB")
        temp_surf = pygame.image.fromstring(
            string_image, (self.win_width, self.win_height), "RGB"
        )
        return pygame.surfarray.array2d(temp_surf)


class RDK(object):
    """implements a random dot stimulus"""

    def __init__(
        self,
        display,
        params,
        motiondirs=(180, 0),
    ):
        win_width = params.WINDOW_WIDTH
        win_height = params.WINDOW_HEIGHT
        n_dots = params.N_DOTS
        win_colour = params.WINDOW_COLOUR
        dot_colour = params.DOT_COLOR
        dot_size = params.DOT_SIZE
        dot_speed = params.DOT_SPEED
        duration = params.TIME_RDK
        dot_coherence = params.DOT_COHERENCE
        subset_ratio = params.SUBSET_RATIO
        aperture_radius = params.APERTURE_RADIUS
        radius = random.randint(0, params.APERTURE_RADIUS)
        tick_rate = params.TICK_RATE

        self.display = display
        self.win_width = win_width
        self.win_height = win_height
        self.win_colour = win_colour
        self.centre = [win_width // 2, win_height // 2]
        self.aperture_radius = aperture_radius
        self.ndots = n_dots
        self.dot_size = dot_size
        self.duration_init = duration
        self.duration = duration
        self.angle = random.randint(0, 360)
        self.radius = radius
        self.max_radius = self.aperture_radius - self.dot_size
        self.dot_speed = dot_speed
        self.dot_colour = dot_colour
        self.tick_rate = tick_rate

        self.coherences = dot_coherence
        self.subset_ratio = subset_ratio

        self.motiondirs = motiondirs

        self.params = params

        self.dots = pygame.sprite.Group()
        self.dots = self.sample_dots(self.max_radius, self.ndots)
        self.clock = pygame.time.Clock()
        self.fix = Fixation(self.display, params)

    def draw(self):
        # draws dots
        for dot in self.dots:
            dot.draw()

    def show(self):
        allframes = np.zeros((self.duration, self.win_width, self.win_height))
        ii = 0
        angle = None
        run = True
        while self.duration > 0 and run:

            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                    run = False
                    angle, *_ = get_mouse_angle(self.centre)

            self.clock.tick(self.tick_rate)
            self.display.fill(self.win_colour)
            self.fix.draw()
            self.draw()
            pygame.display.update()
            frame = self.collect_frame()
            frame = frame.astype("float32")
            frame *= 255.0 / frame.max()
            allframes[ii, :, :] = frame
            ii += 1
            self.duration -= 1
            self.update()
        self.duration = self.duration_init
        self.hide()

        return allframes, angle, ii

    def update(self):
        # updates position of dots
        rand = random.random()
        for dot in self.dots:
            if dot.in_subset:
                dot.move(rand)
            else:
                dot.move()

    def hide(self):
        self.display.fill(self.win_colour)
        pygame.display.update()

    def new_sample(self, angles):
        self.motiondirs = angles
        self.dots = self.sample_dots(self.max_radius, self.ndots)

    def sample_dots(self, max_radius, ndots):
        # use weighted sampling distribution to avoid
        # dots clustering close to centre of screen:
        weights = np.arange(max_radius) / sum(np.arange(max_radius))
        radii = np.random.choice(max_radius, ndots, p=weights)
        dots = pygame.sprite.Group()
        for ii in range(ndots):
            in_subset = random.random() < self.subset_ratio
            dots.add(
                RandDot(
                    self.display,
                    self.params,
                    self.centre,
                    radius=radii[ii],
                    motiondir=self.motiondirs[int(in_subset)],
                    dot_coherence=self.coherences[int(in_subset)],
                    in_subset=in_subset,
                )
            )

        return dots

    def collect_frame(self):
        string_image = pygame.image.tostring(self.display, "RGB")
        temp_surf = pygame.image.fromstring(
            string_image, (self.win_width, self.win_height), "RGB"
        )
        return pygame.surfarray.array2d(temp_surf)


class BlankScreen(object):
    """displays a blank screen. returns an n-D array of all displayed frames"""

    def __init__(
        self,
        display,
        params,
    ):
        time = params.TIME_ITI
        win_width = params.WINDOW_WIDTH
        win_height = params.WINDOW_HEIGHT
        win_colour = params.WINDOW_COLOUR
        tick_rate = params.TICK_RATE

        self.display = display
        self.duration = time
        self.duration_init = time
        self.win_width = win_width
        self.win_height = win_height
        self.win_colour = win_colour
        self.tick_rate = tick_rate
        self.clock = pygame.time.Clock()

    def show(self):
        self.display.fill(self.win_colour)
        allframes = np.zeros((self.duration, self.win_width, self.win_height))
        ii = 0
        while self.duration > 0:
            event = pygame.event.get()
            self.clock.tick(self.tick_rate)
            pygame.display.update()
            frame = self.collect_frame()
            frame = frame.astype("float32")
            frame *= 255.0 / frame.max()
            allframes[ii, :, :] = frame
            ii += 1
            self.duration -= 1
        self.duration = self.duration_init
        return allframes

    def collect_frame(self):
        string_image = pygame.image.tostring(self.display, "RGB")
        temp_surf = pygame.image.fromstring(
            string_image, (self.win_width, self.win_height), "RGB"
        )
        return pygame.surfarray.array2d(temp_surf)


class RandDot(pygame.sprite.Sprite):
    """implements a single random dot."""

    def __init__(
        self,
        display,
        params,
        centre,
        radius,
        dot_coherence,
        motiondir=180,
        in_subset=False,
    ):
        super(RandDot, self).__init__()

        dot_colour = params.DOT_COLOR
        dot_size = params.DOT_SIZE
        dot_speed = params.DOT_SPEED
        aperture_radius = params.APERTURE_RADIUS

        self.dot_size = dot_size
        self.dot_colour = dot_colour
        self.surf = pygame.Surface((self.dot_size, self.dot_size))
        self.surf.fill((self.dot_colour))
        self.centre = centre
        self.angle = random.randint(0, 360)
        self.radius = radius
        self.max_radius = aperture_radius - dot_size
        self.dot_speed = dot_speed
        self.coherence = dot_coherence
        self.fixed_motiondir = motiondir
        self.in_subset = in_subset
        self.x_0, self.y_0 = polar2cartesian(self.angle, self.radius)
        self.rect = self.surf.get_rect(
            center=(self.x_0 + self.centre[0], self.y_0 + self.centre[1])
        )
        self.display = display

        self.set_moves()

    def set_moves(self, rand=None):

        if rand is None:
            rand = random.random()

        if rand < self.coherence:
            self.motiondir = self.fixed_motiondir
        else:
            self.motiondir = random.randint(0, 360)

        self.dx, self.dy = polar2cartesian(self.motiondir, self.dot_speed)

    def move(self, rand=None):

        self.set_moves(rand)

        if self.radius >= self.max_radius:
            self.reset_pos()
        self.x_0 += self.dx
        self.y_0 += self.dy
        self.rect.x += self.dx
        self.rect.y += self.dy
        self.angle, self.radius = cartesian2polar(
            self.rect.x - self.centre[0], self.rect.y - self.centre[1]
        )

    def reset_pos(self):
        self.angle = self.motiondir - 180 + random.randint(-90, 90)
        self.radius = self.max_radius
        self.x_0, self.y_0 = polar2cartesian(self.angle, self.radius)
        self.rect.x = self.x_0 + self.centre[0]
        self.rect.y = self.y_0 + self.centre[1]

    def draw(self):
        self.surf.fill(self.dot_colour)
        self.display.blit(self.surf, self.rect)


class ResultPrompt(object):
    def __init__(self, display, center) -> None:
        self.display = display
        self.center = center

    def show(self):
        run = True
        while run:

            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                    run = False

            points, angle = self.rotate_triangle(self.center, 10)

            pygame.Surface.fill(self.display, (255, 255, 255))
            pygame.draw.polygon(self.display, (0, 0, 0), points)
            pygame.display.update()

        return angle

    def rotate_triangle(self, center, scale):

        angle, vCenter, vMouse = get_mouse_angle(center)

        points = [(-0.5, -0.866), (-0.5, 0.866), (2.0, 0.0)]
        rotated_point = [pygame.math.Vector2(p).rotate(angle) for p in points]

        triangle_points = [(vCenter + p * scale) for p in rotated_point]
        return triangle_points, angle
