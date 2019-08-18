import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


__all__ = ['Observation']


class Observation(object):
    def __init__(self, times, fluxes, target=None):
        self.times = times
        self.fluxes = fluxes
        self.target = target
