# -*- coding: utf-8 -*-
"""
Module Name: Jonswap
Description: This module defines the function to generate a synthetic Jonswap
wave elevation signal.
Author: Eguzkiñe Martinez Puente
Created on: 2023-03-28
"""

import numpy as np


def jonswap_elevation(t, Hs, Tp, dt):
    """
    Generates a synthetic JONSWAP wave elevation signal for a given time range, 
    significant wave height, peak period, and time step.
    """
    N = 2 * len(t)  # Number of time steps
    f = np.fft.fftfreq(N, d=dt)  # Frequency range
    f = f[:N // 2]  # Positive frequencies
    gamma = 3.3  # Spectral shape parameter
    fp = 1 / Tp  # Peak frequency
    sigma = np.where(f <= fp, 0.07, 0.09)  # Spectral width parameter
    alpha = np.exp(-(f - fp)**2 / (2 * sigma**2))  # Scaling factor
    S = np.zeros(len(f))  # Initialize spectrum
    idx = f <= 5 * fp  # Define range for JONSWAP spectrum
    S[idx] = (0.3125 * Hs**2 * fp**4) / ((f[idx]**5) *
                                         np.exp(1.25 * (fp / f[idx])**2)) * gamma**alpha[idx]
    phase = np.random.uniform(0, 2 * np.pi, len(f))  # Random phase
    A = np.sqrt(2 * S * dt)  # Amplitude
    A[0] = 0  # Zero mean
    # Inverse Fourier transform
    elevation = np.fft.ifft(A * np.exp(1j * phase))
    return np.real(elevation) * 100
