"""Verify resample_poly edge behavior: default vs padtype='line'."""
import numpy as np
from scipy.signal import resample_poly

# Constant 8000 across a 35280-sample window (simulates the polling test)
const = np.full(35280, 8000.0)
out_default = resample_poly(const, 160, 441)
out_line = resample_poly(const, 160, 441, padtype="line")
print("default : len", len(out_default), "maxdev", float(np.abs(out_default - 8000).max()))
print("line    : len", len(out_line), "maxdev", float(np.abs(out_line - 8000).max()))

# 1kHz sine must still resample correctly with padtype='line'
t = np.arange(35280) / 44100.0
sine = np.sin(2 * np.pi * 1000.0 * t) * 8000.0
out = resample_poly(sine, 160, 441, padtype="line")
spec = np.abs(np.fft.rfft(out[:12800]))
freqs = np.fft.rfftfreq(12800, 1 / 16000)
print("sine    : len", len(out), "peak freq", float(freqs[int(np.argmax(spec))]))
