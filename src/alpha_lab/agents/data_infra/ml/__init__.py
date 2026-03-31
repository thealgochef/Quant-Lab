"""
ML pipeline for price extrema classification.

Implements a hybrid approach combining:
- PL (Price Level) features from MBP-10 book data
- MS (Market Shift) momentum features from tick data
- Signal features from 20 existing detectors

Based on Sokolovsky & Arnaboldi (2020) adapted for NQ/ES futures.
"""
