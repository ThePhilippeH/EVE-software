#!/usr/bin/env python3
"""
Test script to verify PointCloudMorphology functions work correctly
"""
import sys
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, '.')

from Visualisation.PointCloudMorphology import (
    PointCloudMorphologyAnalysis,
    PointCloudDensityMap,
    PointCloudTemporalEvolution
)

# Create mock data
np.random.seed(42)
n_events = 100

# Create synthetic resultArray (fitted localizations)
resultArray = pd.DataFrame({
    'x': np.random.randn(n_events) * 100 + 5000,  # nm
    'y': np.random.randn(n_events) * 100 + 5000,  # nm
    't': np.random.randn(n_events) * 1000 + 10000,  # µs
    'p': np.random.choice([0, 1], n_events)
})

# Create mock settings
settings = {
    'PixelSize_nm': {'value': 100},  # 100 nm per pixel
    'UseCUDA': {'value': 0}
}

print("Testing PointCloudMorphologyAnalysis...")
try:
    image1, scale1 = PointCloudMorphologyAnalysis(
        resultArray, settings,
        color_by='density',
        show_metrics='True'
    )
    print(f"✓ PointCloudMorphologyAnalysis succeeded! Image shape: {image1.shape}, Scale: {scale1}")
except Exception as e:
    print(f"✗ PointCloudMorphologyAnalysis failed: {e}")

print("\nTesting PointCloudDensityMap...")
try:
    image2, scale2 = PointCloudDensityMap(
        resultArray, settings,
        projection_axis='xy',
        bins=50
    )
    print(f"✓ PointCloudDensityMap succeeded! Image shape: {image2.shape}, Scale: {scale2}")
except Exception as e:
    print(f"✗ PointCloudDensityMap failed: {e}")

print("\nTesting PointCloudTemporalEvolution...")
try:
    image3, scale3 = PointCloudTemporalEvolution(
        resultArray, settings,
        n_slices=5,
        metric='volume'
    )
    print(f"✓ PointCloudTemporalEvolution succeeded! Image shape: {image3.shape}, Scale: {scale3}")
except Exception as e:
    print(f"✗ PointCloudTemporalEvolution failed: {e}")

print("\nAll tests completed!")

