"""
Point Cloud Morphology Visualization Demo

This script demonstrates the new point cloud morphology visualization capabilities.
It creates synthetic point clouds with different morphologies and visualizes them
using the new tools.

Run this script to test the morphology visualization features without needing
actual event data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the new morphology modules
try:
    from eve_smlm.Visualisation.PointCloudMorphology import (
        PointCloudMorphologyAnalysis,
        PointCloudDensityMap,
        PointCloudTemporalEvolution
    )
    from eve_smlm.CandidatePreview.PointCloudMorphologyPreview import (
        MorphologyMetricsTable,
        ShapeCharacterization,
        SpatialDistributionAnalysis
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Visualisation.PointCloudMorphology import (
        PointCloudMorphologyAnalysis,
        PointCloudDensityMap,
        PointCloudTemporalEvolution
    )
    from CandidatePreview.PointCloudMorphologyPreview import (
        MorphologyMetricsTable,
        ShapeCharacterization,
        SpatialDistributionAnalysis
    )


def generate_synthetic_events(shape='sphere', n_events=500, noise=0.1):
    """
    Generate synthetic event data with different morphologies.
    
    Args:
        shape: 'sphere', 'rod', 'disk', 'cluster', or 'random'
        n_events: Number of events to generate
        noise: Amount of random noise to add
        
    Returns:
        DataFrame with columns: x, y, t, p
    """
    np.random.seed(42)
    
    if shape == 'sphere':
        # Spherical point cloud
        phi = np.random.uniform(0, 2*np.pi, n_events)
        theta = np.arccos(2*np.random.uniform(0, 1, n_events) - 1)
        r = np.random.uniform(0.5, 1.0, n_events) ** (1/3)  # Uniform volume distribution
        
        x = r * np.sin(theta) * np.cos(phi) * 10 + 50
        y = r * np.sin(theta) * np.sin(phi) * 10 + 50
        t = r * np.cos(theta) * 10000 + 50000
        
    elif shape == 'rod':
        # Rod-like (elongated along one axis)
        main_axis = np.random.uniform(0, 1, n_events)
        perp1 = np.random.normal(0, 0.1, n_events)
        perp2 = np.random.normal(0, 0.1, n_events)
        
        x = main_axis * 40 + 30
        y = perp1 * 5 + 50
        t = perp2 * 5000 + 50000
        
    elif shape == 'disk':
        # Disk-like (spread in a plane)
        r = np.random.uniform(0, 1, n_events) ** 0.5 * 15
        theta = np.random.uniform(0, 2*np.pi, n_events)
        
        x = r * np.cos(theta) + 50
        y = r * np.sin(theta) + 50
        t = np.random.normal(0, 0.5, n_events) * 2000 + 50000
        
    elif shape == 'cluster':
        # Multiple clusters
        n_clusters = 3
        cluster_centers = np.random.uniform(30, 70, (n_clusters, 3))
        cluster_centers[:, 2] *= 1000  # Scale time
        
        cluster_assignment = np.random.choice(n_clusters, n_events)
        
        x = np.zeros(n_events)
        y = np.zeros(n_events)
        t = np.zeros(n_events)
        
        for i in range(n_clusters):
            mask = cluster_assignment == i
            n = np.sum(mask)
            x[mask] = np.random.normal(cluster_centers[i, 0], 5, n)
            y[mask] = np.random.normal(cluster_centers[i, 1], 5, n)
            t[mask] = np.random.normal(cluster_centers[i, 2], 3000, n)
            
    else:  # random
        # Completely random
        x = np.random.uniform(20, 80, n_events)
        y = np.random.uniform(20, 80, n_events)
        t = np.random.uniform(30000, 70000, n_events)
    
    # Add noise
    x += np.random.normal(0, noise, n_events)
    y += np.random.normal(0, noise, n_events)
    t += np.random.normal(0, noise * 1000, n_events)
    
    # Generate random polarities
    p = np.random.choice([0, 1], n_events)
    
    # Create DataFrame
    events_df = pd.DataFrame({
        'x': x,
        'y': y,
        't': t,
        'p': p
    })
    
    return events_df


def demo_morphology_analysis():
    """Demonstrate morphology analysis on different point cloud shapes."""
    
    print("=" * 70)
    print("Point Cloud Morphology Visualization Demo")
    print("=" * 70)
    
    # Settings dictionary (mimicking GUI settings)
    settings = {
        'PixelSize_nm': {'value': 100},
        'UseCUDA': {'value': 0}
    }
    
    # Test different shapes
    shapes = ['sphere', 'rod', 'disk', 'cluster']
    
    for shape in shapes:
        print(f"\n--- Analyzing {shape.upper()} morphology ---")
        
        # Generate synthetic events
        events = generate_synthetic_events(shape=shape, n_events=300)
        
        # Create dummy fitting result (centroid)
        fitting_result = pd.DataFrame({
            'x': [events['x'].mean() * float(settings['PixelSize_nm']['value'])],
            'y': [events['y'].mean() * float(settings['PixelSize_nm']['value'])],
            't': [events['t'].mean()]
        })
        
        # Empty preview events
        preview_events = pd.DataFrame()
        
        # 1. Morphology Analysis
        print("\n1. Creating 3D Morphology Analysis...")
        fig1 = plt.figure(figsize=(12, 8))
        fig1.suptitle(f'{shape.capitalize()} - Morphology Analysis', fontsize=14, fontweight='bold')
        
        kwargs1 = {
            'color_by': 'density',
            'show_metrics': 'True',
            'alpha': '0.6',
            'point_size': '30',
            'show_pca': 'True',
            'show_hull': 'False'
        }
        
        PointCloudMorphologyAnalysis(events, fitting_result, preview_events, 
                                    fig1, settings, **kwargs1)
        
        # 2. Density Map
        print("2. Creating Density Map...")
        fig2 = plt.figure(figsize=(10, 8))
        fig2.suptitle(f'{shape.capitalize()} - Density Map', fontsize=14, fontweight='bold')
        
        kwargs2 = {
            'projection_axis': 'z',
            'bins': '50',
            'kde_bandwidth': '0.1'
        }
        
        PointCloudDensityMap(events, fitting_result, preview_events, 
                           fig2, settings, **kwargs2)
        
        # 3. Metrics Table
        print("3. Creating Metrics Table...")
        fig3 = plt.figure(figsize=(10, 10))
        fig3.suptitle(f'{shape.capitalize()} - Metrics', fontsize=14, fontweight='bold')
        
        kwargs3 = {
            'show_advanced': 'True'
        }
        
        MorphologyMetricsTable(events, fitting_result, preview_events,
                              fig3, settings, **kwargs3)
        
        # 4. Shape Characterization
        print("4. Creating Shape Characterization...")
        fig4 = plt.figure(figsize=(10, 8))
        fig4.suptitle(f'{shape.capitalize()} - Shape', fontsize=14, fontweight='bold')
        
        kwargs4 = {
            'view': 'xy',
            'show_ellipse': 'True',
            'confidence_level': '0.95'
        }
        
        ShapeCharacterization(events, fitting_result, preview_events,
                            fig4, settings, **kwargs4)

        break
    
    # 5. Temporal Evolution (using cluster shape)
    print("\n--- Analyzing TEMPORAL EVOLUTION ---")
    events = generate_synthetic_events(shape='cluster', n_events=500)
    fitting_result = pd.DataFrame({
        'x': [events['x'].mean() * float(settings['PixelSize_nm']['value'])],
        'y': [events['y'].mean() * float(settings['PixelSize_nm']['value'])],
        't': [events['t'].mean()]
    })
    
    fig5 = plt.figure(figsize=(12, 10))
    fig5.suptitle('Temporal Evolution Analysis', fontsize=14, fontweight='bold')
    
    kwargs5 = {
        'n_slices': '5',
        'metric': 'volume'
    }
    
    PointCloudTemporalEvolution(events, fitting_result, preview_events,
                               fig5, settings, **kwargs5)
    
    # 6. Spatial Distribution Analysis
    print("\n--- Analyzing SPATIAL DISTRIBUTION ---")
    fig6 = plt.figure(figsize=(14, 10))
    fig6.suptitle('Spatial Distribution Analysis', fontsize=14, fontweight='bold')
    
    kwargs6 = {
        'n_bins': '30'
    }
    
    SpatialDistributionAnalysis(events, fitting_result, preview_events,
                               fig6, settings, **kwargs6)
    
    print("\n" + "=" * 70)
    print("Demo complete! Close the plot windows to exit.")
    print("=" * 70)
    
    plt.show()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Starting Point Cloud Morphology Visualization Demo...")
    print("=" * 70)
    print("\nThis demo will create several visualization windows showing")
    print("different morphology analysis tools on synthetic data.")
    print("\nPress Ctrl+C to cancel, or wait for plots to appear...")
    print("=" * 70 + "\n")
    
    try:
        demo_morphology_analysis()
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis is normal if dependencies are not installed.")
        print("Please ensure all requirements are installed:")
        print("  pip install -r requirements.txt")
