"""
Point Cloud Morphology Visualization Module

This module provides advanced visualization tools for analyzing the morphological
properties of event point clouds in 3D space. It includes functions for:
- Interactive 3D point cloud visualization with morphology metrics
- Density-based coloring and clustering visualization
- Spatial distribution analysis
- Temporal evolution of morphology
- Principal component analysis and orientation
"""

import inspect
try:
    from eve_smlm.Utils import utilsHelper
    from eve_smlm.EventDistributions import eventDistributions
except ImportError:
    from Utils import utilsHelper
    from EventDistributions import eventDistributions

import pandas as pd
import numpy as np
import logging
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Required function __function_metadata__
def __function_metadata__():
    return {
        "PointCloudMorphologyAnalysis": {
            "required_kwargs": [
                {"name": "color_by", "display_text": "Color by", 
                 "description": "Coloring scheme for point cloud",
                 "default": "density", "options": ["density", "time", "polarity", "cluster"]},
                {"name": "show_metrics", "display_text": "Show metrics", 
                 "description": "Display morphology metrics on plot", "default": "True"},
            ],
            "optional_kwargs": [
                {"name": "alpha", "display_text": "Point transparency", 
                 "description": "Transparency of points (0-1)", "default": "0.6"},
                {"name": "point_size", "display_text": "Point size", 
                 "description": "Size of scatter points", "default": "20"},
                {"name": "show_pca", "display_text": "Show PCA axes", 
                 "description": "Display principal component axes", "default": "True"},
                {"name": "show_hull", "display_text": "Show convex hull", 
                 "description": "Display convex hull boundary", "default": "False"},
            ],
            "help_string": "Advanced 3D point cloud visualization with morphology analysis including density, PCA, and shape metrics.",
            "display_name": "Point Cloud Morphology Analysis"
        },
        "PointCloudDensityMap": {
            "required_kwargs": [
                {"name": "projection_axis", "display_text": "Projection axis", 
                 "description": "Axis for density projection",
                 "default": "z", "options": ["x", "y", "z", "xy", "xz", "yz"]},
            ],
            "optional_kwargs": [
                {"name": "bins", "display_text": "Number of bins", 
                 "description": "Resolution of density map", "default": "50"},
                {"name": "kde_bandwidth", "display_text": "KDE bandwidth", 
                 "description": "Kernel density estimation bandwidth factor", "default": "0.1"},
            ],
            "help_string": "Creates 2D density maps from 3D point cloud projections with kernel density estimation.",
            "display_name": "Point Cloud Density Projection"
        },
        "PointCloudTemporalEvolution": {
            "required_kwargs": [
                {"name": "n_slices", "display_text": "Number of time slices", 
                 "description": "Number of temporal subdivisions to analyze",
                 "default": "5", "type": int},
            ],
            "optional_kwargs": [
                {"name": "metric", "display_text": "Morphology metric", 
                 "description": "Metric to track over time",
                 "default": "volume", "options": ["volume", "density", "spread", "anisotropy"]},
            ],
            "help_string": "Analyzes temporal evolution of point cloud morphology by dividing events into time slices.",
            "display_name": "Temporal Morphology Evolution"
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
# Helper functions for morphology analysis
#-------------------------------------------------------------------------------------------------------------------------------

def compute_morphology_metrics(events_df, pixel_size=1.0):
    """
    Compute comprehensive morphology metrics for a point cloud.
    
    Args:
        events_df: DataFrame with x, y, t columns
        pixel_size: Size of pixel in nm for scaling
        
    Returns:
        dict: Dictionary of morphology metrics
    """
    metrics = {}
    
    # Extract coordinates
    coords = events_df[['x', 'y', 't']].values
    n_events = len(coords)
    
    if n_events < 4:
        logging.warning("Too few events for morphology analysis")
        return metrics
    
    # Center the coordinates
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    
    # Basic statistics
    metrics['n_events'] = n_events
    metrics['centroid_x'] = centroid[0]
    metrics['centroid_y'] = centroid[1]
    metrics['centroid_t'] = centroid[2]
    
    # Spatial extent
    ranges = np.ptp(coords, axis=0)
    metrics['x_range'] = ranges[0]
    metrics['y_range'] = ranges[1]
    metrics['t_range'] = ranges[2]
    
    # Volume (3D bounding box)
    metrics['volume'] = np.prod(ranges)
    
    # Density
    if metrics['volume'] > 0:
        metrics['density'] = n_events / metrics['volume']
    else:
        metrics['density'] = 0
    
    # Convex hull (if enough points)
    try:
        hull = ConvexHull(coords)
        metrics['convex_hull_volume'] = hull.volume
        metrics['convex_hull_area'] = hull.area
        metrics['compactness'] = n_events / hull.volume if hull.volume > 0 else 0
    except Exception as e:
        logging.debug(f"Could not compute convex hull: {e}")
        metrics['convex_hull_volume'] = None
        metrics['convex_hull_area'] = None
        metrics['compactness'] = None
    
    # PCA for orientation and anisotropy
    try:
        pca = PCA(n_components=3)
        pca.fit(centered_coords)
        
        metrics['pca_components'] = pca.components_
        metrics['explained_variance'] = pca.explained_variance_
        metrics['explained_variance_ratio'] = pca.explained_variance_ratio_
        
        # Anisotropy measures
        ev = pca.explained_variance_
        metrics['anisotropy'] = (ev[0] - ev[2]) / ev[0] if ev[0] > 0 else 0
        metrics['linearity'] = (ev[0] - ev[1]) / ev[0] if ev[0] > 0 else 0
        metrics['planarity'] = (ev[1] - ev[2]) / ev[0] if ev[0] > 0 else 0
        metrics['sphericity'] = ev[2] / ev[0] if ev[0] > 0 else 0
        
    except Exception as e:
        logging.debug(f"Could not perform PCA: {e}")
    
    # Nearest neighbor distances
    if n_events > 1:
        dist_mat = distance_matrix(coords, coords)
        np.fill_diagonal(dist_mat, np.inf)
        nn_distances = np.min(dist_mat, axis=1)
        
        metrics['mean_nn_distance'] = np.mean(nn_distances)
        metrics['std_nn_distance'] = np.std(nn_distances)
        metrics['min_nn_distance'] = np.min(nn_distances)
        metrics['max_nn_distance'] = np.max(nn_distances)
    
    return metrics


def compute_local_density(coords, bandwidth=None):
    """
    Compute local density at each point using kernel density estimation.
    
    Args:
        coords: Nx3 array of coordinates
        bandwidth: KDE bandwidth (None for automatic)
        
    Returns:
        array: Density value at each point
    """
    try:
        if bandwidth is not None:
            kde = gaussian_kde(coords.T, bw_method=bandwidth)
        else:
            kde = gaussian_kde(coords.T)
        densities = kde(coords.T)
        return densities
    except Exception as e:
        logging.warning(f"Could not compute density: {e}")
        return np.ones(len(coords))


def format_metrics_text(metrics):
    """Format morphology metrics as text for display."""
    text_lines = []
    text_lines.append(f"N events: {metrics.get('n_events', 'N/A')}")
    text_lines.append(f"Volume: {metrics.get('volume', 0):.2f}")
    text_lines.append(f"Density: {metrics.get('density', 0):.4f}")
    
    if 'anisotropy' in metrics:
        text_lines.append(f"Anisotropy: {metrics['anisotropy']:.3f}")
        text_lines.append(f"Linearity: {metrics['linearity']:.3f}")
        text_lines.append(f"Planarity: {metrics['planarity']:.3f}")
        text_lines.append(f"Sphericity: {metrics['sphericity']:.3f}")
    
    if 'mean_nn_distance' in metrics:
        text_lines.append(f"Mean NN dist: {metrics['mean_nn_distance']:.2f}")
    
    return '\n'.join(text_lines)


#-------------------------------------------------------------------------------------------------------------------------------
# Callable visualization functions
#-------------------------------------------------------------------------------------------------------------------------------

def PointCloudMorphologyAnalysis(findingResult, fittingResult, previewEvents, figure, settings, **kwargs):
    """
    Advanced 3D point cloud visualization with morphology analysis.
    
    This function creates an interactive 3D visualization of event point clouds
    with comprehensive morphology metrics including density, PCA orientation,
    and shape characteristics.
    """
    # Check kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(
        __function_metadata__(), inspect.currentframe().f_code.co_name, kwargs)
    
    pixel_size = float(settings['PixelSize_nm']['value'])
    
    # Parse parameters
    color_by = kwargs.get('color_by', 'density')
    show_metrics = utilsHelper.strtobool(kwargs.get('show_metrics', 'True'))
    alpha = float(kwargs.get('alpha', 0.6))
    point_size = float(kwargs.get('point_size', 20))
    show_pca = utilsHelper.strtobool(kwargs.get('show_pca', 'True'))
    show_hull = utilsHelper.strtobool(kwargs.get('show_hull', 'False'))
    
    events = findingResult.copy()
    
    if len(events) < 4:
        logging.warning("Too few events for morphology analysis")
        ax = figure.add_subplot(111, projection='3d')
        ax.text(0.5, 0.5, 0.5, 'Insufficient events for analysis',
                ha='center', va='center', transform=ax.transAxes)
        return 1
    
    # Compute morphology metrics
    metrics = compute_morphology_metrics(events, pixel_size)
    
    # Set up 3D plot
    ax = figure.add_subplot(111, projection='3d')
    ax.tick_params(axis="y", pad=0.5)
    ax.tick_params(axis="z", pad=0.5)
    ax.tick_params(axis="x", pad=0.5)
    
    # Extract coordinates
    coords = events[['x', 'y', 't']].values
    x, y, t = coords[:, 0], coords[:, 1], coords[:, 2] * 1e-3  # Convert t to ms
    
    # Determine colors based on coloring scheme
    if color_by == 'density':
        # Color by local density
        colors = compute_local_density(coords)
        cmap = plt.cm.viridis
        label = 'Density'
    elif color_by == 'time':
        # Color by time
        colors = t
        cmap = plt.cm.plasma
        label = 'Time (ms)'
    elif color_by == 'polarity':
        # Color by polarity if available
        if 'p' in events.columns:
            colors = events['p'].values
            cmap = LinearSegmentedColormap.from_list('polarity', ['C1', 'C0'])
            label = 'Polarity'
        else:
            colors = 'C0'
            cmap = None
            label = None
    elif color_by == 'cluster':
        # Color by spatial clustering (simple z-axis binning)
        colors = np.digitize(t, bins=np.linspace(t.min(), t.max(), 10))
        cmap = plt.cm.tab10
        label = 'Time bin'
    else:
        colors = 'C0'
        cmap = None
        label = None
    
    # Scatter plot
    if cmap is not None:
        scatter = ax.scatter(x, y, t, c=colors, s=point_size, alpha=alpha, 
                           cmap=cmap, edgecolors='none')
        if label:
            cbar = figure.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label(label)
    else:
        ax.scatter(x, y, t, c=colors, s=point_size, alpha=alpha, edgecolors='none')
    
    # Plot PCA axes if requested
    if show_pca and 'pca_components' in metrics:
        centroid = metrics['centroid_x'], metrics['centroid_y'], metrics['centroid_t'] * 1e-3
        components = metrics['pca_components']
        var_ratio = metrics['explained_variance_ratio']
        
        # Scale axes by explained variance
        scale_factor = np.ptp(coords, axis=0) * 0.3
        
        for i, (comp, var) in enumerate(zip(components, var_ratio)):
            # Adjust component for time scaling
            comp_scaled = comp.copy()
            comp_scaled[2] *= 1e-3  # Scale time component
            
            axis_scale = scale_factor * np.sqrt(var)
            ax.quiver(centroid[0], centroid[1], centroid[2],
                     comp_scaled[0] * axis_scale[0],
                     comp_scaled[1] * axis_scale[1],
                     comp_scaled[2] * axis_scale[2],
                     color=f'C{i+2}', arrow_length_ratio=0.3, linewidth=2,
                     label=f'PC{i+1} ({var*100:.1f}%)')
    
    # Plot convex hull if requested
    if show_hull:
        try:
            hull = ConvexHull(coords)
            # Plot hull edges
            for simplex in hull.simplices:
                simplex_coords = coords[simplex]
                # Close the triangle
                simplex_coords = np.vstack([simplex_coords, simplex_coords[0]])
                ax.plot(simplex_coords[:, 0], simplex_coords[:, 1], 
                       simplex_coords[:, 2] * 1e-3,
                       'k-', alpha=0.1, linewidth=0.5)
        except Exception as e:
            logging.debug(f"Could not plot convex hull: {e}")
    
    # Plot localization(s)
    if len(fittingResult) > 0:
        ax.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, 
               fittingResult['t'], marker='x', c='red', markersize=10,
               label='Localization(s)', linestyle='none')
    
    # Labels
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    ax.set_zlabel('t [ms]')
    ax.invert_zaxis()
    
    # Add metrics text if requested
    if show_metrics:
        metrics_text = format_metrics_text(metrics)
        ax.text2D(0.02, 0.98, metrics_text, transform=ax.transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    figure.tight_layout()
    
    return 1


def PointCloudDensityMap(findingResult, fittingResult, previewEvents, figure, settings, **kwargs):
    """
    Create 2D density maps from 3D point cloud projections.
    
    This function generates kernel density estimation maps of the point cloud
    projected onto different axes, revealing spatial clustering patterns.
    """
    # Check kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(
        __function_metadata__(), inspect.currentframe().f_code.co_name, kwargs)
    
    pixel_size = float(settings['PixelSize_nm']['value'])
    
    # Parse parameters
    projection_axis = kwargs.get('projection_axis', 'z')
    bins = int(kwargs.get('bins', 50))
    kde_bandwidth = float(kwargs.get('kde_bandwidth', 0.1))
    
    events = findingResult.copy()
    
    if len(events) < 2:
        logging.warning("Too few events for density map")
        return 1
    
    # Determine projection
    if projection_axis == 'z' or projection_axis == 'xy':
        proj_coords = events[['x', 'y']].values
        xlabel, ylabel = 'x [px]', 'y [px]'
    elif projection_axis == 'x' or projection_axis == 'yz':
        proj_coords = events[['y', 't']].values
        proj_coords[:, 1] *= 1e-3  # Convert t to ms
        xlabel, ylabel = 'y [px]', 't [ms]'
    elif projection_axis == 'y' or projection_axis == 'xz':
        proj_coords = events[['x', 't']].values
        proj_coords[:, 1] *= 1e-3  # Convert t to ms
        xlabel, ylabel = 'x [px]', 't [ms]'
    else:
        logging.error(f"Unknown projection axis: {projection_axis}")
        return 1
    
    # Create grid for density estimation
    x_min, x_max = proj_coords[:, 0].min(), proj_coords[:, 0].max()
    y_min, y_max = proj_coords[:, 1].min(), proj_coords[:, 1].max()
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    xx, yy = np.mgrid[x_min:x_max:complex(bins), y_min:y_max:complex(bins)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Kernel density estimation
    try:
        kde = gaussian_kde(proj_coords.T, bw_method=kde_bandwidth)
        density = np.reshape(kde(positions).T, xx.shape)
    except Exception as e:
        logging.error(f"KDE failed: {e}. Using histogram instead.")
        density, _, _ = np.histogram2d(proj_coords[:, 0], proj_coords[:, 1], 
                                      bins=bins, range=[[x_min, x_max], [y_min, y_max]])
        density = density.T
    
    # Plot
    ax = figure.add_subplot(111)
    
    im = ax.imshow(density, origin='lower', extent=[x_min, x_max, y_min, y_max],
                   cmap='hot', aspect='auto', interpolation='bilinear')
    
    # Overlay events as scatter
    ax.scatter(proj_coords[:, 0], proj_coords[:, 1], s=2, c='cyan', 
              alpha=0.3, edgecolors='none', label='Events')
    
    # Overlay localizations
    if len(fittingResult) > 0:
        if projection_axis == 'z' or projection_axis == 'xy':
            ax.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size,
                   'wx', markersize=10, markeredgewidth=2, label='Localizations')
        elif projection_axis == 'x' or projection_axis == 'yz':
            ax.plot(fittingResult['y']/pixel_size, fittingResult['t'],
                   'wx', markersize=10, markeredgewidth=2, label='Localizations')
        elif projection_axis == 'y' or projection_axis == 'xz':
            ax.plot(fittingResult['x']/pixel_size, fittingResult['t'],
                   'wx', markersize=10, markeredgewidth=2, label='Localizations')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Density Map - {projection_axis.upper()} Projection')
    
    cbar = figure.colorbar(im, ax=ax)
    cbar.set_label('Density')
    
    ax.legend(loc='upper right')
    figure.tight_layout()
    
    return 1


def PointCloudTemporalEvolution(findingResult, fittingResult, previewEvents, figure, settings, **kwargs):
    """
    Analyze temporal evolution of point cloud morphology.
    
    This function divides events into temporal slices and tracks how
    morphological properties change over time.
    """
    # Check kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(
        __function_metadata__(), inspect.currentframe().f_code.co_name, kwargs)
    
    pixel_size = float(settings['PixelSize_nm']['value'])

    # Parse parameters
    n_slices = int(kwargs.get('n_slices', 5))
    metric = kwargs.get('metric', 'volume')
    
    events = findingResult.copy()
    
    if len(events) < n_slices * 2:
        logging.warning("Too few events for temporal evolution analysis")
        return 1
    
    # Sort by time
    events = events.sort_values('t')
    
    # Divide into time slices
    slice_size = len(events) // n_slices
    time_values = []
    metric_values = []
    
    for i in range(n_slices):
        start_idx = i * slice_size
        if i == n_slices - 1:
            end_idx = len(events)
        else:
            end_idx = (i + 1) * slice_size
        
        slice_events = events.iloc[start_idx:end_idx]
        
        if len(slice_events) < 4:
            continue
        
        # Compute metrics for this slice
        slice_metrics = compute_morphology_metrics(slice_events, pixel_size)
        
        # Record time (use mean time of slice)
        time_values.append(slice_events['t'].mean() * 1e-3)  # Convert to ms
        
        # Extract requested metric
        if metric == 'volume':
            metric_values.append(slice_metrics.get('volume', 0))
        elif metric == 'density':
            metric_values.append(slice_metrics.get('density', 0))
        elif metric == 'spread':
            # Use max range across dimensions
            metric_values.append(max(slice_metrics.get('x_range', 0),
                                   slice_metrics.get('y_range', 0),
                                   slice_metrics.get('t_range', 0)))
        elif metric == 'anisotropy':
            metric_values.append(slice_metrics.get('anisotropy', 0))
        else:
            logging.warning(f"Unknown metric: {metric}")
            metric_values.append(0)
    
    # Create subplot layout
    ax1 = figure.add_subplot(211, projection='3d')
    ax2 = figure.add_subplot(212)
    
    # Top plot: 3D point cloud with time slices colored
    coords = events[['x', 'y', 't']].values
    t_normalized = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min())
    slice_colors = np.digitize(t_normalized, bins=np.linspace(0, 1, n_slices+1)) - 1
    
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2] * 1e-3,
                         c=slice_colors, s=20, alpha=0.6, cmap='tab10',
                         edgecolors='none')
    
    ax1.set_xlabel('x [px]')
    ax1.set_ylabel('y [px]')
    ax1.set_zlabel('t [ms]')
    ax1.invert_zaxis()
    ax1.set_title('Point Cloud with Time Slices')
    
    # Bottom plot: Metric evolution
    ax2.plot(time_values, metric_values, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel(metric.capitalize())
    ax2.set_title(f'{metric.capitalize()} Evolution Over Time')
    ax2.grid(True, alpha=0.3)
    
    figure.tight_layout()
    
    return 1
