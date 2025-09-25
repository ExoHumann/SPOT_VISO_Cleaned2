#!/usr/bin/env python3
"""
Test the complete data flow from cross section JSON to global coordinates.
Verify that local cross section coordinates map to correct global directions.
"""

import numpy as np
import json
from models.cross_section import CrossSection
from models.axis import Axis

def test_data_flow():
    print("=== Testing Complete Data Flow ===\n")

    # Create a simple axis for testing
    stations = [0.0, 1000.0, 2000.0]  # mm
    x_coords = [0.0, 1000.0, 2000.0]  # mm - along X
    y_coords = [0.0, 0.0, 0.0]        # mm - constant Y
    z_coords = [0.0, 0.0, 0.0]        # mm - constant Z

    axis = Axis(stations, x_coords, y_coords, z_coords, units="mm")

    # Get frames at station 1000mm
    P, T = axis._positions_and_tangents(np.array([1000.0]))
    T_frame, N_frame, B_frame = axis.parallel_transport_frames(np.array([1000.0]))

    print(f"Axis frames at station 1000mm:")
    print(f"  Position P: [{P[0,0]:.1f}, {P[0,1]:.1f}, {P[0,2]:.1f}]")
    print(f"  Tangent T: [{T_frame[0,0]:.3f}, {T_frame[0,1]:.3f}, {T_frame[0,2]:.3f}]")
    print(f"  Normal N:  [{N_frame[0,0]:.3f}, {N_frame[0,1]:.3f}, {N_frame[0,2]:.3f}]")
    print(f"  Binormal B: [{B_frame[0,0]:.3f}, {B_frame[0,1]:.3f}, {B_frame[0,2]:.3f}]")

    # Test embedding with simple cross section points
    # For a bridge deck: local Y = transverse, local Z = height
    # Point 0: centerline, bottom
    # Point 1: right edge, bottom
    # Point 2: right edge, top
    # Point 3: centerline, top
    test_yz = np.array([[
        [0, 0],      # local Y=0 (center), Z=0 (bottom)
        [500, 0],    # local Y=500 (right), Z=0 (bottom)
        [500, 200],  # local Y=500 (right), Z=200 (top)
        [0, 200]     # local Y=0 (center), Z=200 (top)
    ]])  # (1, 4, 2) shape

    print(f"\nTest cross section points (local Y=transverse, Z=height):")
    for i, (y_local, z_local) in enumerate(test_yz[0]):
        print(f"  Point {i}: local Y(transverse)={y_local}, local Z(height)={z_local}")

    # Embed without rotation
    W = axis.embed_section_points_world(
        stations_mm=np.array([1000.0]),
        yz_points_mm=test_yz,
        rotation_deg=0.0
    )

    print(f"\nEmbedded world coordinates:")
    for i, (x, y, z) in enumerate(W[0]):
        print(f"  Point {i}: global X={x:.1f}, Y={y:.1f}, Z={z:.1f}")

    # Analyze the mapping
    print(f"\nCoordinate mapping analysis:")
    print(f"  Point 0 (center, bottom): global X={W[0,0,0]:.1f}, Y={W[0,0,1]:.1f}, Z={W[0,0,2]:.1f}")
    print(f"  Point 1 (right, bottom):  global X={W[0,1,0]:.1f}, Y={W[0,1,1]:.1f}, Z={W[0,1,2]:.1f}")
    print(f"  Point 2 (right, top):     global X={W[0,2,0]:.1f}, Y={W[0,2,1]:.1f}, Z={W[0,2,2]:.1f}")
    print(f"  Point 3 (center, top):    global X={W[0,3,0]:.1f}, Y={W[0,3,1]:.1f}, Z={W[0,3,2]:.1f}")

    # Check which global direction each local coordinate affects
    transverse_diff = W[0,1] - W[0,0]  # Moving from Y=0 to Y=500 (transverse)
    height_diff = W[0,3] - W[0,0]      # Moving from Z=0 to Z=200 (height)

    print(f"\nDirection analysis:")
    print(f"  Increasing local Y (transverse) by 500: ΔX={transverse_diff[0]:.1f}, ΔY={transverse_diff[1]:.1f}, ΔZ={transverse_diff[2]:.1f}")
    print(f"  Increasing local Z (height) by 200: ΔX={height_diff[0]:.1f}, ΔY={height_diff[1]:.1f}, ΔZ={height_diff[2]:.1f}")

    print(f"\nConclusion:")
    print(f"  Local Y (transverse) → Global {'X' if abs(transverse_diff[0])>0.1 else 'Y' if abs(transverse_diff[1])>0.1 else 'Z'} direction")
    print(f"  Local Z (height) → Global {'X' if abs(height_diff[0])>0.1 else 'Y' if abs(height_diff[1])>0.1 else 'Z'} direction")

    # Test with rotation
    print(f"\n=== Testing with 90° rotation ===")
    W_rot = axis.embed_section_points_world(
        stations_mm=np.array([1000.0]),
        yz_points_mm=test_yz,
        rotation_deg=90.0
    )

    print(f"Embedded world coordinates with 90° rotation:")
    for i, (x, y, z) in enumerate(W_rot[0]):
        print(f"  Point {i}: global X={x:.1f}, Y={y:.1f}, Z={z:.1f}")

    transverse_diff_rot = W_rot[0,1] - W_rot[0,0]
    height_diff_rot = W_rot[0,3] - W_rot[0,0]

    print(f"\nDirection analysis with 90° rotation:")
    print(f"  Increasing local Y (transverse) by 500: ΔX={transverse_diff_rot[0]:.1f}, ΔY={transverse_diff_rot[1]:.1f}, ΔZ={transverse_diff_rot[2]:.1f}")
    print(f"  Increasing local Z (height) by 200: ΔX={height_diff_rot[0]:.1f}, ΔY={height_diff_rot[1]:.1f}, ΔZ={height_diff_rot[2]:.1f}")

if __name__ == "__main__":
    test_data_flow()