#!/usr/bin/env python3
"""
Test to verify that rotatio    # Test 0° rotation (should be unchanged)
    points_0deg = axis.embed_section_points_world(stations_mm[:1], yz_points, rotation_deg=0.0)
    print(f"\n0° rotation result shape: {points_0deg.shape}")
    print("0° rotation points (X, Y, Z in mm):")
    for i, point in enumerate(points_0deg[0]):
        print(f"  Point {i}: ({point[0]:6.1f}, {point[1]:6.1f}, {point[2]:6.1f})")

    # Debug: manually compute what 0° should give
    print("\nDebug: Manual calculation for 0° rotation:")
    y_local = yz_points[:, 0]  # [0, 1000, 1000, 0, 0]
    z_local = yz_points[:, 1]  # [0, 0, 1000, 1000, 0]
    print(f"y_local: {y_local}")
    print(f"z_local: {z_local}")
    print(f"N: {N[0]}")
    print(f"B: {B[0]}")
    for i in range(len(y_local)):
        world_point = (y_local[i] * N[0] + z_local[i] * B[0])
        print(f"  Point {i} contribution: y*N + z*B = {y_local[i]}*{N[0]} + {z_local[i]}*{B[0]} = {world_point}")
        print(f"    World coords: {world_point}")athematically correct.
Creates a simple cross section and tests rotation by 90 degrees.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.axis import Axis

def test_rotation_math():
    """Test that 90° rotation actually rotates points correctly."""

    print("=== Testing Rotation Mathematics ===\n")

    # Create a straight axis along X so local YZ plane = global YZ plane
    stations_m = [0.0, 50.0, 100.0]
    x_coords = [0.0, 50.0, 100.0]  # Straight along X
    y_coords = [0.0, 0.0, 0.0]     # No curvature
    z_coords = [0.0, 0.0, 0.0]

    axis = Axis(stations_m, x_coords, y_coords, z_coords, units="m")

    # Create a simple cross section: square with points at (0,0), (1000,0), (1000,1000), (0,1000) mm
    # This is in the YZ plane
    yz_points = np.array([
        [0.0, 0.0],      # origin
        [1000.0, 0.0],   # +Y direction
        [1000.0, 1000.0], # +Y+Z
        [0.0, 1000.0],   # +Z direction
        [0.0, 0.0]       # back to origin
    ])

    stations_mm = np.array([0.0, 50000.0])  # Test at station 0 and 50m

    # Debug: print the local frames at station 0 (index 0)
    P, T = axis._positions_and_tangents(stations_mm)
    _, N, B = axis.parallel_transport_frames(stations_mm)
    
    print(f"At station 0:")
    print(f"  Position P: ({P[0][0]:.1f}, {P[0][1]:.1f}, {P[0][2]:.1f})")
    print(f"  Tangent T: ({T[0][0]:.3f}, {T[0][1]:.3f}, {T[0][2]:.3f})")
    print(f"  Normal N: ({N[0][0]:.3f}, {N[0][1]:.3f}, {N[0][2]:.3f})")
    print(f"  Binormal B: ({B[0][0]:.3f}, {B[0][1]:.3f}, {B[0][2]:.3f})")
    print(f"  N·T = {np.dot(N[0], T[0]):.6f} (should be 0)")
    print(f"  B·T = {np.dot(B[0], T[0]):.6f} (should be 0)")
    print(f"  N·B = {np.dot(N[0], B[0]):.6f} (should be 0)")
    print()

    print("Original cross section points (Y, Z in mm):")
    for i, (y, z) in enumerate(yz_points):
        print(f"  Point {i}: ({y:6.1f}, {z:6.1f})")

    # Test 0° rotation (should be unchanged)
    points_0deg = axis.embed_section_points_world(stations_mm[:1], yz_points, rotation_deg=0.0)
    print(f"\n0° rotation result shape: {points_0deg.shape}")
    print("0° rotation points (X, Y, Z in mm):")
    for i, point in enumerate(points_0deg[0]):
        print(f"  Point {i}: ({point[0]:6.1f}, {point[1]:6.1f}, {point[2]:6.1f})")

    # Test numpy broadcasting
    print("\nTesting numpy broadcasting:")
    z_rot_test = np.array([[0, 0, 1000, 1000, 0]])  # (1,5)
    B_test = np.array([[0, -1, 0]])  # (1,3)
    print(f"z_rot shape: {z_rot_test.shape}")
    print(f"B shape: {B_test.shape}")
    z_rot_expanded = z_rot_test[:, :, None]  # (1,5,1)
    B_expanded = B_test[:, None, :]  # (1,1,3)
    print(f"z_rot_expanded shape: {z_rot_expanded.shape}")
    print(f"B_expanded shape: {B_expanded.shape}")
    result = z_rot_expanded * B_expanded
    print(f"result shape: {result.shape}")
    print(f"result:\n{result}")
    print(f"result[0,2]: {result[0,2]} (should be [0, -1000, 0])")
    points_90deg = axis.embed_section_points_world(stations_mm[:1], yz_points, rotation_deg=90.0)
    print(f"\n90° rotation result shape: {points_90deg.shape}")
    print("90° rotation points (X, Y, Z in mm):")
    for i, point in enumerate(points_90deg[0]):
        print(f"  Point {i}: ({point[0]:6.1f}, {point[1]:6.1f}, {point[2]:6.1f})")

    # Test 180° rotation
    points_180deg = axis.embed_section_points_world(stations_mm[:1], yz_points, rotation_deg=180.0)
    print(f"\n180° rotation result shape: {points_180deg.shape}")
    print("180° rotation points (X, Y, Z in mm):")
    for i, point in enumerate(points_180deg[0]):
        print(f"  Point {i}: ({point[0]:6.1f}, {point[1]:6.1f}, {point[2]:6.1f})")

    # Verify that 90° rotation is correct
    print("\n=== Verification ===")

    # For a 90° rotation in YZ plane:
    # (y, z) -> (-z, y)
    expected_90deg_yz = np.array([
        [0.0, 0.0],        # (0, 0) -> (0, 0)
        [0.0, 1000.0],     # (1000, 0) -> (0, 1000)
        [-1000.0, 1000.0], # (1000, 1000) -> (-1000, 1000)
        [-1000.0, 0.0],    # (0, 1000) -> (-1000, 0)
        [0.0, 0.0]         # back to origin
    ])

    print("Expected 90° rotation in YZ plane:")
    for i, (y, z) in enumerate(expected_90deg_yz):
        print(f"  Point {i}: ({y:6.1f}, {z:6.1f})")

    # Extract Y and Z coordinates from the embedded points
    actual_90deg_yz = points_90deg[0, :, 1:]  # Skip X coordinate, take Y and Z

    print("\nActual 90° rotation YZ coordinates:")
    for i, (y, z) in enumerate(actual_90deg_yz):
        print(f"  Point {i}: ({y:6.1f}, {z:6.1f})")

    # Check if they match (within tolerance)
    diff = np.abs(actual_90deg_yz - expected_90deg_yz)
    max_diff = np.max(diff)
    print(f"\nMaximum difference from expected: {max_diff:.6f} mm")

    if max_diff < 1e-6:
        print("✅ 90° rotation is mathematically correct!")
    else:
        print("❌ 90° rotation does not match expected result!")

    # Test that 180° rotation gives expected result
    expected_180deg_yz = np.array([
        [0.0, 0.0],        # (0, 0) -> (0, 0)
        [-1000.0, 0.0],    # (1000, 0) -> (-1000, 0)
        [-1000.0, -1000.0], # (1000, 1000) -> (-1000, -1000)
        [0.0, -1000.0],    # (0, 1000) -> (0, -1000)
        [0.0, 0.0]         # back to origin
    ])

    actual_180deg_yz = points_180deg[0, :, 1:]
    diff_180 = np.abs(actual_180deg_yz - expected_180deg_yz)
    max_diff_180 = np.max(diff_180)

    print(f"\n180° rotation max difference: {max_diff_180:.6f} mm")
    if max_diff_180 < 1e-6:
        print("✅ 180° rotation is mathematically correct!")
    else:
        print("❌ 180° rotation does not match expected result!")

if __name__ == "__main__":
    test_rotation_math()