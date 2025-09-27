#!/usr/bin/env python3
"""
Smoke test for axis frame functions.
Tests the most important invariants for refactoring safety:
- Frame functions produce unit-norm tangents
- No NaN values in output
- Correct shapes
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.axis import Axis


def test_axis_frame_invariants():
    """Test basic axis frame function invariants."""
    print("=== Testing Axis Frame Invariants ===")
    
    # Build a tiny synthetic axis - simple 3-point straight line
    stations = [0.0, 1000.0, 2000.0]  # 3 stations, 1m apart 
    x_coords = [0.0, 1000.0, 2000.0]  # straight line along X
    y_coords = [0.0, 0.0, 0.0]        # no Y movement
    z_coords = [0.0, 0.0, 0.0]        # no Z movement
    
    axis = Axis(stations, x_coords, y_coords, z_coords, units="mm")
    
    print(f"Created axis with {len(axis)} stations")
    assert len(axis) == 3, f"Expected 3 stations, got {len(axis)}"
    
    # Test stations to sample at
    test_stations = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
    
    # Test both parallel transport and symmetric modes
    modes = ["pt", "symmetric"]
    
    for mode in modes:
        print(f"\nTesting {mode} mode:")
        
        # Call frame_at_stations 
        result = axis.frame_at_stations(test_stations, mode=mode)
        
        # Check result structure
        expected_keys = {"stations_mm", "P", "T", "N", "B", "N_yaw", "B_yaw", "plan_rotation_deg", "mode"}
        assert set(result.keys()) == expected_keys, f"Missing keys in {mode} result: {expected_keys - set(result.keys())}"
        
        P = result["P"]
        T = result["T"] 
        N = result["N"]
        B = result["B"]
        
        # Check shapes
        expected_shape = (len(test_stations), 3)
        assert P.shape == expected_shape, f"Position shape {P.shape} != {expected_shape}"
        assert T.shape == expected_shape, f"Tangent shape {T.shape} != {expected_shape}"
        assert N.shape == expected_shape, f"Normal shape {N.shape} != {expected_shape}"
        assert B.shape == expected_shape, f"Binormal shape {B.shape} != {expected_shape}"
        
        # Check no NaNs
        assert not np.any(np.isnan(P)), f"NaN found in positions for {mode} mode"
        assert not np.any(np.isnan(T)), f"NaN found in tangents for {mode} mode"
        assert not np.any(np.isnan(N)), f"NaN found in normals for {mode} mode"
        assert not np.any(np.isnan(B)), f"NaN found in binormals for {mode} mode"
        
        # Check unit-norm tangents
        T_norms = np.linalg.norm(T, axis=1)
        assert np.allclose(T_norms, 1.0, atol=1e-10), f"Tangents not unit norm in {mode} mode: {T_norms}"
        
        # Check unit-norm normals and binormals
        N_norms = np.linalg.norm(N, axis=1)
        B_norms = np.linalg.norm(B, axis=1)
        assert np.allclose(N_norms, 1.0, atol=1e-10), f"Normals not unit norm in {mode} mode: {N_norms}"
        assert np.allclose(B_norms, 1.0, atol=1e-10), f"Binormals not unit norm in {mode} mode: {B_norms}"
        
        # Check orthogonality (T·N = 0, T·B = 0, N·B = 0)
        TN_dots = np.sum(T * N, axis=1)
        TB_dots = np.sum(T * B, axis=1)
        NB_dots = np.sum(N * B, axis=1)
        
        assert np.allclose(TN_dots, 0.0, atol=1e-10), f"T·N not zero in {mode} mode: {TN_dots}"
        assert np.allclose(TB_dots, 0.0, atol=1e-10), f"T·B not zero in {mode} mode: {TB_dots}"
        assert np.allclose(NB_dots, 0.0, atol=1e-10), f"N·B not zero in {mode} mode: {NB_dots}"
        
        print(f"  ✓ Shapes correct")
        print(f"  ✓ No NaNs found")
        print(f"  ✓ Tangents unit norm (max error: {np.max(np.abs(T_norms - 1.0)):.2e})")
        print(f"  ✓ Normals unit norm (max error: {np.max(np.abs(N_norms - 1.0)):.2e})")
        print(f"  ✓ Binormals unit norm (max error: {np.max(np.abs(B_norms - 1.0)):.2e})")
        print(f"  ✓ Orthogonality preserved (max dot: {np.max([np.max(np.abs(TN_dots)), np.max(np.abs(TB_dots)), np.max(np.abs(NB_dots))]):.2e})")
    
    # Test single station convenience wrapper
    print(f"\nTesting frame_at_station convenience wrapper:")
    single_result = axis.frame_at_station(1000.0, mode="pt")
    
    # Should have same keys but scalar values for some
    assert isinstance(single_result["P"], np.ndarray) and single_result["P"].shape == (3,), "Single station position should be (3,) array"
    assert isinstance(single_result["T"], np.ndarray) and single_result["T"].shape == (3,), "Single station tangent should be (3,) array"
    
    print(f"  ✓ Single station wrapper works correctly")
    
    print("=== Axis Frame Tests PASSED ===")


if __name__ == "__main__":
    try:
        test_axis_frame_invariants()
        print("\n🎉 All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)