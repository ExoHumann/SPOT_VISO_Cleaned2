#!/usr/bin/env python3
"""
Tests for SPOT VISO CLI commands
"""
import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd: list) -> tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def test_cli_help():
    """Test that CLI shows help."""
    exit_code, stdout, stderr = run_command(['python', 'spotviso.py', '--help'])
    assert exit_code == 0, f"Help command failed: {stderr}"
    assert 'SPOT VISO' in stdout, "Help should mention SPOT VISO"
    assert 'check' in stdout, "Help should mention check command"
    print("✅ CLI help test passed")


def test_check_command():
    """Test the check command."""
    exit_code, stdout, stderr = run_command(['python', 'spotviso.py', 'check'])
    
    # Should pass if we have valid data, or warn if not
    assert exit_code in [0, 1], f"Check command should return 0 or 1, got {exit_code}"
    assert '🔍 SPOT VISO Data Validation' in stdout, "Check should show validation header"
    print("✅ Check command test passed")


def test_smoke_test():
    """Test the smoke test command."""
    exit_code, stdout, stderr = run_command(['python', 'spotviso.py', 'test', '-m', 'smoke'])
    
    # Should complete quickly
    assert exit_code in [0, 1], f"Smoke test should return 0 or 1, got {exit_code}"
    assert '🧪 SPOT VISO Smoke Tests' in stdout, "Smoke test should show test header"
    assert 'completed in' in stdout, "Should show completion time"
    print("✅ Smoke test command test passed")


def test_viz_command():
    """Test the visualization command."""
    # Clean up any existing output
    output_file = "spotviso_viz_demo.html"
    if os.path.exists(output_file):
        os.remove(output_file)
    
    exit_code, stdout, stderr = run_command(['python', 'spotviso.py', 'viz', '--case', 'demo'])
    
    assert exit_code == 0, f"Viz command failed: {stderr}"
    assert '📊 SPOT VISO Visualization' in stdout, "Viz should show visualization header"
    assert os.path.exists(output_file), "Viz should create output file"
    
    # Check that HTML file has basic structure
    with open(output_file, 'r') as f:
        content = f.read()
        assert '<html>' in content, "Output should be valid HTML"
        assert 'SPOT VISO' in content, "HTML should mention SPOT VISO"
    
    # Clean up
    os.remove(output_file)
    print("✅ Viz command test passed")


def test_alternate_data_path():
    """Test commands with alternate data path."""
    # Test with a path that might not exist
    exit_code, stdout, stderr = run_command(['python', 'spotviso.py', 'check', '--data-path', 'nonexistent'])
    
    # Should fail gracefully
    assert exit_code == 1, "Check with nonexistent path should fail"
    assert 'does not exist' in stdout, "Should report missing path"
    print("✅ Alternate data path test passed")


def main():
    """Run all tests."""
    print("🧪 Running SPOT VISO CLI tests...")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        test_cli_help()
        test_check_command()
        test_smoke_test()
        test_viz_command()
        test_alternate_data_path()
        
        print("\n" + "=" * 40)
        print("✅ All CLI tests passed!")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())