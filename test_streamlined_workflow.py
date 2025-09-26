#!/usr/bin/env python3
"""
Test script for the streamlined LinearObject workflow.
Tests both file-based and SPOT folder-based loading.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_file_based_loading():
    """Test the file-based loading approach."""
    print("=== Testing File-based Loading ===")
    
    try:
        from linear_object_builder import LinearObjectBuilder
        
        # Set up file paths using available test data
        axis_json = "GIT/RCZ_new/_Axis_JSON.json"
        cross_json = "GIT/RCZ_new/_CrossSection_JSON.json"
        pier_json = "GIT/RCZ_new/_PierObject_JSON.json"
        mainstation_json = "GIT/RCZ_new/_MainStation_JSON.json"
        section_json = "MASTER_SECTION/SectionData.json"
        
        # Check that files exist
        for path in [axis_json, cross_json, pier_json, mainstation_json, section_json]:
            if not os.path.exists(path):
                print(f"WARNING: Test file not found: {path}")
                return False
        
        print("✓ All test files found")
        
        # Create builder and load data
        builder = LinearObjectBuilder(verbose=True)
        builder.load_from_files(axis_json, cross_json, pier_json, mainstation_json, section_json)
        
        print("✓ Data loaded successfully")
        
        # Try to create a pier object
        pier = builder.create_object("PierObject")
        print(f"✓ Created PierObject: {getattr(pier, 'name', 'Unnamed')}")
        
        # Try to build geometry
        try:
            result = builder.build_geometry(pier, twist_deg=0.0)
            print("✓ Geometry built successfully")
            print(f"  Generated {len(result.get('stations_mm', []))} stations")
            return True
        except Exception as e:
            print(f"✗ Geometry build failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ File-based loading failed: {e}")
        return False

def test_spot_folder_loading():
    """Test the SPOT folder-based loading approach."""
    print("\n=== Testing SPOT Folder Loading ===")
    
    try:
        from spot_linear_integration import create_linear_objects_from_spot
        
        # Test with the GIT/RCZ_new folder structure
        master_folder = "GIT"
        branch = "RCZ_new"
        
        if not os.path.exists(os.path.join(master_folder, branch)):
            print(f"WARNING: SPOT folder not found: {master_folder}/{branch}")
            return False
            
        print(f"✓ SPOT folder found: {master_folder}/{branch}")
        
        # Try to load objects
        objects = create_linear_objects_from_spot(
            master_folder=master_folder,
            branch=branch,
            obj_types=["PierObject"],  # Start with just piers
            verbose=True
        )
        
        if objects.get("PierObject"):
            print(f"✓ Created {len(objects['PierObject'])} PierObjects")
            return True
        else:
            print("✗ No PierObjects created")
            return False
            
    except Exception as e:
        print(f"✗ SPOT folder loading failed: {e}")
        return False

def test_runner_integration():
    """Test that the updated runners work."""
    print("\n=== Testing Runner Integration ===")
    
    try:
        # Test run_linear.py help (should work without errors)
        import subprocess
        result = subprocess.run([
            sys.executable, "run_linear.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✓ run_linear.py help works")
        else:
            print("✗ run_linear.py help failed")
            print(f"Return code: {result.returncode}")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
            
        # Test run_pier.py help
        result = subprocess.run([
            sys.executable, "run_pier.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✓ run_pier.py help works")
            return True
        else:
            print("✗ run_pier.py help failed")
            return False
            
    except Exception as e:
        print(f"✗ Runner integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("SPOT VISO Streamlined Workflow Tests")
    print("=" * 40)
    
    tests = [
        test_file_based_loading,
        test_spot_folder_loading, 
        test_runner_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! Streamlined workflow is working.")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())