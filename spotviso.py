#!/usr/bin/env python3
"""
SPOT VISO CLI - Simple command line interface
Provides test runners and basic validation commands.
"""

import sys
import os
import time
import argparse
import importlib.util
from pathlib import Path

# Add current directory to Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def discover_smoke_tests():
    """Discover smoke tests in tests/smoke/ directory."""
    smoke_dir = Path(__file__).parent / "tests" / "smoke"
    if not smoke_dir.exists():
        return []
    
    test_files = []
    for py_file in smoke_dir.glob("test_*.py"):
        if py_file.name != "__init__.py":
            test_files.append(py_file)
    
    return sorted(test_files)


def run_smoke_tests():
    """Run all smoke tests and return success/failure."""
    print("🔍 Discovering smoke tests...")
    
    test_files = discover_smoke_tests()
    if not test_files:
        print("❌ No smoke tests found in tests/smoke/")
        return False
    
    print(f"Found {len(test_files)} smoke test(s):")
    for test_file in test_files:
        print(f"  • {test_file.name}")
    
    print("\n🚀 Running smoke tests...\n")
    
    start_time = time.time()
    failed_tests = []
    
    for test_file in test_files:
        print(f"▶️  Running {test_file.name}...")
        
        # Run the test file as a subprocess to isolate it
        import subprocess
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=test_file.parent.parent.parent)
        
        if result.returncode == 0:
            print(f"✅ {test_file.name} PASSED\n")
        else:
            print(f"❌ {test_file.name} FAILED")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}\n")
            failed_tests.append(test_file.name)
    
    elapsed = time.time() - start_time
    
    print("="*60)
    print(f"Smoke tests completed in {elapsed:.1f}s")
    
    if failed_tests:
        print(f"❌ {len(failed_tests)} test(s) FAILED:")
        for test_name in failed_tests:
            print(f"   • {test_name}")
        return False
    else:
        print(f"✅ All {len(test_files)} smoke test(s) PASSED")
        return True


def validate_data_integrity():
    """Basic validation of data integrity (placeholder for future implementation)."""
    print("🔍 Data integrity validation not yet implemented")
    print("   This would validate GIT/ data files for basic consistency")
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="spotviso",
        description="SPOT VISO Command Line Interface"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("-m", "--marker", choices=["smoke", "all"], default="all",
                           help="Test marker to run (default: all)")
    
    # Check command (placeholder)
    check_parser = subparsers.add_parser("check", help="Validate data integrity")
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    args = parser.parse_args()
    
    # Dispatch commands
    if args.command == "test":
        if args.marker == "smoke":
            print("SPOT VISO - Running Smoke Tests")
            print("="*40)
            success = run_smoke_tests()
            return 0 if success else 1
        else:
            print("❌ Only 'smoke' marker is currently supported")
            print("   Use: spotviso test -m smoke")
            return 1
    
    elif args.command == "check":
        print("SPOT VISO - Data Integrity Check")
        print("="*40)
        success = validate_data_integrity()
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())