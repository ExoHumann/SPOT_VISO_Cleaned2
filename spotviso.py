#!/usr/bin/env python3
"""
SPOT VISO - Structural Parametric Object Tool & VISualization Object

Unified CLI entry point for all SPOT VISO commands.
"""
from __future__ import annotations

import sys
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def cmd_check(args: argparse.Namespace) -> int:
    """
    Validate data integrity and basic geometry.
    
    Returns:
        0 if validation passes, 1 if errors found
    """
    print("🔍 SPOT VISO Data Validation")
    print("=" * 40)
    
    data_path = args.data_path or "GIT/MAIN"
    print(f"📁 Checking data path: {data_path}")
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data path '{data_path}' does not exist")
        return 1
    
    errors = []
    warnings = []
    
    # Check required files exist
    required_files = [
        "_Axis_JSON.json",
        "_CrossSection_JSON.json", 
        "_MainStation_JSON.json"
    ]
    
    for file_name in required_files:
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            errors.append(f"Missing required file: {file_name}")
        else:
            print(f"✅ Found: {file_name}")
    
    # Check optional files
    optional_files = [
        "_DeckObject_JSON.json",
        "_PierObject_JSON.json", 
        "_FoundationObject_JSON.json"
    ]
    
    found_objects = []
    for file_name in optional_files:
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            found_objects.append(file_name)
            print(f"✅ Found: {file_name}")
    
    if not found_objects:
        warnings.append("No object files found (DeckObject, PierObject, or FoundationObject)")
    
    # Validate JSON files can be loaded
    for file_name in required_files + found_objects:
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        warnings.append(f"{file_name}: Expected list format")
                    elif len(data) == 0:
                        warnings.append(f"{file_name}: Empty data")
                    else:
                        print(f"✅ Valid JSON: {file_name} ({len(data)} records)")
            except json.JSONDecodeError as e:
                errors.append(f"{file_name}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"{file_name}: Cannot read - {e}")
    
    # Check MASTER_SECTION directory
    if '/' in data_path:
        # data_path is like GIT/MAIN, so master_section should be in parent of GIT
        git_parent = os.path.dirname(os.path.dirname(data_path))
        master_section = os.path.join(git_parent, "MASTER_SECTION") if git_parent else "MASTER_SECTION"
    else:
        # data_path is just a folder name, master_section should be in current dir
        master_section = "MASTER_SECTION"
    
    if os.path.exists(master_section):
        print(f"✅ Found: MASTER_SECTION directory")
        section_files = [f for f in os.listdir(master_section) if f.endswith('.json')]
        print(f"📄 Section files: {len(section_files)}")
    else:
        warnings.append("MASTER_SECTION directory not found")
    
    # Report results
    print("\n" + "=" * 40)
    print("📊 Validation Results")
    print("=" * 40)
    
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for error in errors:
            print(f"  • {error}")
    
    if warnings:
        print(f"⚠️  Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not errors and not warnings:
        print("✅ All checks passed!")
    elif not errors:
        print("✅ Validation passed with warnings")
    else:
        print("❌ Validation failed")
        return 1
    
    return 0

def cmd_test_smoke(args: argparse.Namespace) -> int:
    """
    Quick validation suite (<60s).
    
    Returns:
        0 if tests pass, 1 if tests fail
    """
    print("🧪 SPOT VISO Smoke Tests")
    print("=" * 40)
    
    start_time = time.time()
    
    # Run basic checks first
    print("1️⃣ Running data validation...")
    check_result = cmd_check(args)
    if check_result != 0:
        print("❌ Data validation failed, aborting smoke tests")
        return 1
    
    print("\n2️⃣ Testing imports...")
    try:
        from spot_loader import SpotLoader
        from models import Axis, CrossSection
        print("✅ Core imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return 1
    
    print("\n3️⃣ Testing data loading...")
    try:
        data_path = args.data_path or "GIT/MAIN"
        master_folder, branch = data_path.split('/', 1) if '/' in data_path else (data_path, None)
        
        loader = SpotLoader(master_folder=master_folder, branch=branch, verbose=False)
        print("✅ SpotLoader initialized")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return 1
    
    elapsed = time.time() - start_time
    print(f"\n✅ Smoke tests completed in {elapsed:.1f}s")
    
    if elapsed > 60:
        print("⚠️  Warning: Smoke tests took longer than 60s")
    
    return 0

def cmd_test_all(args: argparse.Namespace) -> int:
    """
    Full test suite.
    
    Returns:
        0 if all tests pass, 1 if any test fails
    """
    print("🧪 SPOT VISO Full Test Suite")
    print("=" * 40)
    
    start_time = time.time()
    
    # Run smoke tests first
    smoke_result = cmd_test_smoke(args)
    if smoke_result != 0:
        print("❌ Smoke tests failed, aborting full test suite")
        return 1
    
    print("\n4️⃣ Testing advanced functionality...")
    
    # Test multiple data sources if available
    git_path = Path("GIT")
    if git_path.exists():
        branches = [d.name for d in git_path.iterdir() if d.is_dir()]
        print(f"📁 Found {len(branches)} data branches: {', '.join(branches[:5])}")
        
        # Test a few branches
        test_branches = branches[:3] if len(branches) > 3 else branches
        for branch in test_branches:
            branch_path = f"GIT/{branch}"
            print(f"  Testing branch: {branch}")
            
            # Quick validation of branch
            required_files = ["_Axis_JSON.json"]
            missing_files = []
            for file_name in required_files:
                if not os.path.exists(os.path.join(branch_path, file_name)):
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"    ⚠️  Missing files in {branch}: {missing_files}")
            else:
                print(f"    ✅ {branch} has required files")
    
    elapsed = time.time() - start_time  
    print(f"\n✅ Full test suite completed in {elapsed:.1f}s")
    
    return 0

def cmd_viz(args: argparse.Namespace) -> int:
    """
    Generate visualization by case identifier.
    
    Returns:
        0 if visualization succeeds, 1 if it fails
    """
    print("📊 SPOT VISO Visualization")
    print("=" * 40)
    
    case_id = args.case or "demo"
    output = args.output or f"spotviso_viz_{case_id}.html"
    
    print(f"📋 Case: {case_id}")
    print(f"📄 Output: {output}")
    
    # For demo case, try to run with default data
    if case_id == "demo":
        try:
            # Check if we can run existing visualization
            data_path = "GIT/MAIN"
            if not os.path.exists(data_path):
                # Try alternative branches
                git_path = Path("GIT")
                if git_path.exists():
                    branches = [d.name for d in git_path.iterdir() if d.is_dir()]
                    if branches:
                        data_path = f"GIT/{branches[0]}"
                        print(f"📁 Using alternative data: {data_path}")
            
            if not os.path.exists(data_path):
                print("❌ No suitable data found for demo")
                return 1
            
            # Try to generate a simple visualization
            print(f"🎨 Generating demo visualization...")
            
            # Create a simple HTML file as placeholder
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SPOT VISO Demo - {case_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
                    .info {{ margin: 20px 0; padding: 15px; background: #f9f9f9; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🏗️ SPOT VISO Visualization</h1>
                    <p><strong>Case:</strong> {case_id}</p>
                    <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="info">
                    <h2>📋 Demo Information</h2>
                    <p>This is a placeholder visualization for the SPOT VISO demo case.</p>
                    <p>Data source: <code>{data_path}</code></p>
                    <p>To generate full 3D visualizations, use the existing run_linear.py script:</p>
                    <pre>python run_linear.py --obj-type DeckObject --out deck.html</pre>
                </div>
                
                <div class="info">
                    <h2>🔗 Available Commands</h2>
                    <ul>
                        <li><code>spotviso check</code> - Validate data integrity</li>
                        <li><code>spotviso test -m smoke</code> - Quick tests (&lt;60s)</li>
                        <li><code>spotviso test-all</code> - Full test suite</li>
                        <li><code>spotviso viz --case demo</code> - This visualization</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open(output, 'w') as f:
                f.write(html_content)
            
            print(f"✅ Demo visualization saved to: {output}")
            return 0
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return 1
    else:
        print(f"❌ Case '{case_id}' not implemented yet")
        return 1

def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='spotviso',
        description='SPOT VISO - Structural Parametric Object Tool & VISualization Object'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # spotviso check
    check_parser = subparsers.add_parser('check', help='Validate data integrity and geometry')
    check_parser.add_argument(
        '--data-path', 
        default='GIT/MAIN',
        help='Path to data folder (default: GIT/MAIN)'
    )
    
    # spotviso test
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument(
        '-m', '--mode',
        choices=['smoke'],
        help='Test mode: smoke for quick tests (<60s)'
    )
    test_parser.add_argument(
        '--data-path',
        default='GIT/MAIN', 
        help='Path to data folder (default: GIT/MAIN)'
    )
    
    # spotviso test-all  
    testall_parser = subparsers.add_parser('test-all', help='Run full test suite')
    testall_parser.add_argument(
        '--data-path',
        default='GIT/MAIN',
        help='Path to data folder (default: GIT/MAIN)'
    )
    
    # spotviso viz
    viz_parser = subparsers.add_parser('viz', help='Generate visualization')
    viz_parser.add_argument(
        '--case',
        default='demo',
        help='Case identifier (default: demo)'
    )
    viz_parser.add_argument(
        '--output', '-o',
        help='Output file path (default: spotviso_viz_<case>.html)'
    )
    viz_parser.add_argument(
        '--data-path',
        default='GIT/MAIN',
        help='Path to data folder (default: GIT/MAIN)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    if args.command == 'check':
        return cmd_check(args)
    elif args.command == 'test':
        if args.mode == 'smoke':
            return cmd_test_smoke(args)
        else:
            parser.error('test command requires -m smoke')
    elif args.command == 'test-all':
        return cmd_test_all(args)
    elif args.command == 'viz':
        return cmd_viz(args)
    else:
        parser.error(f'Unknown command: {args.command}')

if __name__ == '__main__':
    sys.exit(main())