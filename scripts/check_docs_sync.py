#!/usr/bin/env python3
"""
SPOT VISO Docs Sync Checker

This script scans ARCH_MAP.md and CALL_GRAPH.md for function signatures and verifies
they still match the actual function signatures in the codebase.

Usage:
    python scripts/check_docs_sync.py [--fix] [--verbose]

Exit codes:
    0 - All signatures match
    1 - Signature mismatches found  
    2 - Script error (file not found, parse error, etc.)
"""

import ast
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FunctionSignature:
    """Represents a function signature for comparison."""
    name: str
    module: str
    signature: str
    line_number: Optional[int] = None
    
    def __str__(self):
        return f"{self.module}:{self.name}() -> {self.signature}"


def extract_code_signatures(project_root: Path) -> Dict[str, List[FunctionSignature]]:
    """Extract actual function signatures from Python source code."""
    signatures = {}
    models_path = project_root / "models"
    
    if not models_path.exists():
        print(f"ERROR: Models directory not found: {models_path}")
        return {}
    
    for py_file in models_path.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            module_sigs = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    # Build argument list
                    args = []
                    
                    # Regular arguments
                    for arg in node.args.args:
                        if arg.annotation:
                            args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
                        else:
                            args.append(arg.arg)
                    
                    # Handle defaults
                    if node.args.defaults:
                        num_defaults = len(node.args.defaults)
                        for i, default in enumerate(node.args.defaults):
                            arg_index = len(args) - num_defaults + i
                            if 0 <= arg_index < len(args):
                                args[arg_index] += f" = {ast.unparse(default)}"
                    
                    # Build signature string
                    sig_str = f"def {node.name}({', '.join(args)})"
                    if node.returns:
                        sig_str += f" -> {ast.unparse(node.returns)}"
                    
                    module_sigs.append(FunctionSignature(
                        name=node.name,
                        module=f"models/{py_file.name}",
                        signature=sig_str,
                        line_number=node.lineno
                    ))
            
            if module_sigs:
                signatures[f"models/{py_file.name}"] = module_sigs
                
        except Exception as e:
            print(f"WARNING: Failed to parse {py_file}: {e}")
    
    return signatures


def extract_doc_signatures(doc_file: Path) -> List[FunctionSignature]:
    """Extract function signatures mentioned in documentation."""
    if not doc_file.exists():
        print(f"WARNING: Documentation file not found: {doc_file}")
        return []
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Cannot read {doc_file}: {e}")
        return []
    
    signatures = []
    current_module = None
    
    # Pattern to match module headers like "### models/axis.py"
    module_pattern = re.compile(r'###?\s+models/(\w+\.py)')
    # Pattern to match function definitions in code blocks
    func_pattern = re.compile(r'^def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?', re.MULTILINE)
    # Pattern to match inline function references like "models/axis.py:function_name(...)"
    inline_pattern = re.compile(r'models/(\w+\.py):(\w+)\(([^)]*)\)(?:\s*->\s*([^,\n]+))?')
    
    lines = content.split('\n')
    in_code_block = False
    
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Check for module headers
        module_match = module_pattern.search(line_stripped)
        if module_match:
            current_module = f"models/{module_match.group(1)}"
            continue
        
        # Track code blocks
        if line_stripped.startswith('```'):
            if line_stripped.startswith('```python'):
                in_code_block = True
            else:
                in_code_block = False
            continue
        
        # Extract function signatures from code blocks
        if in_code_block and current_module and line_stripped.startswith('def '):
            func_match = func_pattern.match(line_stripped)
            if func_match:
                signatures.append(FunctionSignature(
                    name=func_match.group(1),
                    module=current_module,
                    signature=line_stripped,
                    line_number=line_num
                ))
        
        # Extract inline function references from call graphs
        for inline_match in inline_pattern.finditer(line):
            module_name = f"models/{inline_match.group(1)}"
            func_name = inline_match.group(2)
            args = inline_match.group(3)
            return_type = inline_match.group(4)
            
            # Reconstruct signature
            sig = f"def {func_name}({args})"
            if return_type:
                sig += f" -> {return_type.strip()}"
            
            signatures.append(FunctionSignature(
                name=func_name,
                module=module_name,
                signature=sig,
                line_number=line_num
            ))
    
    return signatures


def normalize_signature(sig: str) -> str:
    """Normalize a function signature for comparison."""
    # Remove extra whitespace
    sig = re.sub(r'\s+', ' ', sig.strip())
    
    # Normalize Optional[] syntax variations
    sig = re.sub(r'Optional\[([^]]+)\]', r'\1 | None', sig)
    sig = re.sub(r'Union\[([^,]+),\s*None\]', r'\1 | None', sig)
    
    # Normalize typing imports
    sig = re.sub(r'\btyping\.', '', sig)
    
    return sig


def compare_signatures(code_sigs: Dict[str, List[FunctionSignature]], 
                      doc_sigs: List[FunctionSignature]) -> List[str]:
    """Compare code signatures with documentation signatures and return mismatches."""
    mismatches = []
    
    # Build lookup for code signatures
    code_lookup = {}
    for module, sigs in code_sigs.items():
        for sig in sigs:
            key = f"{module}:{sig.name}"
            code_lookup[key] = sig
    
    # Check each documented signature
    for doc_sig in doc_sigs:
        key = f"{doc_sig.module}:{doc_sig.name}"
        
        if key not in code_lookup:
            mismatches.append(f"MISSING: Function {key} documented but not found in code")
            continue
        
        code_sig = code_lookup[key]
        
        # Normalize both signatures for comparison
        doc_norm = normalize_signature(doc_sig.signature)
        code_norm = normalize_signature(code_sig.signature)
        
        if doc_norm != code_norm:
            mismatches.append(f"MISMATCH: {key}")
            mismatches.append(f"  Doc:  {doc_norm}")
            mismatches.append(f"  Code: {code_norm} (line {code_sig.line_number})")
    
    return mismatches


def main():
    parser = argparse.ArgumentParser(description='Check SPOT VISO documentation synchronization')
    parser.add_argument('--fix', action='store_true', 
                       help='Update documentation with current signatures (not implemented)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    parser.add_argument('--project-root', type=Path, default=Path('.'),
                       help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = args.project_root.resolve()
    if args.verbose:
        print(f"Checking project at: {project_root}")
    
    # Check if we're in the right directory
    if not (project_root / "models").exists():
        print("ERROR: Not in SPOT VISO project root (models/ directory not found)")
        return 2
    
    # Extract signatures from code
    if args.verbose:
        print("Extracting signatures from source code...")
    
    code_signatures = extract_code_signatures(project_root)
    
    if not code_signatures:
        print("ERROR: No function signatures found in source code")
        return 2
    
    if args.verbose:
        total_funcs = sum(len(sigs) for sigs in code_signatures.values())
        print(f"Found {total_funcs} public functions in {len(code_signatures)} modules")
    
    # Check documentation files
    doc_files = [
        project_root / "ARCH_MAP.md",
        project_root / "CALL_GRAPH.md"
    ]
    
    all_mismatches = []
    
    for doc_file in doc_files:
        if args.verbose:
            print(f"Checking documentation: {doc_file.name}")
        
        doc_signatures = extract_doc_signatures(doc_file)
        
        if not doc_signatures:
            print(f"WARNING: No function signatures found in {doc_file.name}")
            continue
        
        if args.verbose:
            print(f"Found {len(doc_signatures)} documented functions in {doc_file.name}")
        
        mismatches = compare_signatures(code_signatures, doc_signatures)
        
        if mismatches:
            all_mismatches.extend([f"In {doc_file.name}:"] + mismatches + [""])
    
    # Report results
    if all_mismatches:
        print("DOCS DRIFT DETECTED!")
        print("=" * 50)
        for line in all_mismatches:
            print(line)
        
        print("Please update the documentation files to match current function signatures.")
        return 1
    
    else:
        print("✅ All documented function signatures are up to date!")
        if args.verbose:
            total_checked = sum(len(extract_doc_signatures(f)) for f in doc_files)
            print(f"Validated {total_checked} function signatures across {len(doc_files)} documentation files.")
        return 0


if __name__ == "__main__":
    sys.exit(main())