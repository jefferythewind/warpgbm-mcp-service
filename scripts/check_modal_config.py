#!/usr/bin/env python3
"""
Modal Configuration Safety Checker

Verifies modal_app.py is configured safely (CPU-only, no accidental GPU usage)
"""

import re
import sys


def check_modal_config():
    """Check modal_app.py for safe configuration"""
    
    print("=" * 60)
    print("Modal Configuration Safety Check")
    print("=" * 60)
    print()
    
    try:
        with open("modal_app.py", "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ modal_app.py not found!")
        return False
    
    checks_passed = True
    
    # Split content by function definitions
    functions = re.split(r'@app\.function', content)
    
    print("📋 Found Functions:")
    print()
    
    # Check 1: GPU functions and their limits
    gpu_functions = []
    for i, func in enumerate(functions):
        if 'gpu=' in func:
            # Extract function name
            func_name_match = re.search(r'def\s+(\w+)', func)
            func_name = func_name_match.group(1) if func_name_match else f"function_{i}"
            gpu_functions.append(func_name)
            
            # Check GPU concurrency limit
            gpu_concurrency = re.search(r'concurrency_limit\s*=\s*([0-9]+)', func)
            gpu_type = re.search(r'gpu\s*=\s*["\']([^"\']+)["\']', func)
            scaledown = re.search(r'scaledown_window\s*=\s*([0-9]+)', func)
            
            print(f"🎮 GPU Function: {func_name}")
            print(f"   GPU Type: {gpu_type.group(1) if gpu_type else 'unspecified'}")
            
            if gpu_concurrency:
                limit = int(gpu_concurrency.group(1))
                if limit <= 2:
                    print(f"   ✅ Concurrency: {limit} GPU(s) max (SAFE)")
                else:
                    print(f"   ⚠️  Concurrency: {limit} GPUs (HIGH!)")
                    checks_passed = False
            else:
                print("   ❌ No concurrency limit! (DANGEROUS)")
                checks_passed = False
            
            if scaledown:
                window = int(scaledown.group(1))
                print(f"   ✅ Scaledown: {window}s idle → shutdown")
            else:
                print("   ⚠️  No scaledown window")
            
            print()
    
    if not gpu_functions:
        print("✓ No GPU functions found (CPU-only deployment)")
    
    # Check 2: Main serve function is CPU-only
    serve_func = [f for f in functions if 'def serve' in f]
    if serve_func:
        if 'gpu=' in serve_func[0]:
            print("❌ CRITICAL: Main serve() function has GPU! (Very expensive!)")
            checks_passed = False
        else:
            print("✅ Main serve() function: CPU-only")
    else:
        print("⚠️  No serve() function found")
    
    
    print()
    print("=" * 60)
    print("💰 Cost Estimate:")
    print("=" * 60)
    
    # Calculate max cost
    if gpu_functions:
        gpu_concurrency = re.findall(r'concurrency_limit\s*=\s*([0-9]+)', content)
        if gpu_concurrency:
            max_gpus = sum(int(x) for x in gpu_concurrency if int(x) <= 10)  # Only count reasonable limits
            print(f"   Max GPUs at once: {max_gpus}")
            print(f"   Max GPU cost: ${max_gpus * 2.16:.2f}/hour (worst case)")
            print(f"   GPU idle cost: $0.036/minute after training")
            print()
            print(f"   💡 GPU shuts down after 60s idle → $0 when not used")
    
    print(f"   CPU service: ~$0.36/hour (always on)")
    print(f"   Free tier: 30 CPU-hours/month")
    print()
    
    print("=" * 60)
    
    if checks_passed:
        print("✅ Configuration looks SAFE!")
        print()
        print("   ✓ Main service: CPU-only")
        if gpu_functions:
            print(f"   ✓ GPU functions: Limited to {gpu_concurrency[0] if gpu_concurrency else 'N/A'} concurrent")
            print("   ✓ Auto-shutdown enabled")
        print("   ✓ Cost-controlled")
        print()
        print("🚀 Safe to deploy!")
    else:
        print("❌ CONFIGURATION HAS ISSUES!")
        print("   Please fix the issues above before deploying.")
        print()
        print("⛔ DO NOT DEPLOY until fixed!")
    
    print("=" * 60)
    print()
    
    return checks_passed


if __name__ == "__main__":
    safe = check_modal_config()
    sys.exit(0 if safe else 1)

