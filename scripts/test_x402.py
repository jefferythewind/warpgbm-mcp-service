#!/usr/bin/env python3
"""
Test X402 payment verification

Usage:
    python scripts/test_x402.py                              # Test with mock tx
    python scripts/test_x402.py --tx-hash 0x123abc...        # Test with real tx
    python scripts/test_x402.py --service-url https://...    # Test production
"""

import argparse
import requests
import sys
from typing import Optional


def test_payment_verification(
    service_url: str = "http://localhost:4000",
    tx_hash: Optional[str] = None
):
    """Test the X402 payment verification flow"""
    
    print("=" * 60)
    print("X402 Payment Verification Test")
    print("=" * 60)
    print()
    
    # Use mock tx hash if none provided
    if not tx_hash:
        tx_hash = "0x" + "a" * 64  # Mock transaction hash
        print(f"⚠️  Using MOCK transaction hash (dev mode)")
    else:
        print(f"✓ Using REAL transaction hash")
    
    print(f"Service URL: {service_url}")
    print(f"TX Hash: {tx_hash}")
    print()
    
    # Step 1: Verify payment
    print("Step 1: Verifying payment...")
    try:
        response = requests.post(
            f"{service_url}/x402/verify",
            json={"tx_hash": tx_hash},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Verification failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        if data.get("status") != "paid":
            print(f"❌ Payment not verified: {data}")
            return False
        
        token = data.get("token")
        expires_in = data.get("expires_in")
        
        print(f"✓ Payment verified!")
        print(f"  Token expires in: {expires_in} seconds")
        print(f"  Token: {token[:50]}...")
        print()
        
    except Exception as e:
        print(f"❌ Error verifying payment: {e}")
        return False
    
    # Step 2: Check status without token
    print("Step 2: Checking status (no token)...")
    try:
        response = requests.get(f"{service_url}/x402/status", timeout=10)
        data = response.json()
        print(f"  Authenticated: {data.get('authenticated')}")
        print()
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print()
    
    # Step 3: Check status with token
    print("Step 3: Checking status (with token)...")
    try:
        response = requests.get(
            f"{service_url}/x402/status",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        data = response.json()
        print(f"✓ Authenticated: {data.get('authenticated')}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Step 4: Make a paid request (training)
    print("Step 4: Testing paid request (training)...")
    try:
        response = requests.post(
            f"{service_url}/train",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                "y": [0, 0, 1, 1],
                "model_type": "lightgbm",
                "objective": "binary",
                "n_estimators": 10
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"⚠️  Training request returned: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            print()
        else:
            data = response.json()
            print(f"✓ Training successful!")
            print(f"  Model type: {data.get('model_type')}")
            print(f"  Artifact ID: {data.get('artifact_id')}")
            print(f"  Training time: {data.get('training_time_seconds')}s")
            print()
    
    except Exception as e:
        print(f"⚠️  Error training: {e}")
        print()
    
    # Summary
    print("=" * 60)
    print("✓ X402 Payment Flow Test Complete!")
    print("=" * 60)
    print()
    print("Your token (valid for 1 hour):")
    print(token)
    print()
    print("Use it in requests like:")
    print(f"  curl -H 'Authorization: Bearer {token}' \\")
    print(f"       {service_url}/train -d '{{...}}'")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test X402 payment verification")
    parser.add_argument(
        "--tx-hash",
        help="Real transaction hash (0x...)",
        default=None
    )
    parser.add_argument(
        "--service-url",
        help="Service URL",
        default="http://localhost:4000"
    )
    
    args = parser.parse_args()
    
    success = test_payment_verification(
        service_url=args.service_url,
        tx_hash=args.tx_hash
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

