"""
X402 payment verification and token management

Supports:
- On-chain verification (Base/Ethereum USDC transfers)
- API key authentication (for trusted partners)
- Dev mode for testing
"""

import os
import time
from typing import Optional
import jwt
from fastapi import APIRouter, HTTPException, Depends, Header

from app.models import X402VerifyRequest, X402VerifyResponse


# Configuration
SECRET_KEY = os.getenv("X402_JWT_SECRET", "dev-secret-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRY_SECONDS = 3600  # 1 hour

# Payment configuration
WALLET_ADDRESS = os.getenv("X402_WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")
BASE_RPC_URL = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")
PAYMENT_REQUIRED = os.getenv("X402_PAYMENT_REQUIRED", "false").lower() == "true"

# Pricing (in USDC)
PRICING = {
    "train": 0.01,  # $0.01 per training request
    "predict_from_artifact": 0.001,  # $0.001 per prediction
    "predict_proba_from_artifact": 0.001,
}


router = APIRouter(prefix="/x402", tags=["payment"])


def create_access_token(paid: bool = True, expires_delta: int = TOKEN_EXPIRY_SECONDS) -> str:
    """
    Create a JWT access token for paid requests.
    
    Args:
        paid: Whether payment was verified
        expires_delta: Token expiry in seconds
        
    Returns:
        JWT token string
    """
    payload = {
        "paid": paid,
        "exp": time.time() + expires_delta,
        "iat": time.time()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> bool:
    """
    Verify a JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        True if valid and not expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("paid", False) and payload.get("exp", 0) > time.time()
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False


async def verify_payment_optional(authorization: Optional[str] = Header(None)) -> bool:
    """
    Optional dependency: verify payment token if provided.
    
    Returns True if token is valid, False if not provided or invalid.
    Does not raise exceptions (for optional payment).
    """
    if authorization is None:
        return False
    
    # Extract bearer token
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return False
        return verify_token(token)
    except ValueError:
        return False


async def require_payment(authorization: Optional[str] = Header(None)) -> None:
    """
    Required dependency: enforce payment verification.
    
    Raises HTTPException if token is missing or invalid.
    """
    if authorization is None:
        raise HTTPException(
            status_code=402,
            detail="Payment required. Please provide a valid X402 token or API key."
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        if not verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid or expired token")
            
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")


async def verify_on_chain_payment(tx_hash: str, expected_amount: float) -> bool:
    """
    Verify a USDC payment transaction on Base network.
    
    Args:
        tx_hash: Transaction hash (0x...)
        expected_amount: Expected payment in USDC
        
    Returns:
        True if payment is verified
    """
    # TODO: Implement actual on-chain verification
    # This requires web3.py or similar library to:
    # 1. Connect to Base RPC
    # 2. Get transaction details
    # 3. Verify USDC transfer to WALLET_ADDRESS
    # 4. Check amount matches expected_amount
    
    # For now, basic validation
    if not tx_hash or not tx_hash.startswith("0x") or len(tx_hash) != 66:
        return False
    
    # In dev mode, accept any valid-looking tx hash
    if not PAYMENT_REQUIRED:
        return True
    
    # TODO: Add actual verification here
    # Example:
    # from web3 import Web3
    # w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
    # tx = w3.eth.get_transaction(tx_hash)
    # Verify tx.to == WALLET_ADDRESS
    # Verify tx.value >= expected_amount
    
    return False


@router.post("/verify", response_model=X402VerifyResponse)
async def verify_payment(request: X402VerifyRequest):
    """
    Verify payment and issue access token.
    
    **Methods:**
    1. **On-chain**: Provide `tx_hash` of USDC transfer on Base
    2. **API Key**: Provide `api_key` for trusted partners
    3. **Dev Mode**: If PAYMENT_REQUIRED=false, any valid tx_hash works
    
    **Returns:**
    - `status`: "paid" or "unpaid"
    - `token`: JWT Bearer token (if paid)
    - `expires_in`: Token validity in seconds
    """
    # Method 1: On-chain verification
    if request.tx_hash:
        # Determine expected amount based on endpoint
        # For now, use minimum amount (can be enhanced)
        expected_amount = min(PRICING.values())
        
        if await verify_on_chain_payment(request.tx_hash, expected_amount):
            token = create_access_token()
            return X402VerifyResponse(
                status="paid",
                token=token,
                expires_in=TOKEN_EXPIRY_SECONDS,
                message="Payment verified on-chain"
            )
    
    # Method 2: API Key verification (for partners)
    if hasattr(request, "api_key") and request.api_key:
        # TODO: Verify API key against database
        valid_api_keys = os.getenv("X402_API_KEYS", "").split(",")
        if request.api_key in valid_api_keys:
            token = create_access_token()
            return X402VerifyResponse(
                status="paid",
                token=token,
                expires_in=TOKEN_EXPIRY_SECONDS,
                message="API key verified"
            )
    
    return X402VerifyResponse(
        status="unpaid",
        message="Payment verification failed. Please provide a valid tx_hash or api_key."
    )


@router.get("/status")
async def payment_status(paid: bool = Depends(verify_payment_optional)):
    """
    Check current payment/authentication status.
    """
    return {
        "authenticated": paid,
        "message": "Authenticated" if paid else "No valid token provided"
    }




