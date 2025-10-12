# X402 Payment Setup Guide

This guide will help you set up X402 payments for your WarpGBM MCP service.

## Prerequisites

1. **USDC wallet** (Base network recommended for low fees)
2. **Wallet address** where you'll receive payments
3. **Base RPC URL** (free tier from Alchemy or Infura, or use public endpoint)

## Quick Setup

### 1. Create your `.env` file

```bash
cp .env.example .env
```

### 2. Add your wallet address

Edit `.env` and replace `X402_WALLET_ADDRESS` with your actual address:

```bash
# Get from MetaMask, Coinbase Wallet, Rainbow, etc.
X402_WALLET_ADDRESS=0xYourActualWalletAddressHere
```

**Supported wallets:**
- MetaMask
- Coinbase Wallet  
- Rainbow Wallet
- Any Web3 wallet that supports Base network

### 3. Generate a JWT secret

```bash
# Run this to generate a secure secret
openssl rand -hex 32
```

Copy the output and paste it in `.env`:
```bash
X402_JWT_SECRET=<paste-here>
```

### 4. Test in dev mode (payments optional)

```bash
# In .env
X402_PAYMENT_REQUIRED=false
```

Run the service:
```bash
./run_local.sh
```

Test the payment flow:
```bash
python scripts/test_x402.py
```

### 5. Enable payment enforcement (production)

```bash
# In .env
X402_PAYMENT_REQUIRED=true
```

## Payment Flow

### For Users:

1. **Send USDC** to your wallet address on Base network
   - Training: $0.01 USDC
   - Inference: $0.001 USDC

2. **Get transaction hash** (looks like `0x123abc...`)

3. **Verify payment** by calling `/x402/verify`:
   ```bash
   curl -X POST https://your-service.modal.run/x402/verify \
     -H "Content-Type: application/json" \
     -d '{"tx_hash": "0x123abc..."}'
   ```

4. **Receive access token** (valid for 1 hour)

5. **Make requests** with token in Authorization header:
   ```bash
   curl https://your-service.modal.run/train \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{...}'
   ```

## Testing with Real Wallet

### Step 1: Fund your wallet

Get some test USDC on Base:
- Use [Base Sepolia testnet](https://bridge.base.org/deposit?testnet=true) for testing
- Or use mainnet with real USDC (start with $1-$5)

### Step 2: Send a test payment

**Using MetaMask:**
1. Switch to Base network
2. Send 0.01 USDC to `X402_WALLET_ADDRESS`
3. Copy the transaction hash

**Using Coinbase Wallet:**
1. Navigate to USDC balance
2. Send → Enter your `X402_WALLET_ADDRESS`
3. Amount: 0.01
4. Confirm and copy tx hash

### Step 3: Verify the payment

```bash
python scripts/test_x402.py --tx-hash 0xYourActualTxHash
```

### Step 4: Use the token

```bash
# Token will be printed by test script
export TOKEN="your-token-here"

# Test training
curl -X POST https://your-service.modal.run/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"X": [[1,2],[3,4]], "y": [0,1], "model_type": "lightgbm", "objective": "binary"}'
```

## Deploy to Modal with Environment Variables

```bash
# Set Modal secrets (one-time setup)
modal secret create x402-config \
  X402_WALLET_ADDRESS="0xYourAddress" \
  X402_JWT_SECRET="your-secret" \
  X402_PAYMENT_REQUIRED="true"

# Deploy
./deploy.sh
```

## Monitoring Payments

Check your wallet on Base:
- **BaseScan**: https://basescan.org/address/YOUR_ADDRESS
- Track incoming USDC transfers
- Verify transaction hashes

## Pricing

Current rates (modify in `app/x402.py`):
- Training: $0.01 USDC per request
- Inference: $0.001 USDC per request

## Security Notes

- **Never commit** `.env` to git (it's in `.gitignore`)
- **Rotate JWT secrets** periodically
- **Monitor your wallet** for suspicious activity
- **Use testnet** for development
- **Start with mainnet** only when ready for production

## Troubleshooting

### Payment not verifying?

1. Check transaction on BaseScan
2. Verify you sent USDC (not ETH)
3. Verify correct wallet address
4. Wait for transaction confirmation (usually < 2 seconds on Base)
5. Check logs: `X402_PAYMENT_REQUIRED=true DEBUG=true`

### Token expired?

Tokens last 1 hour. Get a new one by calling `/x402/verify` again with the same tx hash (allowed for 24 hours).

## Next Steps

1. ✅ Set up `.env` with your wallet
2. ✅ Test in dev mode
3. ✅ Send test payment
4. ✅ Verify with real tx hash
5. ✅ Enable payment enforcement
6. ✅ Deploy to Modal
7. 🚀 Open source + monetize!

