# Bad Rabbits Hunter

CPU-multithreaded Node.js miner for the [Bad Rabbits](https://badrabbits.xyz/)
proof-of-work NFT hunt contract on Ethereum mainnet. Submits successful mints
through private mempool RPCs (Flashbots Fast + MEV Blocker) so other hunters
can't front-run the tx.

## What it does

For each round:

1. Fetches the current `challenge`, `difficultyBits`, and minted count from the
   on-chain contract.
2. Spawns N CPU worker threads (one per core minus 1 by default) that hash
   `keccak256(BAD_RABBITS_HUNT || chainId || contract || hunter || challenge || nonce)`
   across a strided nonce range looking for a hash with `difficultyBits` leading
   zero bits.
3. Polls the contract in the background; if the challenge advances mid-search
   (another hunter caught a rabbit), aborts the round and restarts with the new
   challenge — no wasted gas.
4. When a valid nonce is found, re-verifies on-chain via `isValidNonce` and
   simulates the mint via `eth_call`. If either fails (state raced again), drops
   the nonce and re-mines.
5. Signs the mint tx locally then broadcasts the raw bytes to multiple private
   mempool endpoints in parallel — first that accepts wins, the other rejects
   as duplicate.
6. Waits for receipt, parses the `RabbitCaught` event, and (optionally) posts
   a Telegram notification with the OpenSea + Etherscan links.

## Setup

```bash
git clone <this-repo>
cd bad-rabbits-hunter
npm install
cp .env.example .env
# Edit .env — fill in ETH_RPC, PRIVATE_KEY at minimum
node hunt.mjs
```

## Configuration

See [`.env.example`](./.env.example) for the full list of knobs.

| Var | Default | Purpose |
| --- | --- | --- |
| `ETH_RPC` | — | Mainnet RPC URL (Alchemy / Infura / etc.) |
| `PRIVATE_KEY` | — | Hunter wallet private key |
| `BAD_RABBITS_CONTRACT` | — | Contract address (`0x22fD...5839`) |
| `WORKERS` | `cpus() - 1` | CPU worker threads |
| `MEV_RPCS` | Flashbots + MEV Blocker | CSV of private RPCs for tx broadcast |
| `PRIORITY_GWEI` | `2` | Priority fee per gas (gwei) for the mint tx |
| `CHALLENGE_POLL_MS` | `3000` | How often to re-check challenge during mining |
| `TELEGRAM_BOT_TOKEN` | — | Optional bot token for catch notifications |
| `TELEGRAM_CHAT_ID` | — | Telegram chat to notify |

## Expected throughput

Single Node.js worker on a modern CPU core runs `@noble/hashes` keccak256 at
~50–80 kH/s. An 8-core machine reaches ~500–700 kH/s combined.

For difficulty `D` bits, expected hashes-to-find ≈ `2^D`. At 600 kH/s, that's
~1.7 s for a 20-bit challenge, ~7 s for 22 bits, ~30 s for 24 bits.

If the contract is being mined by GPU-class competitors and challenges advance
faster than your search time, CPU mining alone will lose races consistently —
even with private mempool routing. See [`skill.md`](./skill.md) for the GPU
(Rust + CUDA) path.

## Architecture notes

- **Race-aware mining**: workers receive `{ prefix, difficulty, startNonce, stride }`.
  Main thread polls `currentChallenge()` separately and signals abort on change
  via `parentPort.postMessage({ type: 'abort' })`.
- **Encoding parity**: worker prefix is built byte-identical to
  `ethers.solidityPackedKeccak256(['string','uint256','address','address','bytes32','uint256'], ...)`.
  A self-test against ethers is recommended whenever the input layout is changed.
- **No gas wasted on race-losing tx**: `eth_call` simulate runs against latest
  state immediately before broadcast. If it would revert, the round skips
  straight to re-mining.
- **Private broadcast**: tx is signed once, then POSTed in parallel via raw
  `eth_sendRawTransaction` to all `MEV_RPCS`. Duplicate-accept ("already known")
  is treated as success.

## Files

| File | Role |
| --- | --- |
| `hunt.mjs` | Main orchestrator: contract state, race-handling, tx submit |
| `worker.mjs` | Worker thread: tight keccak256 loop on a nonce stride |
| `skill.md` | Recipe for porting the inner mining loop to a CUDA GPU binary |
| `.env.example` | Config template |

## Disclaimer

You are responsible for your own private keys, RPC endpoints, gas costs, and on-chain
actions. Mining is intentionally lossy in the presence of faster competitors —
this tool minimizes gas waste but does not guarantee you'll catch a rabbit.

MIT.
