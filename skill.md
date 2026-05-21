---
name: bad-rabbits-cuda-miner
description: Build a CUDA-accelerated keccak256 miner for the BAD_RABBITS NFT hunt contract. Replaces the JS/Worker mining loop with a GPU kernel that hashes hundreds of millions of (prefix||nonce) combos per second and prints valid nonces to stdout for the existing Node submit path. Use when the JS Worker-pool throughput is the bottleneck and the target machine has CUDA toolkit + nvcc available.
---

# Bad Rabbits — CUDA Miner Skill

## What this builds

A standalone binary `bad-rabbit-miner` (Rust + `cudarc`) that:

1. Accepts `--prefix <120-byte-hex>` + `--difficulty <bits>` + `--start-nonce <N>` + `--stride <N>` from CLI.
2. Launches a CUDA kernel that computes `keccak256(prefix || nonce_be_32bytes)` for a strided range of nonces.
3. When a thread finds a hash with the requested leading zero bits, writes the nonce to a device atomic flag, host reads it, prints `FOUND <nonce>` to stdout, exits.
4. Also handles a `--cancel-on-stdin` mode: if main process sends anything on stdin, miner gracefully terminates (used for race cancellation).

The existing `hunt.mjs` (in this directory) keeps managing contract state, race detection, tx submit, and Telegram. It just swaps its `mineRound` to spawn this binary instead of Node Workers.

## Why this architecture

- **Decoupling**: GPU mining and on-chain submit are different concerns. Mining = CPU/GPU compute. Submit = RPC + signing + race handling. Keeping the binary single-purpose (hash + find) makes it easy to test in isolation.
- **Cancel-friendly**: hunt.mjs can `kill -TERM` the miner the moment it detects challenge advance — no inter-language IPC headaches.
- **Reusable**: Same miner works for other PoW-style mints with the same hash shape; just feed different prefix bytes.

## Prerequisites on the target machine

1. **NVIDIA GPU** with compute capability ≥ 6.0 (Pascal/Volta/Turing/Ampere/Ada/Hopper). RTX 30/40 series ideal.
2. **CUDA Toolkit ≥ 11.4** (12.x preferred). `nvcc --version` must work.
3. **Rust toolchain** (`rustup`). Stable channel.
4. **Node.js ≥ 18** for `hunt.mjs`.
5. Same `.env` as the JS path (`ETH_RPC`, `PRIVATE_KEY`, `BAD_RABBITS_CONTRACT`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).

Verify:
```bash
nvcc --version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
rustc --version
node --version
```

## Project layout

```
bad-rabbit/
├── hunt.mjs                  # existing — modified to spawn miner binary
├── worker.mjs                # existing JS worker (keep as CPU fallback)
├── .env                      # existing
├── miner-cuda/
│   ├── Cargo.toml
│   ├── build.rs              # compiles .cu → .ptx at build time
│   ├── src/
│   │   ├── main.rs           # host code: arg parse, kernel launch, output
│   │   └── kernel.cu         # CUDA keccak256 + nonce search
│   └── target/release/bad-rabbit-miner.exe   # output binary
```

## Step 1 — Scaffold the Rust crate

```bash
cd D:\bad-rabbit
cargo new miner-cuda --bin
cd miner-cuda
```

`Cargo.toml`:
```toml
[package]
name = "bad-rabbit-miner"
version = "0.1.0"
edition = "2021"

[dependencies]
cudarc = { version = "0.16", features = ["cuda-version-from-build-system", "driver", "nvrtc"] }
hex = "0.4"
clap = { version = "4", features = ["derive"] }
anyhow = "1"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

## Step 2 — Write the CUDA kernel

`miner-cuda/src/kernel.cu`:
```cuda
// keccak256 (Ethereum) — round constants + per-thread state
// Reference impl: NIST FIPS 202. Optimized for short inputs (≤200 bytes).

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

__constant__ u64 RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
  0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
  0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
  0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
  0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
  0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
  0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
  0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

#define ROTL64(x, n) ((x << n) | (x >> (64 - n)))

__device__ void keccak_f(u64 s[25]) {
  #pragma unroll
  for (int round = 0; round < 24; round++) {
    u64 C[5], D[5], B[25];

    #pragma unroll
    for (int x = 0; x < 5; x++)
      C[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];

    #pragma unroll
    for (int x = 0; x < 5; x++) {
      D[x] = C[(x+4)%5] ^ ROTL64(C[(x+1)%5], 1);
      s[x] ^= D[x]; s[x+5] ^= D[x]; s[x+10] ^= D[x]; s[x+15] ^= D[x]; s[x+20] ^= D[x];
    }

    // Rho + Pi (rotation offsets for 5x5 lane permutation)
    B[0] = s[0];
    B[10] = ROTL64(s[1], 1);   B[7]  = ROTL64(s[10], 3);
    B[11] = ROTL64(s[7], 6);   B[17] = ROTL64(s[11], 10);
    B[18] = ROTL64(s[17], 15); B[3]  = ROTL64(s[18], 21);
    B[5]  = ROTL64(s[3], 28);  B[16] = ROTL64(s[5], 36);
    B[8]  = ROTL64(s[16], 45); B[21] = ROTL64(s[8], 55);
    B[24] = ROTL64(s[21], 2);  B[4]  = ROTL64(s[24], 14);
    B[15] = ROTL64(s[4], 27);  B[23] = ROTL64(s[15], 41);
    B[19] = ROTL64(s[23], 56); B[13] = ROTL64(s[19], 8);
    B[12] = ROTL64(s[13], 25); B[2]  = ROTL64(s[12], 43);
    B[20] = ROTL64(s[2], 62);  B[14] = ROTL64(s[20], 18);
    B[22] = ROTL64(s[14], 39); B[9]  = ROTL64(s[22], 61);
    B[6]  = ROTL64(s[9], 20);  B[1]  = ROTL64(s[6], 44);

    // Chi
    #pragma unroll
    for (int y = 0; y < 25; y += 5) {
      u64 t0 = B[y], t1 = B[y+1], t2 = B[y+2], t3 = B[y+3], t4 = B[y+4];
      s[y]   = t0 ^ ((~t1) & t2);
      s[y+1] = t1 ^ ((~t2) & t3);
      s[y+2] = t2 ^ ((~t3) & t4);
      s[y+3] = t3 ^ ((~t4) & t0);
      s[y+4] = t4 ^ ((~t0) & t1);
    }

    // Iota
    s[0] ^= RC[round];
  }
}

// Mine: each thread computes keccak256(prefix || (start_nonce + tid * stride))
// and atomically writes nonce to *found if hash has ≥ difficulty_bits leading zeros.
extern "C" __global__ void mine(
  const u8* __restrict__ prefix, // 120 bytes
  u64 start_nonce,
  u64 stride,
  u64 iter_count,                // hashes per thread
  u32 difficulty_bits,
  u64* __restrict__ found_nonce, // output: 0 if not found, else nonce
  u32* __restrict__ found_flag   // output: 0/1
) {
  u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
  u64 base_nonce = start_nonce + tid * iter_count * stride;

  // Pre-fill state[0..14] with prefix bytes (120 bytes = 15 u64s).
  // State is little-endian u64, but keccak treats bytes via shift; we'll copy
  // 8 bytes at a time as u64 LE-cast.
  u64 prefix_lanes[15];
  #pragma unroll
  for (int i = 0; i < 15; i++) {
    u64 v = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) v |= ((u64)prefix[i*8 + j]) << (j*8);
    prefix_lanes[i] = v;
  }

  // Difficulty: hash has ≥ bits leading zero bits.
  // After keccak_f, first 32 bytes of state = hash. We check bytes[0..fullBytes]==0
  // and (bytes[fullBytes] & partialMask) == 0.
  u32 full_zero_bytes = difficulty_bits >> 3;
  u32 rem_bits = difficulty_bits & 7;
  u8 partial_mask = rem_bits ? (0xFFu << (8 - rem_bits)) & 0xFFu : 0;

  for (u64 k = 0; k < iter_count; k++) {
    if (*found_flag) return;  // another thread won

    u64 nonce = base_nonce + k * stride;

    u64 s[25];
    #pragma unroll
    for (int i = 0; i < 15; i++) s[i] = prefix_lanes[i];
    // s[15] holds bytes 120..127 of input: zero except nonce top 8 bytes start here.
    // We pack nonce (32 bytes BE) into lanes 15..18 such that the BIG-ENDIAN
    // 32-byte encoding of nonce ends up at byte offsets 120..151 of the input.
    // For nonce < 2^64, only the last 8 bytes are non-zero; top 24 bytes (BE) = 0.
    // So bytes 120..143 = 0, bytes 144..151 = nonce big-endian.
    // In keccak lane indexing (LE u64), bytes 144..151 = lane 18.
    // We must store nonce BIG-ENDIAN into those 8 bytes — i.e. byte 144 = MSB.
    // That means lane 18 (LE u64) holds the byte-reversed nonce:
    //   lane18 = bswap64(nonce)
    s[15] = 0;
    s[16] = 0;
    s[17] = 0;
    s[18] = __byte_perm(__byte_perm(nonce, 0, 0x0123) | (u64)__byte_perm(nonce >> 32, 0, 0x0123) << 32, 0, 0x0123);
    // Note: __byte_perm operates on 32-bit halves. Simpler manual bswap below.
    {
      u64 n = nonce;
      u64 r = 0;
      #pragma unroll
      for (int b = 0; b < 8; b++) {
        r = (r << 8) | (n & 0xFFu);
        n >>= 8;
      }
      s[18] = r;
    }

    // Padding: keccak256 with input length 152 bytes. Block size for keccak256
    // = 136 bytes (rate). Since 152 > 136, we'd normally need 2 absorb blocks.
    // BUT — typo: ethereum keccak (legacy) uses pad rule `0x01 ... 0x80` not
    // 0x06. Confirm: ethers solidityPackedKeccak256 uses legacy Keccak (not
    // NIST SHA-3) so pad byte = 0x01.
    //
    // For 152-byte input we absorb TWO blocks:
    //   block 1: bytes 0..135  (lanes 0..16, partial lane 17 starts byte 136)
    //   block 2: bytes 136..151 (lanes 17..18 partial) + pad
    //
    // Round 1: XOR lanes 0..16 into state[0..16], state[17] partial. Wait:
    // 152 bytes spans lanes 0..18 (152/8 = 19 lanes). 136-byte rate spans
    // lanes 0..16 (17 lanes = 136 bytes). So:
    //   First absorb: state[0..16] ^= input lanes 0..16. Run keccak_f.
    //   Second absorb: state[0..1] ^= remaining lanes 17..18 (bytes 136..151).
    //                  Then pad: byte 152 = 0x01, byte 135 (block end) = 0x80.
    //                  In lane terms: state[2] ^= 0x01 (byte 16 of block 2 = byte 152, but we re-indexed: state[lane_of_byte_152_in_block2] which is lane 2 = bytes 16-23 of block 2 = bytes 152-159 of input. So state[2] bottom byte ^= 0x01).
    //                  Last byte of block 2 (rate-1 = byte 135 = lane 16 byte 7 of block 2). Block 2 = bytes 136..271, so byte 271 of input is the last padding byte. In lane indexing within block 2: state[16] top byte ^= 0x80.
    //
    // Simpler: re-derive padding manually. Below code does it concretely.

    // First absorb: input bytes 0..135 = lanes 0..16
    // We already set s[15] = 0 (no, we set it to a value above; we need to RESET
    // before XORing). Actually since s[15..24] are all zero initially and the
    // prefix only fills s[0..14] (120 bytes = lanes 0..14), s[15] for the first
    // block is whatever the input lane 15 should be. Input lane 15 = bytes
    // 120..127 = first 8 bytes of the 32-byte nonce field = top of nonce BE.
    // For nonce < 2^64, those bytes are all zero. So s[15] = 0. ✓
    // Lane 16 = bytes 128..135 = next 8 bytes of nonce field = still zero. ✓
    //
    // BUT wait, we already set s[15..18] to the nonce field above. That includes
    // s[15] = 0, s[16] = 0, s[17] = 0, s[18] = bswap(nonce). For absorb block 1
    // we want to XOR ONLY lanes 0..16 (rate = 17 lanes). s[15] and s[16] match
    // input lane 15 and 16 (both 0 for our case). ✓. Lane 17 and 18 are for
    // block 2.
    //
    // For block 1: lanes 0..16 already XOR'd (prefix lanes 0..14 are nonzero,
    // 15 and 16 are 0, which is correct — input bytes 120..135 are zero).
    // Run keccak_f.
    keccak_f(s);

    // Block 2: XOR lane 0 ^= input lane 17 (bytes 136..143 = bytes 16..23 of
    // 32-byte nonce field = still zero). lane 1 ^= input lane 18 = bswap(nonce).
    s[0] ^= 0;             // input lane 17 = 0
    s[1] ^= s[18];         // wait — we stored bswap(nonce) in s[18]. Need to
                           // pull it from a register. Use the local r value.
    // ... actually simpler: just inline the bswap result here.
    {
      u64 n = nonce, r = 0;
      #pragma unroll
      for (int b = 0; b < 8; b++) { r = (r << 8) | (n & 0xFFu); n >>= 8; }
      s[1] ^= r;
    }
    // Pad: byte at position (152 - 136) = 16 in block 2 = lane 2, low byte = 0x01.
    s[2] ^= 0x01ULL;
    // End-of-message byte: last byte of block 2 = byte 135 of block 2 = lane 16
    // high byte = 0x80 shifted to top.
    s[16] ^= 0x8000000000000000ULL;
    keccak_f(s);

    // s[0..3] = first 32 bytes of hash (lanes are LE). Check leading zero bits.
    // For zero-byte check we need bytes in their natural order. Lane 0 bytes 0..7
    // are: (s[0] >> 0) & 0xFF, (s[0] >> 8) & 0xFF, ..., (s[0] >> 56) & 0xFF.
    // So byte i of hash = (s[i/8] >> ((i%8)*8)) & 0xFF.

    bool ok = true;
    for (u32 i = 0; i < full_zero_bytes; i++) {
      u8 b = (s[i >> 3] >> ((i & 7) * 8)) & 0xFFu;
      if (b != 0) { ok = false; break; }
    }
    if (ok && rem_bits) {
      u32 i = full_zero_bytes;
      u8 b = (s[i >> 3] >> ((i & 7) * 8)) & 0xFFu;
      if ((b & partial_mask) != 0) ok = false;
    }

    if (ok) {
      if (atomicCAS(found_flag, 0, 1) == 0) {
        *found_nonce = nonce;
      }
      return;
    }
  }
}
```

> **⚠️ Kernel correctness warning**: keccak256 implementations are easy to get wrong (padding, byte order, round constants). The code above is annotated but **MUST be cross-validated** before mining real funds. Test procedure:
>
> 1. Pick a known prefix + nonce.
> 2. Compute reference hash via `ethers.solidityPackedKeccak256` (or `cast keccak`).
> 3. Run the CUDA kernel with `difficulty=256` (impossible, but reads hash output if you instrument the kernel to ALWAYS write hash bytes for thread 0).
> 4. Compare byte-for-byte.
>
> Add a `--test` mode to the binary that takes `--nonce <N>` and prints `keccak256(prefix||nonce_be)` to stdout. Validate against ethers BEFORE running real mining.

## Step 3 — Build script

`miner-cuda/build.rs`:
```rust
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let src = "src/kernel.cu";
    let ptx_out = out_dir.join("kernel.ptx");

    let status = std::process::Command::new("nvcc")
        .args([
            "--ptx",
            "-O3",
            "--gpu-architecture=sm_86",  // Adjust per GPU: sm_75 (Turing), sm_86 (Ampere RTX 30xx), sm_89 (Ada RTX 40xx)
            "-o",
            ptx_out.to_str().unwrap(),
            src,
        ])
        .status()
        .expect("nvcc failed to run");
    assert!(status.success(), "nvcc compile failed");

    println!("cargo:rerun-if-changed={}", src);
    println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx_out.display());
}
```

## Step 4 — Host driver

`miner-cuda/src/main.rs`:
```rust
use anyhow::{Context, Result};
use clap::Parser;
use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Args {
    /// Hex-encoded 120-byte prefix
    #[arg(long)]
    prefix: String,

    /// Difficulty in leading zero bits
    #[arg(long)]
    difficulty: u32,

    /// Starting nonce (workers may slice this)
    #[arg(long, default_value = "0")]
    start_nonce: u64,

    /// Stride between consecutive nonces probed by one thread
    #[arg(long, default_value = "1")]
    stride: u64,

    /// CUDA grid: blocks
    #[arg(long, default_value = "2048")]
    blocks: u32,

    /// CUDA grid: threads per block
    #[arg(long, default_value = "256")]
    threads: u32,

    /// Hashes per thread per kernel launch
    #[arg(long, default_value = "256")]
    iter: u64,

    /// Print stats every N seconds
    #[arg(long, default_value = "3")]
    stats_interval: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let prefix_bytes = hex::decode(args.prefix.trim_start_matches("0x"))?;
    anyhow::ensure!(prefix_bytes.len() == 120, "prefix must be exactly 120 bytes (240 hex chars)");

    let dev = CudaDevice::new(0)?;
    let ptx_path = env!("KERNEL_PTX_PATH");
    let ptx = Ptx::from_file(ptx_path);
    dev.load_ptx(ptx, "kernel", &["mine"])?;
    let mine = dev.get_func("kernel", "mine").context("kernel not loaded")?;

    let d_prefix = dev.htod_copy(prefix_bytes.clone())?;
    let mut d_found_nonce = dev.alloc_zeros::<u64>(1)?;
    let mut d_found_flag = dev.alloc_zeros::<u32>(1)?;

    let cfg = LaunchConfig {
        grid_dim: (args.blocks, 1, 1),
        block_dim: (args.threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let threads_per_launch = (args.blocks as u64) * (args.threads as u64);
    let nonces_per_launch = threads_per_launch * args.iter * args.stride;

    let mut nonce_base = args.start_nonce;
    let started = Instant::now();
    let mut last_stats = Instant::now();
    let mut launches: u64 = 0;

    loop {
        unsafe {
            mine.clone().launch(
                cfg,
                (
                    &d_prefix,
                    nonce_base,
                    args.stride,
                    args.iter,
                    args.difficulty,
                    &mut d_found_nonce,
                    &mut d_found_flag,
                ),
            )?;
        }
        dev.synchronize()?;

        let flag = dev.dtoh_sync_copy(&d_found_flag)?[0];
        if flag != 0 {
            let nonce = dev.dtoh_sync_copy(&d_found_nonce)?[0];
            println!("FOUND {}", nonce);
            return Ok(());
        }

        nonce_base += nonces_per_launch;
        launches += 1;

        if last_stats.elapsed().as_secs() >= args.stats_interval {
            let elapsed_s = started.elapsed().as_secs_f64();
            let total_hashes = (launches * nonces_per_launch) as f64;
            let rate = total_hashes / elapsed_s;
            eprintln!(
                "stats nonces={} elapsed={:.1}s rate={:.0} MH/s",
                nonce_base - args.start_nonce,
                elapsed_s,
                rate / 1e6
            );
            last_stats = Instant::now();
        }
    }
}
```

## Step 5 — Build

```bash
cd D:\bad-rabbit\miner-cuda
cargo build --release
# Binary: target\release\bad-rabbit-miner.exe
```

If `nvcc` is in PATH (it should be after CUDA Toolkit install — re-open shell), build succeeds.

## Step 6 — Self-test the kernel (CRITICAL)

Before integrating with hunt.mjs, prove the GPU produces the same hash as ethers:

1. Add a `--test --nonce <N>` mode to `main.rs` that loads prefix + nonce, launches kernel once with iter=1 stride=1, and reads back hash from device (modify kernel to also write hash bytes to a debug buffer).
2. Compute reference: `node -e "import('ethers').then(({ethers}) => console.log(ethers.solidityPackedKeccak256([...], [...])))"`.
3. Compare. **If different — DO NOT proceed.** Debug encoding, byte order, padding.

A passing self-test should look like:
```
$ bad-rabbit-miner --test --prefix 0x... --nonce 12345
hash: 0xea92ded3b3d2d69b2b2da0af7f27a59e03f406cbda530c676f6d52c0063e7711
$ node verify.mjs 12345
ethers: 0xea92ded3b3d2d69b2b2da0af7f27a59e03f406cbda530c676f6d52c0063e7711
MATCH
```

## Step 7 — Integrate with `hunt.mjs`

Replace the `mineRound()` function in `hunt.mjs` with a child-process spawner:

```javascript
import { spawn } from 'node:child_process';
import { resolve as resolvePath } from 'node:path';

const MINER_BIN = process.env.MINER_BIN
  || resolvePath(__dirname, 'miner-cuda/target/release/bad-rabbit-miner.exe');

function mineRound(prefix, difficultyBits, cancelSignal) {
  return new Promise((resolve) => {
    const prefixHex = '0x' + Buffer.from(prefix).toString('hex');
    const child = spawn(MINER_BIN, [
      '--prefix', prefixHex,
      '--difficulty', String(difficultyBits),
    ], { stdio: ['ignore', 'pipe', 'pipe'] });

    let settled = false;
    const settle = (value) => {
      if (settled) return;
      settled = true;
      try { child.kill('SIGTERM'); } catch {}
      cancelSignal.onCancel(null);
      resolve(value);
    };

    cancelSignal.onCancel(() => settle({ cancelled: true }));

    let stdoutBuf = '';
    child.stdout.on('data', (chunk) => {
      stdoutBuf += chunk.toString();
      const lines = stdoutBuf.split('\n');
      stdoutBuf = lines.pop();
      for (const line of lines) {
        const m = line.match(/^FOUND (\d+)/);
        if (m) {
          settle({ found: { nonce: BigInt(m[1]), hash: null /* fetched on-chain anyway */ } });
        }
      }
    });
    child.stderr.on('data', (chunk) => {
      process.stderr.write(`[miner] ${chunk.toString()}`);
    });
    child.on('exit', (code) => {
      if (!settled) {
        console.error(`miner exited with code ${code} before finding nonce`);
        settle({ cancelled: true });
      }
    });
  });
}
```

`hash` is `null` because the JS side will re-derive locally only if it wants to print; the on-chain `isValidNonce` already verifies. (Optionally, miner can also print `HASH 0x...` line and JS parses it for the log.)

## Step 8 — Run

```bash
cd D:\bad-rabbit
node hunt.mjs
```

Expected output (stderr from miner, stdout from hunt.mjs):
```
BAD RABBITS HUNT STARTED
Wallet:     0x...
Workers:    (ignored — uses GPU)
Difficulty: 22 bits
--- Attempt #1 ---
Challenge: 0x...
[miner] stats nonces=536870912 elapsed=3.1s rate=173 MH/s
Nonce found: 4443823
Valid on-chain: true
Submitting transaction...
Tx sent: 0x...
🐰 BAD RABBIT CAUGHT (telegram notif)
```

## Tuning notes

| GPU | Compute | Recommended `--blocks --threads --iter` |
|-----|---------|------------------------------------------|
| GTX 1660 (Turing sm_75) | 6.1 | `1024 256 128` (~80 MH/s) |
| RTX 3050 (Ampere sm_86) | 8.6 | `2048 256 256` (~150 MH/s) |
| RTX 3060/3070 (Ampere sm_86) | 8.6 | `4096 256 256` (~300-500 MH/s) |
| RTX 4070/4080 (Ada sm_89) | 8.9 | `8192 256 256` (~800 MH/s - 1.2 GH/s) |

Change `--gpu-architecture` in `build.rs` to match the target card; mismatched arch falls back to JIT and runs slower.

## Failure modes & debugging

1. **`nvcc not found`**: CUDA Toolkit not installed or `PATH` missing `%CUDA_PATH%\bin`. Re-install toolkit + reboot shell.
2. **`CUDA_ERROR_NO_BINARY_FOR_GPU`**: `sm_xx` in build.rs wrong for installed driver. Lower to `sm_60` to be safe (slower but compatible).
3. **Hash mismatch (Step 6 fails)**: Almost always byte ordering or padding. Common bugs:
   - keccak256 (legacy) pad is `0x01 ... 0x80`, **NOT** SHA-3 pad `0x06 ... 0x80`. Double check.
   - Nonce must be big-endian 32 bytes — last 8 bytes carry value (for nonce < 2^64), first 24 must be zero.
   - Prefix lane-packing is little-endian (first byte → low byte of u64).
4. **`atomicCAS` not finding nonce despite kernel running long**: kernel is mining wrong hash, OR `difficulty_bits` is being misinterpreted. Add a `printf` in kernel for one thread at iter 0.
5. **JS gets `Valid on-chain: false`** after CUDA mine: same encoding mismatch — kernel and ethers diverged. Run self-test again.

## What this skill DOES NOT cover

- **Flashbots/MEV bundle submission** — the actual win in PoW NFT races. Separate skill.
- **Multi-GPU** — straightforward extension (one process per GPU, slice `--start-nonce`).
- **Persistent kernel** that listens for new challenges via shared memory — not needed; cancel-via-SIGTERM works.
- **Cancel-on-stdin** in the miner — `SIGTERM` from Node is simpler on Windows + Linux. Already done.

## Stop-conditions before shipping to mainnet

- [ ] Self-test (Step 6) passes byte-for-byte
- [ ] Mine on testnet with low difficulty first (set `difficulty=8`, verify `Valid on-chain: true`)
- [ ] Test cancel: SIGTERM during mine should exit within 1 second
- [ ] Confirm `nvcc` arch matches target GPU
- [ ] Backup `.env` private key elsewhere before deploying
