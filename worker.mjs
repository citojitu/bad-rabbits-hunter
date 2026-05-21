import { parentPort, workerData } from 'node:worker_threads';
import { keccak_256 } from '@noble/hashes/sha3.js';

const { prefix, difficultyBits, startNonce, stride, sessionId } = workerData;

// Input layout: [prefix bytes (120)] [nonce uint256 BE (32)] = 152 bytes.
// Prefix is built once in main thread to match ethers.solidityPacked exactly.
const input = new Uint8Array(prefix.length + 32);
input.set(prefix, 0);
const view = new DataView(input.buffer);
const nonceOffset = prefix.length;

// Pre-compute zero-byte check: keccak hash bytes interpreted as uint256 BE,
// hash ≤ 2^(256-difficultyBits) iff the first `difficultyBits` bits are 0.
// Cheaper than BigInt comparison: just scan leading bytes.
const fullZeroBytes = difficultyBits >> 3;
const remBits = difficultyBits & 7;
const partialMask = remBits === 0 ? 0 : (0xff << (8 - remBits)) & 0xff;

function checkLeading(hash) {
  for (let i = 0; i < fullZeroBytes; i++) if (hash[i] !== 0) return false;
  if (remBits === 0) return true;
  return (hash[fullZeroBytes] & partialMask) === 0;
}

// JS Number safe up to 2^53. Mining nonces rarely exceed 2^48 even at high
// difficulties, so we keep nonce as Number for speed (BigInt ops are ~3x slower).
// Write only the last 8 bytes — top 24 stay zero from initial Uint8Array fill.
function writeNonceBE(offset, n) {
  const hi = Math.floor(n / 0x100000000);
  const lo = n >>> 0;
  view.setUint32(offset + 24, hi, false);
  view.setUint32(offset + 28, lo, false);
}

let aborted = false;
parentPort.on('message', msg => {
  if (msg?.type === 'abort') aborted = true;
});

let nonce = Number(startNonce);
const step = Number(stride);
let countSinceReport = 0;
let lastReport = Date.now();
// Report every 16k hashes — at ~50k H/s per worker this fires every ~300ms,
// so the user sees progress even when rounds get cancelled in <2s by race.
const REPORT_EVERY = 16384;

parentPort.postMessage({ type: 'started', sessionId });

while (!aborted) {
  writeNonceBE(nonceOffset, nonce);
  const hash = keccak_256(input);
  if (checkLeading(hash)) {
    parentPort.postMessage({
      type: 'found',
      sessionId,
      nonce: String(nonce),
      hash: '0x' + Buffer.from(hash).toString('hex'),
    });
    break;
  }
  nonce += step;
  countSinceReport++;
  if (countSinceReport >= REPORT_EVERY) {
    parentPort.postMessage({ type: 'stats', sessionId, count: countSinceReport });
    countSinceReport = 0;
    lastReport = Date.now();
  }
}
