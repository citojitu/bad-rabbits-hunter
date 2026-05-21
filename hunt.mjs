import 'dotenv/config';
import { ethers } from 'ethers';
import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const RPC = process.env.ETH_RPC;
const PRIVATE_KEY = process.env.PRIVATE_KEY;
const CONTRACT = process.env.BAD_RABBITS_CONTRACT;
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;
const NUM_WORKERS = parseInt(process.env.WORKERS || String(Math.max(1, cpus().length - 1)), 10);
const CHALLENGE_POLL_MS = parseInt(process.env.CHALLENGE_POLL_MS || '3000', 10);

// Private RPCs for tx submission — broadcast raw tx to BOTH in parallel
// (signed once, identical hash → first that accepts wins, the other rejects
// as duplicate). Bypasses public mempool → no front-run. Override via
// .env MEV_RPCS as comma-separated list.
const MEV_RPCS = (process.env.MEV_RPCS || [
  'https://rpc.flashbots.net/fast',
  'https://rpc.mevblocker.io',
].join(',')).split(',').map(s => s.trim()).filter(Boolean);

if (!RPC) throw new Error('Missing ETH_RPC');
if (!PRIVATE_KEY) throw new Error('Missing PRIVATE_KEY');
if (!CONTRACT) throw new Error('Missing BAD_RABBITS_CONTRACT');

const provider = new ethers.JsonRpcProvider(RPC);
const wallet = new ethers.Wallet(PRIVATE_KEY, provider);

// Broadcast a signed raw tx to many endpoints in parallel. Returns the first
// success {endpoint, txHash, raw}. If all fail, throws aggregate error.
async function broadcastRawTx(rawHex) {
  const body = JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_sendRawTransaction',
    params: [rawHex],
  });
  const attempts = MEV_RPCS.map(async (url) => {
    try {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 10000);
      const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: ac.signal,
      });
      clearTimeout(timer);
      const json = await r.json();
      if (json.error) {
        // Many endpoints return "already known" / "nonce too low" / "replacement
        // tx underpriced" for the SECOND+ accepts — treat as silent success
        // since the tx is already in flight.
        const msg = (json.error.message || '').toLowerCase();
        if (msg.includes('already known') || msg.includes('known transaction')) {
          return { url, accepted: true, duplicate: true };
        }
        throw new Error(`${url}: ${json.error.message}`);
      }
      return { url, accepted: true, hash: json.result };
    } catch (e) {
      throw new Error(`${url}: ${e.message}`);
    }
  });

  const results = await Promise.allSettled(attempts);
  const ok = results.filter(r => r.status === 'fulfilled' && r.value.accepted);
  if (ok.length === 0) {
    const errs = results
      .filter(r => r.status === 'rejected')
      .map(r => '  - ' + r.reason.message)
      .join('\n');
    throw new Error(`All ${MEV_RPCS.length} MEV RPCs rejected:\n${errs}`);
  }
  const accepted = ok.map(r => r.value.url);
  const txHash = ok.find(r => r.value.hash)?.value.hash || null;
  return { accepted, txHash };
}

const net = await provider.getNetwork();
const CHAIN_ID = net.chainId;
if (process.env.CHAIN_ID && BigInt(process.env.CHAIN_ID) !== CHAIN_ID) {
  console.warn(`WARN: .env CHAIN_ID=${process.env.CHAIN_ID} but provider says ${CHAIN_ID}. Using provider value.`);
}

const abi = [
  'function currentChallenge() view returns (bytes32)',
  'function difficultyBits() view returns (uint8)',
  'function mintEnabled() view returns (bool)',
  'function totalMinted() view returns (uint256)',
  'function isValidNonce(address hunter,uint256 nonce) view returns (bool)',
  'function quoteHuntFee(address hunter,uint256 nonce) view returns (uint256)',
  'function mint(uint256 nonce) payable',
  'event RabbitCaught(address indexed hunter,uint256 indexed totonId,uint256 indexed nonce,bytes32 proofHash,uint256 huntFee,bytes32 nextChallenge)'
];

const contract = new ethers.Contract(CONTRACT, abi, wallet);

// Build the 120-byte prefix that workers will hash with varying nonce appended.
// Must match ethers.solidityPacked layout byte-for-byte:
//   'BAD_RABBITS_HUNT' (16B utf-8)
//   chainId (32B uint256 BE)
//   contract address (20B)
//   hunter address (20B)
//   challenge (32B bytes32)
// Total: 120 bytes. Nonce (32B uint256 BE) gets appended per iteration in worker.
function buildPrefix(chainIdBig, contractAddr, hunterAddr, challenge) {
  const buf = Buffer.alloc(120);
  let o = 0;
  Buffer.from('BAD_RABBITS_HUNT', 'utf8').copy(buf, o); o += 16;
  let c = chainIdBig;
  for (let i = 31; i >= 0; i--) { buf[o + i] = Number(c & 0xffn); c >>= 8n; }
  o += 32;
  Buffer.from(contractAddr.replace(/^0x/, ''), 'hex').copy(buf, o); o += 20;
  Buffer.from(hunterAddr.replace(/^0x/, ''), 'hex').copy(buf, o); o += 20;
  Buffer.from(challenge.replace(/^0x/, ''), 'hex').copy(buf, o); o += 32;
  return buf;
}

async function notifyTelegram(text) {
  if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) return;
  try {
    const r = await fetch(
      `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHAT_ID,
          text,
          parse_mode: 'HTML',
          disable_web_page_preview: false,
        }),
      }
    );
    if (!r.ok) {
      console.warn(`telegram notify ${r.status}:`, (await r.text()).slice(0, 200));
    }
  } catch (e) {
    console.warn('telegram notify error:', e.message);
  }
}

// Run a mining round: spawn workers, await either a 'found' or external cancel.
// Returns { found?: {nonce, hash}, cancelled?: true }.
function mineRound(prefix, difficultyBits, cancelSignal) {
  return new Promise((resolve) => {
    const sessionId = Math.random().toString(36).slice(2);
    const workers = [];
    let settled = false;
    let totalHashes = 0;
    const started = Date.now();

    const cleanup = () => {
      for (const w of workers) {
        try { w.postMessage({ type: 'abort' }); } catch {}
        w.terminate().catch(() => {});
      }
    };

    const settle = (value) => {
      if (settled) return;
      settled = true;
      clearInterval(reporter);
      cancelSignal.onCancel(null);
      cleanup();
      resolve(value);
    };

    const reporter = setInterval(() => {
      const sec = Math.max(0.001, (Date.now() - started) / 1000);
      const rate = totalHashes / sec;
      console.log(`workers=${NUM_WORKERS} hashes=${totalHashes} rate=${(rate / 1000).toFixed(1)} kH/s`);
    }, 3000);

    cancelSignal.onCancel(() => settle({ cancelled: true }));

    for (let i = 0; i < NUM_WORKERS; i++) {
      const w = new Worker(join(__dirname, 'worker.mjs'), {
        workerData: {
          prefix: Uint8Array.from(prefix),
          difficultyBits,
          startNonce: String(i),
          stride: String(NUM_WORKERS),
          sessionId,
        },
      });
      w.on('message', (msg) => {
        if (msg?.sessionId !== sessionId) return;
        if (msg.type === 'found') {
          settle({ found: { nonce: BigInt(msg.nonce), hash: msg.hash } });
        } else if (msg.type === 'stats') {
          totalHashes += msg.count;
        }
      });
      w.on('error', (e) => {
        console.error('worker error:', e.message);
      });
      workers.push(w);
    }
  });
}

// Cancel signal that mineRound can subscribe to. Main loop calls .cancel()
// when it detects challenge changed via background polling.
function makeCancelSignal() {
  let cb = null;
  return {
    onCancel(fn) { cb = fn; },
    cancel() { if (cb) cb(); },
  };
}

console.log('BAD RABBITS HUNT STARTED');
console.log('Wallet:    ', wallet.address);
console.log('Contract:  ', CONTRACT);
console.log('Chain:     ', CHAIN_ID.toString());
console.log('Workers:   ', NUM_WORKERS);
console.log(`Read RPC:  ${new URL(RPC).hostname}`);
console.log(`MEV RPCs:  ${MEV_RPCS.length} (${MEV_RPCS.map(u => new URL(u).hostname).join(', ')})`);

const mintEnabled = await contract.mintEnabled();
if (!mintEnabled) throw new Error('Mint is disabled.');

const difficultyBits = Number(await contract.difficultyBits());
console.log('Difficulty:', difficultyBits, 'bits');

await notifyTelegram(
  `🐰 <b>Bad Rabbit hunt started (multi-thread)</b>\n` +
  `Wallet: <code>${wallet.address}</code>\n` +
  `Difficulty: ${difficultyBits} bits\n` +
  `Workers: ${NUM_WORKERS}\n` +
  `Chain: ${CHAIN_ID.toString()}`
);

let attempt = 0;
while (true) {
  attempt++;
  let currentChallenge = await contract.currentChallenge();
  const totalMinted = await contract.totalMinted();
  console.log('');
  console.log(`--- Attempt #${attempt} ---`);
  console.log('Challenge: ', currentChallenge);
  console.log('Minted:    ', totalMinted.toString());

  const prefix = buildPrefix(CHAIN_ID, CONTRACT, wallet.address, currentChallenge);
  const cancelSignal = makeCancelSignal();
  const minePromise = mineRound(prefix, difficultyBits, cancelSignal);

  // Background poller: triggers cancel if contract challenge advances while
  // we're mining. Separate from the rate-limited reporter inside mineRound.
  const poller = setInterval(async () => {
    try {
      const fresh = await contract.currentChallenge();
      if (fresh !== currentChallenge) {
        console.log('Challenge advanced during mining — cancelling round');
        cancelSignal.cancel();
      }
    } catch {}
  }, CHALLENGE_POLL_MS);

  const result = await minePromise;
  clearInterval(poller);

  if (result.cancelled) {
    console.log('Round cancelled, restarting with new challenge');
    continue;
  }

  const { nonce, hash } = result.found;
  console.log('');
  console.log('Nonce found:', nonce.toString());
  console.log('Hash:       ', hash);

  const validOnchain = await contract.isValidNonce(wallet.address, nonce);
  console.log('Valid on-chain:', validOnchain);

  if (!validOnchain) {
    const freshChallenge = await contract.currentChallenge();
    if (freshChallenge !== currentChallenge) {
      console.log('Challenge advanced between local find and on-chain check — re-mining');
      continue;
    }
    throw new Error(
      'Local nonce invalid on-chain even though challenge unchanged — encoding mismatch or unknown contract rule.'
    );
  }

  let txValue;
  try {
    txValue = await contract.quoteHuntFee(wallet.address, nonce);
  } catch (e) {
    console.log('quoteHuntFee reverted — likely challenge raced, re-mining');
    continue;
  }
  console.log('Hunt fee:   ', ethers.formatEther(txValue), 'ETH');

  // Priority fee in gwei. Default 2 = normal high-priority tx. Bump via
  // .env PRIORITY_GWEI=10 (or higher) if race competition needs more bidding,
  // but remember: priority × gas (~80k) is paid even on winning tx, so 20+ gwei
  // can cost more than the NFT mint fee itself.
  const priorityGwei = BigInt(process.env.PRIORITY_GWEI || '2');
  const feeData = await provider.getFeeData();
  const baseTip = feeData.maxPriorityFeePerGas || ethers.parseUnits('2', 'gwei');
  const maxPriority = priorityGwei * 10n ** 9n;
  const maxFee = (feeData.maxFeePerGas || (baseTip * 2n)) + maxPriority;

  console.log(`Priority: ${ethers.formatUnits(maxPriority, 'gwei')} gwei tip, max ${ethers.formatUnits(maxFee, 'gwei')} gwei`);
  console.log(`Building + broadcasting tx to ${MEV_RPCS.length} private RPCs in parallel...`);

  // Build the calldata for mint(nonce). Sign locally then broadcast raw to all
  // MEV-protect endpoints. Identical hash + nonce → only one actually lands,
  // others reject as duplicate (already known) which we treat as success.
  let rawHex, txHash, accepted;
  try {
    const data = contract.interface.encodeFunctionData('mint', [nonce]);
    const populated = await wallet.populateTransaction({
      to: CONTRACT,
      data,
      value: txValue,
      maxPriorityFeePerGas: maxPriority,
      maxFeePerGas: maxFee,
    });

    // Final simulate: re-run mint() as eth_call right NOW against latest state.
    // If it'd revert here, we'd just be wasting gas — skip + re-mine instead.
    try {
      await provider.call({
        to: CONTRACT,
        data,
        value: txValue,
        from: wallet.address,
      });
    } catch (simErr) {
      console.log('Pre-flight simulate reverted — state changed, re-mining');
      continue;
    }

    const signed = await wallet.signTransaction(populated);
    rawHex = signed;
    txHash = ethers.keccak256(signed);
    const result = await broadcastRawTx(rawHex);
    accepted = result.accepted;
    if (result.txHash && result.txHash !== txHash) {
      console.warn(`Hash mismatch: local=${txHash} rpc=${result.txHash} — using local`);
    }
  } catch (e) {
    console.error('Broadcast failed:', e.message);
    console.log('Race lost or all RPCs rejected — re-mining with fresh challenge');
    continue;
  }

  console.log('Tx sent:', txHash);
  console.log('Accepted by:', accepted.map(u => new URL(u).hostname).join(', '));
  await notifyTelegram(
    `🐰 <b>Tx submitted (private)</b>\n` +
    `Wallet: <code>${wallet.address}</code>\n` +
    `Nonce: <code>${nonce.toString()}</code>\n` +
    `Fee: ${ethers.formatEther(txValue)} ETH\n` +
    `Accepted: ${accepted.length}/${MEV_RPCS.length} RPCs\n` +
    `<a href="https://etherscan.io/tx/${txHash}">Etherscan ${txHash.slice(0, 10)}…</a>`
  );

  const tx = { hash: txHash };

  console.log('Waiting confirmation...');
  let receipt;
  try {
    // Watch on regular RPC — MEV-protect endpoints may not surface the tx
    // until it's actually mined; the public RPC sees mempool/block sooner.
    receipt = await provider.waitForTransaction(tx.hash);
  } catch (e) {
    console.error('tx.wait failed:', e.shortMessage || e.message);
    await notifyTelegram(
      `⚠️ Tx wait failed: ${e.shortMessage || e.message}\n` +
      `<a href="https://etherscan.io/tx/${tx.hash}">Tx ${tx.hash}</a>`
    );
    continue;
  }

  if (receipt.status !== 1) {
    console.error('Tx reverted on-chain');
    await notifyTelegram(
      `❌ Tx reverted on-chain.\n` +
      `<a href="https://etherscan.io/tx/${tx.hash}">Tx ${tx.hash}</a>\n` +
      `Gas used: ${receipt.gasUsed.toString()}`
    );
    continue;
  }

  console.log('');
  console.log('Transaction confirmed.');
  console.log('Block:    ', receipt.blockNumber);
  console.log('Gas used: ', receipt.gasUsed.toString());
  console.log('Etherscan:', `https://etherscan.io/tx/${tx.hash}`);

  let caught = false;
  for (const log of receipt.logs) {
    try {
      const parsed = contract.interface.parseLog(log);
      if (parsed?.name === 'RabbitCaught') {
        caught = true;
        const totonId = parsed.args.totonId.toString();
        const osUrl = `https://opensea.io/assets/ethereum/${CONTRACT}/${totonId}`;
        console.log('');
        console.log('BAD RABBIT CAUGHT');
        console.log('Toton ID: ', totonId);
        console.log('Hunter:   ', parsed.args.hunter);
        console.log('Nonce:    ', parsed.args.nonce.toString());
        console.log('Proof:    ', parsed.args.proofHash);
        console.log('OpenSea:  ', osUrl);
        await notifyTelegram(
          `🐰💥 <b>BAD RABBIT CAUGHT!</b>\n` +
          `Toton ID: <b>#${totonId}</b>\n` +
          `Hunter: <code>${parsed.args.hunter}</code>\n` +
          `Nonce: <code>${parsed.args.nonce.toString()}</code>\n` +
          `Hunt fee: ${ethers.formatEther(parsed.args.huntFee)} ETH\n\n` +
          `<a href="${osUrl}">View on OpenSea</a>\n` +
          `<a href="https://etherscan.io/tx/${tx.hash}">View tx on Etherscan</a>`
        );
      }
    } catch {}
  }

  if (!caught) {
    await notifyTelegram(
      `⚠️ Tx confirmed but no RabbitCaught event found.\n` +
      `<a href="https://etherscan.io/tx/${tx.hash}">Tx ${tx.hash}</a>`
    );
  }

  break;
}
