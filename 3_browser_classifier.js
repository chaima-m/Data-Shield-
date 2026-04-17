/**
 * DataShield AI — Browser Classifier (MiniLM + SetFit)
 * ======================================================
 * Works with the ONNX export from datashield_fast.py
 *
 * Architecture:
 *   1. Tokenize text → input_ids, attention_mask
 *   2. Run backbone.onnx → last_hidden_state (mean pooling → embedding)
 *   3. Run embedding through JS logistic regression head (loaded from config)
 *
 * Files needed in extension /models/ dir:
 *   backbone_quantized.onnx
 *   tokenizer.json
 *   vocab.txt
 *   datashield_config.json   ← contains label map + classifier weights
 *
 * npm install onnxruntime-web franc-min
 */

import * as ort from 'onnxruntime-web';
import { franc } from 'franc-min';

// ─── Constants ────────────────────────────────────────────────────────────────

const LABELS     = ['safe', 'pii', 'financial', 'confidential', 'health', 'credentials'];
const MAX_LENGTH = 128;    // MiniLM works well at 128
const CACHE_MAX  = 500;
const CONF_THRESHOLD = 0.60;

const MODEL_PATH  = chrome?.runtime?.getURL?.('models/backbone_quantized.onnx') ?? '/models/backbone_quantized.onnx';
const VOCAB_PATH  = chrome?.runtime?.getURL?.('models/vocab.txt')               ?? '/models/vocab.txt';
const CONFIG_PATH = chrome?.runtime?.getURL?.('models/datashield_config.json')  ?? '/models/datashield_config.json';

// ─── Language Detection ───────────────────────────────────────────────────────

export function detectLanguage(text) {
  if (!text || text.trim().length < 15) return 'en';
  const iso3 = franc(text.trim(), { only: ['eng', 'fra', 'arb'], minLength: 10 });
  return { eng: 'en', fra: 'fr', arb: 'ar' }[iso3] || 'en';
}

// ─── WordPiece Tokenizer ──────────────────────────────────────────────────────

class WordPieceTokenizer {
  constructor() {
    this.vocab = new Map();
    this.unkId = 100; this.padId = 0; this.clsId = 101; this.sepId = 102;
  }

  async load(url) {
    const text  = await fetch(url).then(r => r.text());
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
    lines.forEach((w, i) => this.vocab.set(w, i));
    this.unkId = this.vocab.get('[UNK]') ?? 100;
    this.padId = this.vocab.get('[PAD]') ?? 0;
    this.clsId = this.vocab.get('[CLS]') ?? 101;
    this.sepId = this.vocab.get('[SEP]') ?? 102;
  }

  encode(text, maxLen = MAX_LENGTH) {
    const words   = this._basicTokenize(text);
    const tokens  = this._wordpiece(words);
    const ids     = [this.clsId, ...tokens, this.sepId].slice(0, maxLen);
    const padded  = new Array(maxLen).fill(this.padId);
    const mask    = new Array(maxLen).fill(0);
    ids.forEach((id, i) => { padded[i] = id; mask[i] = 1; });
    return { inputIds: new Int32Array(padded), attentionMask: new Int32Array(mask) };
  }

  _basicTokenize(text) {
    return text.toLowerCase().normalize('NFC')
      .replace(/[^\p{L}\p{N}\s]/gu, ' $& ')
      .split(/\s+/).filter(Boolean);
  }

  _wordpiece(words) {
    const ids = [];
    for (const word of words) {
      if (this.vocab.has(word)) { ids.push(this.vocab.get(word)); continue; }
      let rem = word, parts = [], fail = false;
      while (rem.length > 0) {
        let found = false;
        for (let e = rem.length; e > 0; e--) {
          const sub = parts.length === 0 ? rem.slice(0, e) : '##' + rem.slice(0, e);
          if (this.vocab.has(sub)) { parts.push(this.vocab.get(sub)); rem = rem.slice(e); found = true; break; }
        }
        if (!found) { fail = true; break; }
      }
      ids.push(...(fail ? [this.unkId] : parts));
    }
    return ids;
  }
}

// ─── Softmax & Mean Pool ──────────────────────────────────────────────────────

function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

/**
 * Mean pool the last_hidden_state using attention mask.
 * Output: Float32Array of shape [hidden_size]
 */
function meanPool(hiddenState, attentionMask, seqLen, hiddenSize) {
  const embedding = new Float32Array(hiddenSize);
  let count = 0;
  for (let t = 0; t < seqLen; t++) {
    if (attentionMask[t] === 0) continue;
    for (let h = 0; h < hiddenSize; h++) {
      embedding[h] += hiddenState[t * hiddenSize + h];
    }
    count++;
  }
  if (count > 0) {
    for (let h = 0; h < hiddenSize; h++) embedding[h] /= count;
  }
  return embedding;
}

// ─── Lightweight JS Classifier Head ──────────────────────────────────────────
/**
 * After ONNX export we need to run the SetFit classification head in JS.
 * SetFit uses a LogisticRegression head. We serialize its weights to JSON
 * and load them here for dot-product classification.
 *
 * If weights aren't available (first run), falls back to cosine similarity
 * against prototype embeddings (also stored in config).
 */
class ClassifierHead {
  constructor() {
    this.weights  = null;   // shape: [n_classes, hidden_size]
    this.biases   = null;   // shape: [n_classes]
    this.protos   = null;   // fallback: prototype embeddings
    this.ready    = false;
  }

  async load(configUrl) {
    try {
      const cfg = await fetch(configUrl).then(r => r.json());
      if (cfg.weights && cfg.biases) {
        this.weights = cfg.weights;   // Array[6][hidden_size]
        this.biases  = cfg.biases;    // Array[6]
        this.ready   = true;
        console.log('[DataShield] Classifier head loaded (logistic regression)');
      } else if (cfg.prototypes) {
        this.protos = cfg.prototypes; // Array[6][hidden_size]
        this.ready  = true;
        console.log('[DataShield] Classifier head loaded (prototype similarity)');
      }
    } catch (e) {
      console.warn('[DataShield] No classifier head found, using max-norm fallback');
    }
  }

  predict(embedding) {
    // Option A: Logistic regression
    if (this.weights && this.biases) {
      const logits = this.weights.map((row, i) => {
        let dot = this.biases[i];
        for (let j = 0; j < row.length; j++) dot += row[j] * embedding[j];
        return dot;
      });
      return softmax(logits);
    }

    // Option B: Cosine similarity to prototypes
    if (this.protos) {
      const embNorm = Math.sqrt(embedding.reduce((s, v) => s + v * v, 0)) + 1e-8;
      const sims = this.protos.map(proto => {
        const pNorm = Math.sqrt(proto.reduce((s, v) => s + v * v, 0)) + 1e-8;
        const dot   = proto.reduce((s, v, j) => s + v * embedding[j], 0);
        return dot / (embNorm * pNorm);
      });
      return softmax(sims);
    }

    // Fallback: uniform distribution (should not happen in production)
    return new Array(LABELS.length).fill(1 / LABELS.length);
  }
}

// ─── Main Classifier ─────────────────────────────────────────────────────────

export class DataShieldClassifier {
  constructor() {
    this.session   = null;
    this.tokenizer = new WordPieceTokenizer();
    this.head      = new ClassifierHead();
    this.cache     = new Map();
    this.ready     = false;
    this.hiddenSize = 384;   // MiniLM-L12 hidden size
  }

  async initialize() {
    try {
      ort.env.wasm.numThreads = 2;
      ort.env.wasm.simd       = true;

      await Promise.all([
        ort.InferenceSession.create(MODEL_PATH, {
          executionProviders:     ['wasm'],
          graphOptimizationLevel: 'all',
        }).then(s => { this.session = s; }),
        this.tokenizer.load(VOCAB_PATH),
        this.head.load(CONFIG_PATH),
      ]);

      // Detect hidden size from model metadata if possible
      const outputNames = this.session.outputNames;
      console.log('[DataShield] ONNX outputs:', outputNames);

      this.ready = true;
      console.log('[DataShield] Classifier ready ✓');
    } catch (err) {
      console.error('[DataShield] Init failed:', err);
    }
  }

  async classify(text) {
    if (!this.ready) return this._fallback(text, 'not_ready');

    try {
      const lang    = detectLanguage(text);
      const enc     = this.tokenizer.encode(text, MAX_LENGTH);

      const inputIds = new ort.Tensor('int64',
        BigInt64Array.from(enc.inputIds, v => BigInt(v)), [1, MAX_LENGTH]);
      const attnMask = new ort.Tensor('int64',
        BigInt64Array.from(enc.attentionMask, v => BigInt(v)), [1, MAX_LENGTH]);

      const feeds = {};
      const names = this.session.inputNames;
      if (names.includes('input_ids'))      feeds.input_ids      = inputIds;
      if (names.includes('attention_mask')) feeds.attention_mask = attnMask;

      const output      = await this.session.run(feeds);
      const outputKey   = this.session.outputNames[0];   // last_hidden_state
      const rawData     = output[outputKey].data;        // Float32Array, shape [1, seq, hidden]

      // Mean pool: [1, seq, hidden] → [hidden]
      const embedding = meanPool(rawData, enc.attentionMask, MAX_LENGTH, this.hiddenSize);

      // Classify
      const probs   = this.head.predict(embedding);
      const predIdx = probs.indexOf(Math.max(...probs));

      return {
        label:       LABELS[predIdx],
        confidence:  probs[predIdx],
        isSensitive: predIdx !== 0,
        language:    lang,
        source:      'llm',
        allScores:   Object.fromEntries(LABELS.map((l, i) => [l, probs[i]])),
      };
    } catch (err) {
      console.error('[DataShield] Inference error:', err);
      return this._fallback(text, 'error');
    }
  }

  async classifyWithCache(text) {
    const key = text.trim().toLowerCase().slice(0, 200);
    if (this.cache.has(key)) return { ...this.cache.get(key), cached: true };
    const result = await this.classify(text);
    if (this.cache.size >= CACHE_MAX) this.cache.delete(this.cache.keys().next().value);
    this.cache.set(key, result);
    return result;
  }

  async classifyDebounced(text, delay = 300) {
    clearTimeout(this._timer);
    return new Promise(r => { this._timer = setTimeout(() => this.classifyWithCache(text).then(r), delay); });
  }

  _fallback(text, reason) {
    return { label: 'unknown', confidence: 0, isSensitive: false, language: detectLanguage(text), source: reason };
  }
}

export const classifier = new DataShieldClassifier();
