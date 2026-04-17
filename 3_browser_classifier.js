/**
 * DataShield AI — Browser Extension LLM Integration
 * ===================================================
 * Runs the quantized ONNX classifier entirely in-browser via onnxruntime-web (WebAssembly).
 * Works with the fine-tuned distilbert-base-multilingual-cased or xlm-roberta-base model.
 *
 * Setup in your extension:
 *   npm install onnxruntime-web franc-min
 *   Place model files from model_onnx_quantized/ in extension's /models/ directory:
 *     - model_quantized.onnx
 *     - tokenizer.json
 *     - tokenizer_config.json
 *     - vocab.txt (or sentencepiece.bpe.model for XLM-R)
 *     - special_tokens_map.json
 *
 * Usage:
 *   import { DataShieldClassifier } from './classifier.js';
 *   const clf = new DataShieldClassifier();
 *   await clf.initialize();
 *   const result = await clf.classifyWithCache("My SSN is 123-45-6789");
 *   // { label: 'pii', confidence: 0.97, isSensitive: true, language: 'en' }
 */

import * as ort from 'onnxruntime-web';
import { franc } from 'franc-min';

// ─── Constants ────────────────────────────────────────────────────────────────

const LABELS      = ['safe', 'pii', 'financial', 'confidential', 'health', 'credentials'];
const MAX_LENGTH  = 256;
const CACHE_MAX   = 500;
const MODEL_PATH  = chrome.runtime.getURL('models/model_quantized.onnx');
const VOCAB_PATH  = chrome.runtime.getURL('models/vocab.txt');

// Confidence threshold below which we fall back to pattern engine
const LLM_CONFIDENCE_THRESHOLD = 0.65;

// ─── Language Detection ───────────────────────────────────────────────────────

/**
 * Detect language using franc (ISO 639-3 → our 2-letter code).
 * Falls back to 'en' for short texts or unknown languages.
 */
export function detectLanguage(text) {
  if (!text || text.trim().length < 15) return 'en';
  const iso3 = franc(text.trim(), { only: ['eng', 'fra', 'arb'], minLength: 10 });
  const map = { eng: 'en', fra: 'fr', arb: 'ar' };
  return map[iso3] || 'en';
}

// ─── Minimal WordPiece Tokenizer ──────────────────────────────────────────────
// A lightweight tokenizer that covers DistilmBERT vocab without heavy dependencies.
// For XLM-RoBERTa, replace this with the SentencePiece-based tokenizer.

export class MinimalWordPieceTokenizer {
  constructor() {
    this.vocab     = null;   // word → id
    this.invVocab  = null;   // id → word
    this.unkId     = 100;
    this.padId     = 0;
    this.clsId     = 101;
    this.sepId     = 102;
  }

  async load(vocabUrl) {
    const resp  = await fetch(vocabUrl);
    const text  = await resp.text();
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
    this.vocab    = new Map(lines.map((word, idx) => [word, idx]));
    this.invVocab = new Map(lines.map((word, idx) => [idx, word]));
    // Resolve special token IDs from actual vocab (handles different orderings)
    this.unkId = this.vocab.get('[UNK]') ?? 100;
    this.padId = this.vocab.get('[PAD]') ?? 0;
    this.clsId = this.vocab.get('[CLS]') ?? 101;
    this.sepId = this.vocab.get('[SEP]') ?? 102;
  }

  /**
   * Encode text using WordPiece tokenization.
   * Returns { inputIds: Int32Array, attentionMask: Int32Array }
   */
  encode(text, maxLength = MAX_LENGTH) {
    const tokens   = this._wordpiece(this._basicTokenize(text));
    const ids      = [this.clsId, ...tokens, this.sepId];
    const truncated = ids.slice(0, maxLength);
    const padded   = new Array(maxLength).fill(this.padId);
    const mask     = new Array(maxLength).fill(0);
    for (let i = 0; i < truncated.length; i++) {
      padded[i] = truncated[i];
      mask[i]   = 1;
    }
    return {
      inputIds:      new Int32Array(padded),
      attentionMask: new Int32Array(mask),
    };
  }

  _basicTokenize(text) {
    // Lowercase + whitespace tokenize (multilingual compatible)
    return text
      .toLowerCase()
      .normalize('NFC')
      .replace(/[^\p{L}\p{N}\s]/gu, ' $& ')  // split on punctuation
      .split(/\s+/)
      .filter(Boolean);
  }

  _wordpiece(words) {
    const ids = [];
    for (const word of words) {
      if (this.vocab.has(word)) {
        ids.push(this.vocab.get(word));
        continue;
      }
      // WordPiece segmentation
      let remaining = word;
      let wordIds   = [];
      let failed    = false;
      while (remaining.length > 0) {
        let found = false;
        for (let end = remaining.length; end > 0; end--) {
          const sub     = remaining.slice(0, end);
          const subKey  = wordIds.length === 0 ? sub : '##' + sub;
          if (this.vocab.has(subKey)) {
            wordIds.push(this.vocab.get(subKey));
            remaining = remaining.slice(end);
            found = true;
            break;
          }
        }
        if (!found) { failed = true; break; }
      }
      ids.push(...(failed ? [this.unkId] : wordIds));
    }
    return ids;
  }
}

// ─── Softmax Helper ───────────────────────────────────────────────────────────

function softmax(arr) {
  const max  = Math.max(...arr);
  const exp  = arr.map(x => Math.exp(x - max));
  const sum  = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

// ─── Main Classifier Class ────────────────────────────────────────────────────

export class DataShieldClassifier {
  constructor() {
    this.session   = null;
    this.tokenizer = new MinimalWordPieceTokenizer();
    this.cache     = new Map();
    this.ready     = false;
    this.initError = null;
  }

  /**
   * Initialize the ONNX session and tokenizer.
   * Call once on extension startup (background service worker).
   */
  async initialize() {
    try {
      // Configure ONNX Runtime: WebAssembly (CPU, offline-safe)
      ort.env.wasm.numThreads  = 2;
      ort.env.wasm.simd        = true;
      ort.env.wasm.proxy       = false;  // run in same thread as caller
      ort.env.wasm.wasmPaths   = chrome.runtime.getURL('wasm/');

      // Load model and tokenizer in parallel
      const [session] = await Promise.all([
        ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: true,
        }),
        this.tokenizer.load(VOCAB_PATH),
      ]);

      this.session = session;
      this.ready   = true;
      console.log('[DataShield] LLM classifier initialized ✓');
    } catch (err) {
      this.initError = err;
      console.error('[DataShield] Classifier init failed:', err);
      // Non-fatal: pattern-based detection will still work
    }
  }

  /**
   * Classify a text string.
   * Returns: { label, confidence, isSensitive, language, source }
   *   source: 'llm' | 'fallback_not_ready' | 'fallback_error'
   */
  async classify(text) {
    if (!this.ready || !this.session) {
      return this._safeFallback(text, 'fallback_not_ready');
    }

    try {
      const lang    = detectLanguage(text);
      const encoded = this.tokenizer.encode(text, MAX_LENGTH);

      // Build ONNX tensors (int64 required by most transformers ONNX exports)
      const inputIds = new ort.Tensor(
        'int64',
        BigInt64Array.from(encoded.inputIds, v => BigInt(v)),
        [1, MAX_LENGTH]
      );
      const attentionMask = new ort.Tensor(
        'int64',
        BigInt64Array.from(encoded.attentionMask, v => BigInt(v)),
        [1, MAX_LENGTH]
      );

      // Determine input names from session metadata
      const inputNames = this.session.inputNames;
      const feeds = {};
      if (inputNames.includes('input_ids'))      feeds['input_ids']      = inputIds;
      if (inputNames.includes('attention_mask'))  feeds['attention_mask']  = attentionMask;
      // token_type_ids: optional, set to zeros if required
      if (inputNames.includes('token_type_ids')) {
        feeds['token_type_ids'] = new ort.Tensor(
          'int64',
          new BigInt64Array(MAX_LENGTH).fill(0n),
          [1, MAX_LENGTH]
        );
      }

      const output  = await this.session.run(feeds);
      const logits  = Array.from(output.logits?.data ?? output[Object.keys(output)[0]].data);
      const probs   = softmax(logits);
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
      return this._safeFallback(text, 'fallback_error');
    }
  }

  /**
   * classify() + LRU cache.
   * Cache key is the first 200 chars of the normalized text.
   */
  async classifyWithCache(text) {
    const key = text.trim().toLowerCase().slice(0, 200);

    if (this.cache.has(key)) {
      return { ...this.cache.get(key), cached: true };
    }

    const result = await this.classify(text);
    this._cacheSet(key, result);
    return result;
  }

  /**
   * Classify a text in streaming/debounced mode.
   * Cancels any pending classification and starts a new one after `delayMs`.
   */
  async classifyDebounced(text, delayMs = 300) {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    return new Promise(resolve => {
      this._debounceTimer = setTimeout(async () => {
        const result = await this.classifyWithCache(text);
        resolve(result);
      }, delayMs);
    });
  }

  clearCache() {
    this.cache.clear();
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  _cacheSet(key, value) {
    if (this.cache.size >= CACHE_MAX) {
      // Evict oldest entry (Map preserves insertion order)
      this.cache.delete(this.cache.keys().next().value);
    }
    this.cache.set(key, value);
  }

  _safeFallback(text, reason) {
    return {
      label:       'unknown',
      confidence:  0,
      isSensitive: false,
      language:    detectLanguage(text),
      source:      reason,
    };
  }
}

// ─── Singleton export ─────────────────────────────────────────────────────────
// Import and call initialize() once in your background service worker.

export const classifier = new DataShieldClassifier();
