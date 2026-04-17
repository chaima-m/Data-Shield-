/**
 * DataShield AI — Combined Detection Engine
 * ==========================================
 * Merges all three detection layers:
 *   Layer 1: Pattern / regex engine (instant, always on)
 *   Layer 2: LLM classifier (ONNX, ~50-100ms)
 *   Layer 3: Company profile rule engine (loaded from admin-generated profile.json)
 *
 * All three run in parallel; ANY layer flagging sensitive → block prompt.
 *
 * Usage (background service worker):
 *   import { DetectionEngine } from './detection_engine.js';
 *   const engine = new DetectionEngine();
 *   await engine.initialize();
 *
 *   // In content script, via message passing:
 *   const result = await engine.analyze(promptText);
 */

import { classifier, detectLanguage } from './3_browser_classifier.js';

// ─── Layer 1: Pattern Engine ──────────────────────────────────────────────────

const PATTERNS = {
  // Credit / debit cards (Luhn-validated separately)
  credit_card: {
    regex:   /\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b/g,
    label:   'financial',
    severity: 'critical',
  },
  // IBAN
  iban: {
    regex:   /\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}(?:[A-Z0-9]{0,16})\b/g,
    label:   'financial',
    severity: 'critical',
  },
  // US SSN
  ssn_us: {
    regex:   /\b(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}\b/g,
    label:   'pii',
    severity: 'critical',
  },
  // Email addresses
  email: {
    regex:   /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g,
    label:   'pii',
    severity: 'medium',
  },
  // Phone numbers (international)
  phone: {
    regex:   /(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{2,4}[\s-]?\d{2,4}/g,
    label:   'pii',
    severity: 'medium',
  },
  // API keys / tokens (entropy-based prefix matching)
  api_key: {
    regex:   /(?:sk[-_](?:prod|live|test|proj)[-_][A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{36}|AIzaSy[A-Za-z0-9_-]{33}|AKIA[0-9A-Z]{16}|ya29\.[A-Za-z0-9_-]{50,})/g,
    label:   'credentials',
    severity: 'critical',
  },
  // Passwords in plain text
  password_plain: {
    regex:   /(?:password|passwd|pwd|mdp|كلمة المرور)\s*[:=]\s*\S+/gi,
    label:   'credentials',
    severity: 'critical',
  },
  // DB connection strings
  db_conn: {
    regex:   /(?:postgres(?:ql)?|mysql|mongodb|redis|mssql):\/\/[^:\s]+:[^@\s]+@[^\s]+/gi,
    label:   'credentials',
    severity: 'critical',
  },
  // IPv4 addresses (internal ranges → higher risk)
  ip_private: {
    regex:   /\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b/g,
    label:   'confidential',
    severity: 'medium',
  },
};

/**
 * Luhn algorithm to validate credit card numbers.
 */
function luhnCheck(num) {
  const digits = num.replace(/\D/g, '').split('').map(Number);
  let sum = 0;
  digits.reverse().forEach((d, i) => {
    if (i % 2 === 1) { d *= 2; if (d > 9) d -= 9; }
    sum += d;
  });
  return sum % 10 === 0;
}

export class PatternEngine {
  analyze(text) {
    const matches = [];

    for (const [name, config] of Object.entries(PATTERNS)) {
      let match;
      const regex = new RegExp(config.regex.source, config.regex.flags);
      while ((match = regex.exec(text)) !== null) {
        // Extra validation for credit cards
        if (name === 'credit_card' && !luhnCheck(match[0])) continue;

        matches.push({
          patternName: name,
          matchedText: match[0],
          startIndex:  match.index,
          endIndex:    match.index + match[0].length,
          label:       config.label,
          severity:    config.severity,
        });
      }
    }

    return {
      matched:   matches.length > 0,
      matches,
      label:     matches.length > 0 ? matches[0].label : 'safe',
      source:    'pattern',
    };
  }
}

// ─── Layer 3: Company Rule Engine ────────────────────────────────────────────

export class CompanyRuleEngine {
  constructor() {
    this.loaded        = false;
    this.exactTerms    = [];      // lowercase strings
    this.regexPatterns = [];      // compiled RegExp objects
    this.keywordSet    = new Set(); // lowercase keywords
  }

  /**
   * Load company profile from local storage (populated by admin tool).
   * profile.json format matches the Python admin_processor output.
   */
  async loadProfile() {
    try {
      const result = await chrome.storage.local.get('companyProfile');
      if (!result.companyProfile) {
        console.log('[DataShield] No company profile loaded.');
        return;
      }
      const profile = JSON.parse(result.companyProfile);
      this._compile(profile);
      this.loaded = true;
      console.log(
        `[DataShield] Company profile loaded: ${this.exactTerms.length} exact terms, ` +
        `${this.regexPatterns.length} patterns, ${this.keywordSet.size} keywords`
      );
    } catch (err) {
      console.error('[DataShield] Failed to load company profile:', err);
    }
  }

  _compile(profile) {
    const rules = profile.detection_rules || {};

    // Exact match terms (entity names, client codes, etc.)
    this.exactTerms = (rules.exact_match || []).map(e =>
      (e.text || e).toLowerCase().trim()
    ).filter(Boolean);

    // Regex patterns from admin-extracted internal codes
    this.regexPatterns = (rules.pattern_match || []).map(p => {
      try { return new RegExp(p.pattern, 'gi'); }
      catch { return null; }
    }).filter(Boolean);

    // Keyword set for fast membership test
    this.keywordSet = new Set(
      (rules.keyword_match || []).map(k => k.toLowerCase())
    );
  }

  analyze(text) {
    if (!this.loaded) {
      return { matched: false, source: 'company_rules', reason: 'no_profile' };
    }

    const lower = text.toLowerCase();

    // Exact match
    for (const term of this.exactTerms) {
      if (term.length > 3 && lower.includes(term)) {
        return {
          matched:     true,
          matchedTerm: term,
          label:       'confidential',
          severity:    'high',
          source:      'company_rules',
          reason:      'exact_match',
        };
      }
    }

    // Regex pattern match
    for (const pattern of this.regexPatterns) {
      if (pattern.test(text)) {
        return {
          matched:     true,
          matchedTerm: pattern.source,
          label:       'confidential',
          severity:    'high',
          source:      'company_rules',
          reason:      'pattern_match',
        };
      }
    }

    // Keyword density check (≥2 sensitive keywords → flag)
    const words = lower.split(/\s+/);
    const hits  = words.filter(w => this.keywordSet.has(w));
    if (hits.length >= 2) {
      return {
        matched:     true,
        matchedTerm: hits.join(', '),
        label:       'confidential',
        severity:    'medium',
        source:      'company_rules',
        reason:      'keyword_density',
      };
    }

    return { matched: false, source: 'company_rules' };
  }
}

// ─── Combined Engine ──────────────────────────────────────────────────────────

export class DetectionEngine {
  constructor() {
    this.patternEngine = new PatternEngine();
    this.companyEngine = new CompanyRuleEngine();
    this._initPromise  = null;
  }

  async initialize() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = Promise.all([
      classifier.initialize(),
      this.companyEngine.loadProfile(),
    ]);
    await this._initPromise;
    console.log('[DataShield] Detection engine ready.');
  }

  /**
   * Run all three layers in parallel.
   * Returns a unified AnalysisResult.
   */
  async analyze(text) {
    if (!text || text.trim().length < 3) {
      return this._buildResult(false, 'safe', 'none', [], {});
    }

    // Run all three layers simultaneously
    const [patternResult, llmResult, companyResult] = await Promise.all([
      Promise.resolve(this.patternEngine.analyze(text)),
      classifier.classifyWithCache(text),
      Promise.resolve(this.companyEngine.analyze(text)),
    ]);

    // ── Resolve final verdict ────────────────────────────────────────────────
    const isSensitive =
      (patternResult.matched) ||
      (llmResult.isSensitive && llmResult.confidence >= 0.65) ||
      (companyResult.matched);

    // Pick the most severe label
    const label = this._resolveLabel(patternResult, llmResult, companyResult);

    // Collect triggered sources for UI display
    const sources = [];
    if (patternResult.matched)  sources.push('pattern');
    if (llmResult.isSensitive && llmResult.confidence >= 0.65) sources.push('llm');
    if (companyResult.matched)  sources.push('company_profile');

    // Collect all matched spans for highlighting
    const highlights = patternResult.matches.map(m => ({
      start:  m.startIndex,
      end:    m.endIndex,
      text:   m.matchedText,
      reason: m.patternName,
    }));
    if (companyResult.matched && companyResult.matchedTerm) {
      // Find position of company term in text
      const idx = text.toLowerCase().indexOf(companyResult.matchedTerm.toLowerCase());
      if (idx >= 0) {
        highlights.push({
          start:  idx,
          end:    idx + companyResult.matchedTerm.length,
          text:   companyResult.matchedTerm,
          reason: 'company_profile',
        });
      }
    }

    return this._buildResult(isSensitive, label, sources.join('+') || 'none', highlights, {
      pattern: patternResult,
      llm:     llmResult,
      company: companyResult,
    });
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  _resolveLabel(pattern, llm, company) {
    // Priority: critical pattern > credentials > pii > financial > health > confidential > safe
    const PRIORITY = { credentials: 6, pii: 5, financial: 4, health: 3, confidential: 2, safe: 0 };
    const candidates = [
      pattern.matched ? pattern.label : 'safe',
      llm.isSensitive ? llm.label : 'safe',
      company.matched ? (company.label || 'confidential') : 'safe',
    ];
    return candidates.reduce((best, l) =>
      (PRIORITY[l] ?? 0) > (PRIORITY[best] ?? 0) ? l : best, 'safe'
    );
  }

  _buildResult(isSensitive, label, source, highlights, layerDetails) {
    return {
      isSensitive,
      label,
      source,
      highlights,
      layers:     layerDetails,
      timestamp:  Date.now(),
    };
  }
}

// ─── Singleton ────────────────────────────────────────────────────────────────

export const detectionEngine = new DetectionEngine();
