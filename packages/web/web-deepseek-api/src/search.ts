/**
 * DeepSeek Search API provider: one direct request per search, without an
 * auxiliary model request. Falls back to DuckDuckGo HTML search when no valid
 * DeepSeek API credential is available (e.g. local Ollama / custom providers).
 * @module @deepseek-ai/dsh-web-deepseek-api/search
 */
import type { WebSearchProvider, WebSearchRequest, WebSearchResult } from '@deepseek-ai/dsh-web'
import {
  dispatchDeepSeekWeb, hasDeepSeekWebCredential, invalidResponse, isRecord, searchAborted,
  type DeepSeekWebAuth, type DeepSeekWebGuidance,
} from './transport.ts'

/** Stable id the Search API provider registers under in the search registry. */
export const DEEPSEEK_SEARCH_API_PROVIDER_ID = 'deepseek-search-api'

/** Default Search and Fetch API base, including its `/v1/tools` prefix; `/search` and `/fetch` are appended. */
export const DEEPSEEK_WEB_API_BASE_URL = 'https://api.deepseek.com/v1/tools'

/** Default Search API result count, the service's own default. */
export const DEEPSEEK_DEFAULT_COUNT = 10

/** Default Search requests started per second, the service's account-wide 5 QPS limit. */
export const DEEPSEEK_SEARCH_REQUESTS_PER_SECOND = 5

/** Longest `ori_question` the Search API accepts, in characters. */
const ORIGINAL_QUESTION_MAX_CHARS = 128

/** Authentication, endpoint, and result settings snapshotted for one search. */
export interface DeepSeekSearchApiOptions {
  /** Ordered authentication sources of the search, each bound to the Search endpoint its credential is sent to. */
  auth: DeepSeekWebAuth
  /** Settings that endpoint failures of the search name. */
  guidance: DeepSeekWebGuidance
  /** Results requested per search, from 1 through 50; a smaller request bound wins. */
  count: number
  /** Most Search requests this provider starts in any one-second window. */
  requestsPerSecond: number
  /** Read the user question that triggered the search; undefined or blank sends none. */
  originalQuestion?: () => string | undefined
}

/**
 * Validate a Search API response and project its sources into the web service.
 * @param response - untrusted JSON from the Search endpoint.
 * @returns sources with provider order, the display `snippet`, the model-facing `summary`, and nullable
 *   publication dates preserved.
 * @throws {@link WebError} when the response does not contain valid search results.
 */
export function mapSearchResponse(response: unknown): WebSearchResult {
  if (!isRecord(response) || !Array.isArray(response.results)) throw invalidResponse()
  const sources = response.results.map((item: unknown) => {
    if (!isRecord(item) || typeof item.url !== 'string' || item.url.length === 0
      || typeof item.title !== 'string' || typeof item.snippet !== 'string'
      || (item.summary !== undefined && typeof item.summary !== 'string')
      || (item.date !== null && typeof item.date !== 'string')) throw invalidResponse()
    return {
      url: item.url,
      ...item.title.length > 0 ? { title: item.title } : {},
      ...item.snippet.length > 0 ? { snippet: item.snippet } : {},
      ...typeof item.summary === 'string' && item.summary.length > 0 ? { summary: item.summary } : {},
      ...item.date !== null && item.date.length > 0 ? { publishedAt: item.date } : {},
    }
  })
  return { sources, truncated: false }
}

/**
 * Search through the DeepSeek Search API, with automatic fallback to free
 * DuckDuckGo web search when DeepSeek credentials are missing or rejected.
 */
export class DeepSeekSearchApiProvider implements WebSearchProvider {
  readonly id = DEEPSEEK_SEARCH_API_PROVIDER_ID
  /** Monotonic start times (ms) of requests inside the current window, oldest first. */
  private readonly starts: number[] = []
  /** Picker description: what this provider suits and what it costs or cannot do. */
  readonly description: NonNullable<WebSearchProvider['description']> = {
    en: 'Uses DeepSeek Search API when configured, or falls back to free web search (DuckDuckGo).',
    zh: '配置了 DeepSeek API 时使用官方搜索，否则自动回退到免费网页搜索（DuckDuckGo）。',
    ja: '設定時は DeepSeek Search API を使用し、未設定時は無料ウェブ検索（DuckDuckGo）にフォールバックします。',
    es: 'Usa la API de búsqueda de DeepSeek si está configurada, o recurre a búsqueda web gratuita (DuckDuckGo).',
    fr: 'Utilise l’API de recherche DeepSeek si configurée, sinon bascule automatiquement sur la recherche web gratuite (DuckDuckGo).',
    de: 'Nutzt bei Konfiguration die DeepSeek-Such-API oder fällt auf kostenlose Websuche (DuckDuckGo) zurück.',
    ru: 'Использует DeepSeek Search API при наличии ключа или автоматически переключается на бесплатный веб-поиск (DuckDuckGo).',
    pt: 'Usa a API de busca do DeepSeek quando configurada, ou recorre à busca web gratuita (DuckDuckGo).',
  }

  /**
   * @param resolveOptions - the options for the NEXT search, read once at each
   * search's entry so one search never mixes two settings sections.
   */
  constructor(private readonly resolveOptions: () => DeepSeekSearchApiOptions) {}

  available(): boolean {
    const options = this.resolveOptions()
    return isPositiveInteger(options.count) && isPositiveInteger(options.requestsPerSecond)
  }

  async search(request: WebSearchRequest, signal?: AbortSignal): Promise<WebSearchResult> {
    const options = this.resolveOptions()
    const maxResults = Math.min(options.count, request.maxResults ?? options.count)
    if (hasDeepSeekWebCredential(options.auth)) {
      const question = boundedQuestion(options.originalQuestion?.())
      try {
        return await dispatchDeepSeekWeb(options.auth, {
          body: {
            query: request.query,
            count: maxResults,
            need_content: false,
            ...question === undefined ? {} : { ori_question: question },
          },
          decode: mapSearchResponse,
          waitToSend: waitSignal => this.waitForRequestSlot(options.requestsPerSecond, waitSignal),
          guidance: options.guidance,
        }, signal)
      } catch (error) {
        if (signal?.aborted === true) throw error
      }
    }
    return searchDuckDuckGo(request.query, maxResults, signal)
  }

  /**
   * Wait until fewer than `limit` requests started within the last second,
   * inclusive of its first millisecond, then record this start.
   * @throws {@link WebError} `WEB_ABORTED` when `signal` aborts while waiting.
   */
  private async waitForRequestSlot(limit: number, signal?: AbortSignal): Promise<void> {
    for (;;) {
      if (signal?.aborted === true) throw searchAborted(signal)
      const now = performance.now()
      let oldest = this.starts[0]
      while (oldest !== undefined && oldest < now - 1000) {
        this.starts.shift()
        oldest = this.starts[0]
      }
      if (oldest === undefined || this.starts.length < limit) {
        this.starts.push(now)
        return
      }
      await abortableDelay(oldest + 1001 - now, signal)
    }
  }
}

/**
 * Query DuckDuckGo's HTML endpoint and extract `{ url, title, snippet }` results.
 * Used when DeepSeek Search API credentials are absent or rejected (e.g. Ollama / custom APIs).
 */
async function searchDuckDuckGo(query: string, maxResults: number, signal?: AbortSignal): Promise<WebSearchResult> {
  if (signal?.aborted === true) throw searchAborted(signal)
  const endpoint = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`
  const res = await fetch(endpoint, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
      'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
    },
    ...(signal ? { signal } : {}),
  })
  if (!res.ok) {
    throw invalidResponse()
  }
  const html = await res.text()
  const sources: WebSearchResult['sources'] = []
  const linkRegex = /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/giu
  const snippetRegex = /<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/giu

  const links: Array<{ url: string; title: string }> = []
  for (const match of html.matchAll(linkRegex)) {
    const rawHref = decodeHtmlEntities(match[1] ?? '')
    const url = extractDuckDuckGoTarget(rawHref)
    const title = stripHtml(match[2] ?? '')
    if (url.startsWith('http://') || url.startsWith('https://')) {
      links.push({ url, title })
    }
  }

  const snippets: string[] = []
  for (const match of html.matchAll(snippetRegex)) {
    snippets.push(stripHtml(match[1] ?? ''))
  }

  const limit = Math.max(1, Math.min(maxResults, 20))
  for (let i = 0; i < links.length && sources.length < limit; i++) {
    const link = links[i]!
    const snippet = snippets[i] ?? ''
    sources.push({
      url: link.url,
      ...(link.title.length > 0 ? { title: link.title } : {}),
      ...(snippet.length > 0 ? { snippet } : {}),
    })
  }
  return { sources, truncated: false }
}

/** Unwrap DuckDuckGo `/l/?uddg=...` redirect URLs into direct HTTP(S) URLs. */
function extractDuckDuckGoTarget(rawHref: string): string {
  const normalized = rawHref.startsWith('//') ? `https:${rawHref}` : rawHref
  try {
    const parsed = new URL(normalized)
    const uddg = parsed.searchParams.get('uddg')
    if (uddg && (uddg.startsWith('http://') || uddg.startsWith('https://'))) {
      return uddg
    }
  } catch {
    // Fall through to raw href when not a full URL
  }
  return normalized
}

/** Strip HTML tags and collapse whitespace. */
function stripHtml(input: string): string {
  return decodeHtmlEntities(input.replace(/<[^>]+>/gu, ' ')).replace(/\s+/gu, ' ').trim()
}

/** Decode common HTML entities. */
function decodeHtmlEntities(input: string): string {
  return input
    .replace(/&amp;/gu, '&')
    .replace(/&lt;/gu, '<')
    .replace(/&gt;/gu, '>')
    .replace(/&quot;/gu, '"')
    .replace(/&#39;|&apos;/gu, "'")
    .replace(/&nbsp;/gu, ' ')
}

/** Resolve after `ms`, or early when `signal` aborts; the caller rechecks the signal. */
function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const onAbort = () => {
      clearTimeout(timer)
      resolve()
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * Bound the triggering question to what `ori_question` accepts: whitespace
 * collapses to single spaces and the first 128 characters remain.
 * @param question - the user question, when one exists.
 * @returns the bounded question, or undefined when nothing remains.
 */
function boundedQuestion(question: string | undefined): string | undefined {
  const collapsed = question?.replace(/\s+/gu, ' ').trim() ?? ''
  if (collapsed.length === 0) return undefined
  return Array.from(collapsed).slice(0, ORIGINAL_QUESTION_MAX_CHARS).join('')
}

/** True for positive integral request limits. */
function isPositiveInteger(value: number): boolean {
  return Number.isInteger(value) && value > 0
}
