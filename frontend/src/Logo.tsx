export function Logo({ wordmark = false, size = 28 }: { wordmark?: boolean; size?: number }) {
  return <span className="brand" aria-label="OmniRAG"><svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M16 2.7 28.8 10v12L16 29.3 3.2 22V10L16 2.7Z" stroke="currentColor" strokeWidth="1.6"/><path d="m16 7.7 8.4 4.8v7L16 24.3l-8.4-4.8v-7L16 7.7Z" stroke="currentColor" strokeWidth="1.5"/><path d="m16 12.1 4.6 2.6v2.6L16 19.9l-4.6-2.6v-2.6l4.6-2.6Z" fill="currentColor"/></svg>{wordmark && <span className="wordmark">OmniRAG</span>}</span>;
}
