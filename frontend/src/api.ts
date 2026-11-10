import type { AppSettings, ChatResponse, Collection, Conversation, DocumentItem, EvaluationStatus, Preferences, Profile } from './types';

const base = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 120_000);
  try {
    const response = await fetch(`${base}/api/v1${path}`, { ...init, signal: controller.signal, headers: { ...(init.body && !(init.body instanceof Blob) ? { 'Content-Type': 'application/json' } : {}), ...init.headers } });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.error?.message ?? body?.detail ?? body?.error_message ?? `Request failed (${response.status})`);
    }
    return response.status === 204 ? undefined as T : response.json();
  } finally { window.clearTimeout(timeout); }
}
export const api = {
  preferences: () => request<Preferences>('/preferences'),
  savePreferences: (value: Partial<Preferences>) => request<Preferences>('/preferences', { method: 'PUT', body: JSON.stringify(value) }),
  conversations: () => request<Conversation[]>('/conversations'),
  conversation: (id: string) => request<Conversation & { messages: Array<Record<string, any>> }>(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  chat: (body: { query_text: string; selected_profile: Profile; conversation_id?: string; collection_id?: string }) => request<ChatResponse>('/chat', { method: 'POST', body: JSON.stringify(body) }),
  documents: () => request<DocumentItem[]>('/documents'),
  deleteDocument: (id: string) => request<void>(`/documents/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  reingest: (id: string) => request<DocumentItem>(`/documents/${encodeURIComponent(id)}/reingest`, { method: 'POST' }),
  collections: () => request<Collection[]>('/collections'),
  createCollection: (name: string, description: string) => request<Collection>('/collections', { method: 'POST', body: JSON.stringify({ name, description }) }),
  assignDocument: (collectionId: string, documentId: string) => request<void>(`/collections/documents/${encodeURIComponent(documentId)}`, { method: 'PUT', body: JSON.stringify({ collection_id: collectionId }) }),
  deleteCollection: (id: string) => request<void>(`/collections/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  evaluation: () => request<EvaluationStatus>('/evaluations/status'),
  runEvaluation: () => request<unknown>('/evaluations/runs', { method: 'POST' }),
  profiles: () => request<{ profiles: Record<string, unknown> }>('/profiles'),
  settings: () => request<AppSettings>('/settings'),
};
export function uploadFile(file: File, collectionId: string | undefined, onProgress: (value: number) => void): Promise<DocumentItem> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const query = new URLSearchParams({ filename: file.name });
    if (collectionId) query.set('collection_id', collectionId);
    xhr.open('PUT', `${base}/api/v1/documents/upload?${query}`);
    xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100)); };
    xhr.onerror = () => reject(new Error('Upload failed. Check your connection and retry.'));
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) { try { reject(new Error(JSON.parse(xhr.responseText)?.detail ?? `Upload failed (${xhr.status})`)); } catch { reject(new Error(`Upload failed (${xhr.status})`)); } return; }
      try { resolve(JSON.parse(xhr.responseText)); } catch { reject(new Error('Upload returned an invalid response.')); }
    };
    xhr.send(file);
  });
}
