import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';

describe('API error responses', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('surfaces ingestion failure details from a failed document response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        status: 'failed',
        error_message: 'Embedding provider unavailable',
      }),
    }));

    await expect(api.documents()).rejects.toThrow('Embedding provider unavailable');
  });
});
