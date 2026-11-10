import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

vi.mock('./api', () => ({ api: {
  preferences: vi.fn().mockResolvedValue({ preferred_name: null, onboarding_skipped: false, default_profile: 'core', escalation_preference: 'when_needed' }),
  savePreferences: vi.fn().mockImplementation(async (value) => ({ preferred_name: value.preferred_name ?? null, onboarding_skipped: value.onboarding_skipped ?? false, default_profile: value.default_profile ?? 'core', escalation_preference: value.escalation_preference ?? 'when_needed' })),
  conversations: vi.fn().mockResolvedValue([]), collections: vi.fn().mockResolvedValue([]), documents: vi.fn().mockResolvedValue([]),
  evaluation: vi.fn().mockResolvedValue({ available: false, dataset_available: false, suites: [], message: 'No labeled evaluation data.' }),
}, uploadFile: vi.fn() }));

describe('OmniRAG workspace', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  afterEach(() => cleanup());
  it('asks for a preferred name and persists it before greeting', async () => {
    render(<App/>);
    expect(await screen.findByText('What would you like me to call you?')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Preferred name'), { target: { value: 'Alex' } });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    expect(await screen.findByRole('heading', { name: 'Let’s get started, Alex.' })).toBeInTheDocument();
  });
  it('supports profile selection in the shared composer', async () => {
    render(<App/>);
    await screen.findByRole('heading', { name: /Let’s get started/ });
    fireEvent.click(document.querySelector('.profile-select')!);
    expect(screen.getByRole('button', { name: /Research/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Research/ }));
    expect(document.querySelector('.profile-select')).toHaveAttribute('aria-expanded', 'false');
  });
  it('keeps labeled evaluation unavailable when backend says no dataset', async () => {
    render(<App/>); await screen.findByRole('heading', { name: /Let’s get started/ }); fireEvent.click(screen.getByText('Evaluation'));
    await waitFor(() => expect(screen.getByText('No labeled evaluation data.')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Run evaluation' })).toBeDisabled();
  });
});
