const envApiBase = (import.meta.env.VITE_API_BASE as string | undefined)?.trim();
export const API_BASE = envApiBase || 'http://localhost:8000';

export interface ChunkInfo {
    id: string;
    text: string;
    metadata: Record<string, unknown>;
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    chunks?: ChunkInfo[];
}

export interface ChatResponse {
    answer: string;
    chunks: ChunkInfo[];
    config_hash?: string;
    session_id?: string;
}

export interface HealthResponse {
    status: string;
    model?: string;
    collection?: string;
}

export async function fetchHealth(): Promise<HealthResponse | null> {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.error('Health check failed:', e);
        return null;
    }
}

export async function resetSession(): Promise<string | null> {
    try {
        const res = await fetch(`${API_BASE}/session/reset`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to reset session');
        const data = await res.json();
        return data.session_id;
    } catch (e) {
        console.error('Session reset failed:', e);
        return null;
    }
}

export async function sendMessage(
    message: string,
    sessionId?: string,
    history: ChatMessage[] = []
): Promise<ChatResponse> {
    const historyPayload = history.map((entry: ChatMessage): { role: ChatMessage['role']; content: string } => ({
        role: entry.role,
        content: entry.content
    }));
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message,
            session_id: sessionId,
            history: historyPayload
        })
    });

    if (!res.ok) {
        throw new Error(`Chat API error: ${res.statusText}`);
    }

    return res.json();
}
