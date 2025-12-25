import { useEffect, useRef, useState, type MouseEvent } from 'react';
import './styles/global.css';
import ChatInterface, { type ChatInterfaceHandle } from './components/ChatInterface';
import { fetchHealth, type HealthResponse } from './api/client';

function App(): JSX.Element {
  const chatRef = useRef<ChatInterfaceHandle | null>(null);
  const [serverInfo, setServerInfo] = useState<HealthResponse | null>(null);

  const handleClear = (): void => {
    chatRef.current?.handleClear();
  };

  const handleMouseEnter = (event: MouseEvent<HTMLButtonElement>): void => {
    event.currentTarget.style.color = 'var(--color-text-primary)';
  };

  const handleMouseLeave = (event: MouseEvent<HTMLButtonElement>): void => {
    event.currentTarget.style.color = 'var(--color-text-secondary)';
  };

  useEffect((): void => {
    fetchHealth().then((info: HealthResponse | null): void => {
      setServerInfo(info);
    });
  }, []);

  return (
    <div className="app-container" style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: 'radial-gradient(circle at 50% -20%, #1a1a2e 0%, var(--color-bg-deep) 60%)'
    }}>
      <header style={{
        padding: '1.5rem 2rem',
        borderBottom: '1px solid var(--color-glass-border)',
        backdropFilter: 'blur(10px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1.25rem',
            fontWeight: 500,
            letterSpacing: '0.02em',
            color: 'var(--color-text-primary)'
          }}>
            LawQA RAG Studio
          </h1>
          <div style={{
            display: 'flex',
            gap: '0.75rem',
            fontSize: '0.75rem',
            color: 'var(--color-text-secondary)',
            letterSpacing: '0.02em',
            fontFamily: 'var(--font-sans)'
          }}>
            <span>Model: {serverInfo?.model ?? '—'}</span>
            <span>Collection: {serverInfo?.collection ?? '—'}</span>
          </div>
        </div>
        <button
          onClick={handleClear}
          style={{
            background: 'transparent',
            border: 'none',
            padding: '0.5rem 1rem',
            color: 'var(--color-text-secondary)',
            fontSize: '0.875rem',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            fontFamily: 'var(--font-sans)'
          }}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          Clear
        </button>
      </header>

      <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <ChatInterface ref={chatRef} />
      </main>
    </div>
  );
}

export default App;
