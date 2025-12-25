import { useState, useEffect, useRef, useLayoutEffect, forwardRef, useImperativeHandle, type ForwardedRef } from 'react';
import { Send, Scale } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize, { defaultSchema, type Schema } from 'rehype-sanitize';
import { sendMessage, resetSession, fetchHealth, type ChatMessage, type ChunkInfo } from '../api/client';
import styles from './ChatInterface.module.css';

export interface ChatInterfaceHandle {
    handleClear: () => Promise<void>;
}

interface ChatInterfaceProps {}

const markdownSchema: Schema = {
    ...defaultSchema,
    tagNames: Array.from(defaultSchema.tagNames ?? []).concat([
        'table',
        'thead',
        'tbody',
        'tr',
        'th',
        'td'
    ]),
    attributes: {
        ...defaultSchema.attributes,
        a: Array.from(defaultSchema.attributes?.a ?? []).concat(['href', 'target', 'rel']),
        th: Array.from(defaultSchema.attributes?.th ?? []).concat(['align']),
        td: Array.from(defaultSchema.attributes?.td ?? []).concat(['align']),
        code: Array.from(defaultSchema.attributes?.code ?? []).concat(['className'])
    }
};

const getDisplayScore = (metadata: Record<string, unknown> | undefined): { label: string; value: number } | null => {
    if (!metadata) return null;
    const rerankScore = metadata.rerank_score;
    if (typeof rerankScore === 'number' && Number.isFinite(rerankScore)) {
        return { label: 'rerank', value: rerankScore };
    }
    const baseScore = metadata.score;
    if (typeof baseScore === 'number' && Number.isFinite(baseScore)) {
        return { label: 'score', value: baseScore };
    }
    return null;
};

const ChatInterface = forwardRef<ChatInterfaceHandle, ChatInterfaceProps>((_: ChatInterfaceProps, ref: ForwardedRef<ChatInterfaceHandle>): JSX.Element => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [inputObj, setInputObj] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | undefined>(undefined);
    const [activeChunks, setActiveChunks] = useState<ChunkInfo[] | null>(null);
    const messagesContainerRef = useRef<HTMLDivElement>(null);
    const lastUserMessageRef = useRef<HTMLDivElement | null>(null);
    const spacerRef = useRef<HTMLDivElement | null>(null);
    const firstUserOffsetRef = useRef<number | null>(null);

    useEffect((): void => {
        fetchHealth().then((info): void => {
            if (info?.status === 'ok') console.log('Backend is healthy');
            else console.warn('Backend unavailable');
        });
    }, []);

    const handleClear = async (): Promise<void> => {
        const newId = await resetSession();
        if (newId) {
            setSessionId(newId);
        }
        setMessages([]);
        lastUserMessageRef.current = null;
        firstUserOffsetRef.current = null;
        setActiveChunks(null);
    };

    useImperativeHandle(ref, (): ChatInterfaceHandle => ({
        handleClear
    }));

    const handleSubmit = async (e: React.FormEvent): Promise<void> => {
        e.preventDefault();
        if (!inputObj.trim() || isLoading) return;

        const userText = inputObj;
        setInputObj('');
        setIsLoading(true);

        const userMsg: ChatMessage = { role: 'user', content: userText };
        setMessages((prev: ChatMessage[]): ChatMessage[] => [...prev, userMsg]);
    };

    const setLastUserMessageRef = (node: HTMLDivElement | null): void => {
        if (node) {
            lastUserMessageRef.current = node;
        }
    };

    useLayoutEffect((): void => {
        if (!messagesContainerRef.current || messages.length === 0) return;

        const container = messagesContainerRef.current;
        const spacerEl = spacerRef.current;
        if (!spacerEl) return;

        const computedStyle = getComputedStyle(document.documentElement);
        const offsetVar = computedStyle.getPropertyValue('--scroll-offset');
        const scrollOffset = parseInt(offsetVar, 10) || 260;

        const lastMessage = messages[messages.length - 1];

        if (lastMessage.role !== 'user') return;

        const lastMsgEl = lastUserMessageRef.current;
        if (!lastMsgEl) return;

        if (firstUserOffsetRef.current === null) {
            const visibleOffset = Math.max(0, lastMsgEl.offsetTop - container.scrollTop);
            firstUserOffsetRef.current = visibleOffset;
        }

        const desiredOffset = firstUserOffsetRef.current ?? scrollOffset;
        const targetTop = Math.max(0, lastMsgEl.offsetTop - desiredOffset);
        const maxScrollTop = container.scrollHeight - container.clientHeight;

        if (targetTop > maxScrollTop) {
            const spacerHeight = spacerEl.offsetHeight;
            const extra = targetTop - maxScrollTop;
            const nextHeight = Math.max(scrollOffset, spacerHeight + extra);
            if (spacerEl.style.height !== `${nextHeight}px`) {
                spacerEl.style.height = `${nextHeight}px`;
            }
        }

        requestAnimationFrame((): void => {
            container.scrollTo({ top: targetTop, behavior: 'smooth' });
        });
    }, [messages]);

    const fetchReply = async (text: string): Promise<void> => {
        try {
            const response = await sendMessage(text, sessionId, messages);
            if (response.session_id) setSessionId(response.session_id);

            const aiMsg: ChatMessage = { role: 'assistant', content: response.answer, chunks: response.chunks };
            setMessages((prev: ChatMessage[]): ChatMessage[] => [...prev, aiMsg]);
        } catch (err) {
            console.error(err);
            const errorMsg: ChatMessage = { role: 'assistant', content: "Error: Could not connect to LawQA server." };
            setMessages((prev: ChatMessage[]): ChatMessage[] => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect((): void => {
        if (isLoading && messages.length > 0 && messages[messages.length - 1].role === 'user') {
            fetchReply(messages[messages.length - 1].content);
        }
    }, [messages, isLoading]);

    const renderContent = (msg: ChatMessage): JSX.Element => {
        if (msg.role === 'assistant') {
            return (
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[[rehypeSanitize, markdownSchema]]}
                    components={{
                        a: ({ href, children, ...props }) => (
                            <a href={href} target="_blank" rel="noreferrer" {...props}>
                                {children}
                            </a>
                        )
                    }}
                >
                    {msg.content}
                </ReactMarkdown>
            );
        }
        return (
            <>
                {msg.content.split('\n').map((line: string, idx: number): JSX.Element => (
                    <p key={idx}>{line}</p>
                ))}
            </>
        );
    };

    const renderChunkModal = (): JSX.Element | null => {
        if (!activeChunks) return null;
        return (
            <div
                className={styles.chunkOverlay}
                role="dialog"
                aria-modal="true"
                onClick={(): void => setActiveChunks(null)}
            >
                <div
                    className={styles.chunkModal}
                    onClick={(e: React.MouseEvent<HTMLDivElement>): void => e.stopPropagation()}
                >
                    <div className={styles.chunkHeader}>
                        <h3>Retrieved Chunks</h3>
                        <button
                            type="button"
                            className={styles.chunkClose}
                            onClick={(): void => setActiveChunks(null)}
                        >
                            Close
                        </button>
                    </div>
                    <div className={styles.chunkList}>
                        {activeChunks.map((chunk, idx): JSX.Element => {
                            const score = getDisplayScore(chunk.metadata);
                            return (
                            <div key={`${chunk.id}-${idx}`} className={styles.chunkItem}>
                                <div className={styles.chunkMeta}>
                                    <span className={styles.chunkId}>Chunk {chunk.id}</span>
                                    {score && (
                                        <span className={styles.chunkScore}>
                                            {score.label}: {score.value.toFixed(4)}
                                        </span>
                                    )}
                                    {chunk.metadata?.law_id && (
                                        <span className={styles.chunkTag}>
                                            {String(chunk.metadata.law_id)}
                                        </span>
                                    )}
                                </div>
                                <div className={styles.chunkText}>
                                    {chunk.text}
                                </div>
                            </div>
                        )})}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className={clsx(styles.container)}>
            <div
                ref={messagesContainerRef}
                id="messages"
                className={styles.messageList}
            >
                {messages.length === 0 && (
                    <div className={styles.emptyState}>
                        <Scale className={styles.emptyIcon} size={48} />
                        <h2>LawQA RAG Studio</h2>
                        <p>Ask legal questions with precision.</p>
                    </div>
                )}

                <AnimatePresence initial={false}>
                    {messages.map((msg: ChatMessage, i: number): JSX.Element => (
                        <motion.div
                            key={i}
                            ref={i === messages.length - 1 && msg.role === 'user' ? setLastUserMessageRef : undefined}
                            initial={{ opacity: 0, y: 20, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            transition={{ duration: 0.4, ease: "easeOut" }}
                            className={clsx(
                                styles.messageBubble,
                                msg.role === 'user' ? styles.userBubble : styles.aiBubble
                            )}
                        >
                            <div className={styles.avatar}>
                                {msg.role === 'user' ? 'You' : 'AI'}
                            </div>
                            <div className={styles.content}>
                                {renderContent(msg)}
                                {msg.role === 'assistant' && msg.chunks && msg.chunks.length > 0 && (
                                    <div className={styles.chunkTriggerRow}>
                                        <button
                                            type="button"
                                            className={styles.chunkTrigger}
                                            onClick={(): void => setActiveChunks(msg.chunks ?? null)}
                                        >
                                            Chunks ({msg.chunks.length})
                                        </button>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {isLoading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className={styles.loadingIndicator}
                    >
                        <span className={styles.dot}></span>
                        <span className={styles.dot}></span>
                        <span className={styles.dot}></span>
                    </motion.div>
                )}

                {messages.length > 0 && (
                    <div className={styles.scrollSpacer} aria-hidden="true" ref={spacerRef} />
                )}
            </div>

            <div className={styles.inputArea}>
                <div className={styles.inputWrapper}>
                    <form onSubmit={handleSubmit} className={styles.form}>
                        <input
                            type="text"
                            value={inputObj}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>): void => setInputObj(e.target.value)}
                            placeholder="Ask your question..."
                            className={styles.input}
                            autoFocus
                        />
                        <button
                            type="submit"
                            disabled={!inputObj.trim() || isLoading}
                            className={styles.sendBtn}
                        >
                            <Send size={20} />
                        </button>
                    </form>
                </div>
            </div>
            {renderChunkModal()}
        </div>
    );
});

ChatInterface.displayName = 'ChatInterface';

export default ChatInterface;
