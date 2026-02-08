import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import ReactMarkdown from 'react-markdown';
import { Send, Trash2, Loader2, Bot, User, Sparkles, Mic, MicOff, Volume2 } from 'lucide-react';
import { chatApi, transcribeApi, userApi, logApi } from '../services/api';
import { useAuthStore } from '../hooks/useAuth';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export default function Chat() {
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [proactiveAlert, setProactiveAlert] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const queryClient = useQueryClient();
  const { token } = useAuthStore();

  // Fetch user profile for personalized greeting
  const { data: profile } = useQuery({
    queryKey: ['userProfile'],
    queryFn: async () => {
      const res = await userApi.getProfile();
      return res.data as { name?: string; preferred_name?: string };
    },
  });

  // WebSocket: listen for proactive alerts (self-healing, errors)
  useEffect(() => {
    if (!token) return;
    const wsUrl = `${(window.location.protocol === 'https:' ? 'wss:' : 'ws:')}//${window.location.host}/ws/${token}`;
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'proactive_alert' && data.message) {
          setProactiveAlert(data.message);
          queryClient.invalidateQueries({ queryKey: ['chatHistory'] });
        }
      } catch {
        // ignore
      }
    };
    return () => ws.close();
  }, [token, queryClient]);

  const startRecording = useCallback(async () => {
    try {
      logApi.event('mic_start', {}).catch(() => {});
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeTypes = ['audio/webm', 'audio/mp4', 'audio/ogg', 'audio/wav', ''];
      const mimeType = mimeTypes.find((t) => !t || MediaRecorder.isTypeSupported(t)) || '';
      const ext = mimeType.includes('mp4') ? 'mp4'
        : mimeType.includes('ogg') ? 'ogg'
        : mimeType.includes('wav') ? 'wav'
        : 'webm';

      const options = mimeType ? { mimeType } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        logApi.event('mic_stop', { mimeType, chunkCount: chunksRef.current.length }).catch(() => {});
        stream.getTracks().forEach(track => track.stop());
        const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' });
        setIsTranscribing(true);
        setMicError(null);
        logApi.event('transcription_start', { format: ext, blobSize: blob.size }).catch(() => {});
        try {
          const reader = new FileReader();
          reader.onloadend = async () => {
            const base64 = (reader.result as string).split(',')[1];
            try {
              const res = await transcribeApi.transcribe(base64, ext);
              if (res.data.success && res.data.text) {
                logApi.event('transcription_success', { textLen: res.data.text?.length }).catch(() => {});
                setInput((prev) => prev + (prev ? ' ' : '') + res.data.text);
                inputRef.current?.focus();
              } else {
                logApi.event('transcription_failed', { error: res.data.error }).catch(() => {});
                setMicError(res.data.error || 'Transcription failed');
              }
            } catch (err: any) {
              const errMsg = err?.response?.data?.detail || 'Transcription service unavailable';
              logApi.event('transcription_failed', { error: errMsg, network: true }).catch(() => {});
              setMicError(errMsg);
            }
            setIsTranscribing(false);
          };
          reader.readAsDataURL(blob);
        } catch (e) {
          logApi.event('transcription_failed', { error: 'Failed to process audio', detail: String(e) }).catch(() => {});
          setIsTranscribing(false);
          setMicError('Failed to process audio');
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (e) {
      logApi.event('mic_denied', { error: 'Microphone access denied' }).catch(() => {});
      setMicError('Microphone access denied. Check browser permissions.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  // Fetch chat history (no staleTime so we get fresh data after restart)
  const { data: history, isLoading } = useQuery({
    queryKey: ['chatHistory'],
    queryFn: async () => {
      const response = await chatApi.getHistory();
      return response.data.messages as Message[];
    },
    staleTime: 0,
  });

  // Send message mutation
  const sendMutation = useMutation({
    mutationFn: async (content: string) => {
      const response = await chatApi.sendMessage(content);
      return response.data;
    },
    onMutate: async (content) => {
      await queryClient.cancelQueries({ queryKey: ['chatHistory'] });
      const previousMessages = queryClient.getQueryData<Message[]>(['chatHistory']);

      queryClient.setQueryData<Message[]>(['chatHistory'], (old = []) => [
        ...old,
        { role: 'user', content, timestamp: new Date().toISOString() },
      ]);

      return { previousMessages };
    },
    onSuccess: (data) => {
      // Immediately append the assistant response from the API result
      if (data?.response) {
        queryClient.setQueryData<Message[]>(['chatHistory'], (old = []) => [
          ...old,
          { role: 'assistant', content: data.response, timestamp: data.timestamp || new Date().toISOString() },
        ]);
      } else {
        // Fallback: refetch from server
        queryClient.invalidateQueries({ queryKey: ['chatHistory'] });
      }
    },
    onError: (_, __, context) => {
      queryClient.setQueryData(['chatHistory'], context?.previousMessages);
    },
  });

  // Clear history mutation
  const clearMutation = useMutation({
    mutationFn: chatApi.clearHistory,
    onSuccess: () => {
      queryClient.setQueryData(['chatHistory'], []);
    },
  });

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  const handleSend = () => {
    if (!input.trim() || sendMutation.isPending) return;
    sendMutation.mutate(input.trim());
    setInput('');
    inputRef.current?.focus();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 border-b flex items-center justify-between backdrop-blur-sm border-theme" style={{ backgroundColor: 'color-mix(in srgb, var(--bg-primary) 80%, transparent)' }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-md shadow-blue-500/15">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight text-theme-primary">Chat with Aria</h1>
            <p className="text-[11px] text-theme-secondary">AI Assistant</p>
          </div>
        </div>
        <button
          onClick={() => clearMutation.mutate()}
          disabled={clearMutation.isPending}
          className="btn-ghost flex items-center gap-2 text-sm"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Clear</span>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-6 h-6 animate-spin text-theme-secondary" />
          </div>
        ) : (
          <div className="space-y-5 max-w-3xl mx-auto">
            {/* Proactive alert (self-healing, errors) */}
            {proactiveAlert && (
              <div className="flex gap-3 animate-fade-in justify-start">
                <div className="w-8 h-8 rounded-lg bg-amber-500/20 border border-amber-500/30 flex items-center justify-center flex-shrink-0 mt-1">
                  <Sparkles className="w-4 h-4 text-amber-400" />
                </div>
                <div className="max-w-[75%] px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/20">
                  <p className="text-amber-200/90 text-sm">{proactiveAlert}</p>
                  <button onClick={() => setProactiveAlert(null)} className="mt-2 text-xs text-amber-400/70 hover:text-amber-400">Dismiss</button>
                </div>
              </div>
            )}
            {/* Aria intro: only show "What should I call you?" when no name; else "Hi [Name], welcome back!" */}
            {history?.length === 0 && (
              <div className="flex gap-3 animate-fade-in justify-start">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-1 shadow-md shadow-blue-500/15">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="max-w-[75%] px-4 py-3 message-assistant">
                  <div className="markdown-content">
                    <p>
                      {profile?.name || profile?.preferred_name ? (
                        <>Hi <strong>{profile.name || profile.preferred_name}</strong>, welcome back! What can I help with?</>
                      ) : (
                        <>Hey — I'm <strong>Aria</strong>. What should I call you, and what can I help with?</>
                      )}
                    </p>
                  </div>
                </div>
              </div>
            )}
            {history?.map((message, index) => (
              <div
                key={index}
                className={`flex gap-3 animate-fade-in ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-1 shadow-md shadow-blue-500/15">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                )}

                <div
                  className={`max-w-[75%] px-4 py-3 ${
                    message.role === 'user'
                      ? 'message-user'
                      : 'message-assistant'
                  }`}
                >
                  {message.role === 'assistant' ? (
                    <div className="markdown-content">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="whitespace-pre-wrap text-[14px] leading-relaxed">{message.content}</p>
                  )}
                </div>

                {message.role === 'user' && (
                  <div className="w-8 h-8 rounded-lg card-theme flex items-center justify-center flex-shrink-0 mt-1 border-theme">
                    <User className="w-4 h-4 text-theme-secondary" />
                  </div>
                )}
              </div>
            ))}

            {sendMutation.isPending && (
              <div className="flex gap-3 justify-start animate-fade-in">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-1 shadow-md shadow-blue-500/15">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="message-assistant px-4 py-3">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-theme backdrop-blur-sm" style={{ backgroundColor: 'color-mix(in srgb, var(--bg-primary) 80%, transparent)' }}>
        {micError && (
          <div className="max-w-3xl mx-auto mb-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center justify-between">
            <span>{micError}</span>
            <button onClick={() => setMicError(null)} className="ml-2 text-red-400/60 hover:text-red-400">&times;</button>
          </div>
        )}
        <div className="max-w-3xl mx-auto flex gap-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={isRecording ? 'Listening...' : isTranscribing ? 'Transcribing...' : 'Type a message or /help for commands...'}
            rows={1}
            className="flex-1 px-4 py-3 glass-input rounded-xl resize-none text-[14px]"
            style={{ maxHeight: '150px' }}
          />
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isTranscribing}
            className={`px-4 py-3 rounded-xl transition-all ${
              isRecording
                ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
                : isTranscribing
                  ? 'bg-theme-muted text-theme-secondary'
                  : 'btn-theme-secondary'
            }`}
            title={isRecording ? 'Stop recording' : 'Voice input'}
          >
            {isTranscribing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : isRecording ? (
              <MicOff className="w-5 h-5" />
            ) : (
              <Mic className="w-5 h-5" />
            )}
          </button>
          <button
            onClick={handleSend}
            disabled={!input.trim() || sendMutation.isPending}
            className="px-4 py-3 btn-primary rounded-xl"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
