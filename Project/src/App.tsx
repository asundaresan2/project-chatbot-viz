import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import type { Message, ChatState } from './types';

function App() {
  const [state, setState] = useState<ChatState>({
    messages: [
      {
        id: '1',
        role: 'system',
        content: 'Welcome! I can help you Run Statistical Analytics, query IFC Data sets and Create Interactive Visualizations.',
        timestamp: Date.now()
      }
    ],
    isLoading: false
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [state.messages]);

  const handleSendMessage = async (content: string, attachment?: File) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: Date.now()
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isLoading: true
    }));

    try {
      const formData = new FormData();
      formData.append('message', content);
      formData.append('history', JSON.stringify(state.messages));
      if (attachment) {
        formData.append('file', attachment);
      }

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      setState(prev => ({
        ...prev,
        messages: [...prev.messages, ...data.messages],
        isLoading: false
      }));
    } catch (error) {
      console.error('Error:', error);
      setState(prev => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            id: Date.now().toString(),
            role: 'system',
            content: 'Sorry, there was an error processing your request.',
            timestamp: Date.now()
          }
        ],
        isLoading: false
      }));
    }
  };

  return (
    <div className="flex h-screen flex-col bg-gray-100">
      <header className="bg-white border-b px-4 py-3">
        <h1 className="text-xl font-semibold text-gray-800">IFC Data Wizzard</h1>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl">
          {state.messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="mx-auto w-full max-w-3xl px-4">
        <ChatInput 
          onSend={handleSendMessage} 
          disabled={state.isLoading}
          allowedFileTypes={['.pdf', '.xlsx', '.xls', '.png', '.jpg', '.jpeg']}
          maxFileSize={2 * 1024 * 1024}
        />
      </footer>
    </div>
  );
}

export default App;