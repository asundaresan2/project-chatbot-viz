import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User, Terminal } from 'lucide-react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const icons = {
    user: <User className="w-6 h-6" />,
    assistant: <Bot className="w-6 h-6" />,
    python: <Terminal className="w-6 h-6" />,
    system: <Bot className="w-6 h-6" />
  };

  const bgColors = {
    user: 'bg-blue-50',
    assistant: 'bg-gray-50',
    python: 'bg-green-50',
    system: 'bg-yellow-50'
  };

  return (
    <div className={`p-4 ${bgColors[message.role]} rounded-lg mb-4`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-1">
          {icons[message.role]}
        </div>
        <div className="flex-1 overflow-hidden">
          <div className="prose max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}