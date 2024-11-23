import React, { useState, useRef } from 'react';

interface ChatInputProps {
  onSend: (message: string, attachment?: File) => void;
  disabled?: boolean;
  allowedFileTypes: string[];
  maxFileSize: number;
}

export function ChatInput({ onSend, disabled, allowedFileTypes, maxFileSize }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [attachment, setAttachment] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSend(message, attachment || undefined);
      setMessage('');
      setAttachment(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedFileTypes.includes(fileExtension)) {
      alert('Invalid file type. Allowed types: ' + allowedFileTypes.join(', '));
      return;
    }

    if (file.size > maxFileSize) {
      alert('File is too large. Maximum size is 2MB');
      return;
    }

    setAttachment(file);
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2 py-4">
      <div className="relative flex-1">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          disabled={disabled}
          className="w-full rounded-lg border border-gray-300 px-4 py-2 pr-12"
          placeholder="Type your message..."
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
        >
          📎
        </button>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept={allowedFileTypes.join(',')}
          onChange={handleFileChange}
        />
      </div>
      {attachment && (
        <div className="text-sm text-gray-600">
          {attachment.name}
          <button
            type="button"
            onClick={() => setAttachment(null)}
            className="ml-2 text-red-500"
          >
            ×
          </button>
        </div>
      )}
      <button
        type="submit"
        disabled={disabled || !message.trim()}
        className="rounded-lg bg-blue-500 px-4 py-2 text-white hover:bg-blue-600 disabled:opacity-50"
      >
        Send
      </button>
    </form>
  );
}