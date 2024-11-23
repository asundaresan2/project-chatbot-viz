export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'python';
  content: string;
  timestamp: number;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
}