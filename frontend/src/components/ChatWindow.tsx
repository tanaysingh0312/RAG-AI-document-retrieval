import React, { useState, useRef, useEffect } from 'react';
// Importing necessary icons for the chat interface.
import { Send, Upload, Paperclip, Bot, User, ChevronDown, Info, XCircle } from 'lucide-react';
// Importing child components for displaying individual messages and loading indicators.
import ChatMessage from './ChatMessage'; // Omitted .tsx extension.
import LoadingDots from './LoadingDots'; // Omitted .tsx extension.

/**
 * ChatWindow Component
 * This component represents the main chat interface where users can interact with the AI assistant.
 * It handles message input, display of chat history, file uploads, and displays context information.
 *
 * @param {object} props - The properties passed to the component.
 * @param {object} props.user - The authenticated user object.
 * @param {string | null} props.currentSessionId - The ID of the currently active chat session.
 * @param {string} props.currentSessionTitle - The title of the currently active chat session.
 * @param {Array<object>} props.chatHistory - The array of message objects for the current session.
 * @param {Function} props.setChatHistory - Function to update the chat history in the parent component (App.jsx).
 * @param {Function} props.onNewSessionCreated - Callback for when a new session is created (first message sent).
 * @param {boolean} props.sidebarOpen - State of the sidebar, used for responsive adjustments.
 */
const ChatWindow = ({ user, currentSessionId, currentSessionTitle, chatHistory, setChatHistory, onNewSessionCreated, sidebarOpen }) => {
  // --- Local State Management ---
  // Internal state for messages displayed in this component. Synchronized with `chatHistory` prop.
  const [messages, setMessages] = useState(chatHistory);
  // State for the text input field's value.
  const [inputValue, setInputValue] = useState('');
  // State to indicate if an API call (sending message, uploading file, etc.) is in progress.
  const [isLoading, setIsLoading] = useState(false);
  // State to store and display any error messages to the user.
  const [error, setError] = useState('');
  // State for the currently selected AI model (e.g., 'mistral').
  const [selectedModel, setSelectedModel] = useState('mistral');
  // State to control the visibility of the AI model selection dropdown.
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  // State to control the visibility of the modal displaying context information.
  const [showContextInfo, setShowContextInfo] = useState(false);
  // State to store the actual context (retrieved chunks) used for the last AI response.
  const [lastContextUsed, setLastContextUsed] = useState('');
  // State to store the filenames of the source documents from which context was retrieved.
  const [lastSourceFilenames, setLastSourceFilenames] = useState([]);
  // State to store additional metadata retrieved alongside the context.
  const [lastRetrievedMetadata, setLastRetrievedMetadata] = useState({});

  // --- Refs for DOM Element Interaction ---
  // Ref to automatically scroll to the bottom of the chat messages area.
  const messagesEndRef = useRef(null);
  // Ref to the chat messages container, used for checking scroll position.
  const chatMessagesContainerRef = useRef(null);
  // Ref to the textarea input field, used for programmatic height adjustment.
  const inputRef = useRef(null);

  // Base URL for backend API calls, dynamically determined.
  const API_BASE_URL = `${window.location.origin}/api`;

  // --- AI Model Configuration ---
  // Array of available AI models with their IDs, names, and descriptions.
  const models = [
    { id: 'mistral', name: 'Mistral (Default)', description: 'FastAPI default model for general queries.' },
    // Add more models here if supported by the backend.
  ];

  // --- Effects for UI Synchronization and Behavior ---

  /**
   * useEffect hook to synchronize the internal `messages` state with the `chatHistory` prop.
   * This is crucial when switching between chat sessions in the sidebar, ensuring the
   * ChatWindow displays the correct history for the newly selected session.
   * It also adds an initial assistant greeting for new, unsaved chats.
   */
  useEffect(() => {
    setMessages(chatHistory); // Update internal messages state.
    // If chat history is empty and no session is currently selected, add an initial greeting.
    if (chatHistory.length === 0 && currentSessionId === null) {
      setMessages([
        {
          id: 'initial-assistant-greeting', // A unique ID for this introductory message.
          role: 'assistant', // Role is 'assistant'.
          content: 'Hello! I am your Pyrotech AI Document Assistant. I can help you analyze documents, answer questions, and provide insights. How can I assist you today?',
          timestamp: new Date().toISOString() // Timestamp for the message.
        }
      ]);
    }
  }, [chatHistory, currentSessionId]); // Dependencies: `chatHistory` (when parent updates) and `currentSessionId` (when session changes).

  /**
   * useEffect hook for auto-scrolling to the bottom of the chat messages.
   * It implements "smarter" scrolling: only scrolls if the user is already near the bottom
   * or if it's the very first messages being rendered (e.g., initial greeting, new session load).
   */
  useEffect(() => {
    const chatContainer = chatMessagesContainerRef.current;
    if (chatContainer) {
      // Calculate if the user's current scroll position is near the bottom.
      // A threshold of 100px is used.
      const isNearBottom = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < 100;

      // Determine when to auto-scroll:
      // 1. If there are only a few messages (initial load).
      // 2. If the user was already near the bottom when new messages arrived (e.g., AI response).
      // 3. If a new session's history is being loaded for the first time.
      if (messages.length <= 2 || isNearBottom || (messages.length > 0 && chatHistory.length === 0 && currentSessionId !== null)) {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); // Smooth scroll to the bottom.
      }
    }
  }, [messages, chatHistory, currentSessionId]); // Dependencies: `messages` (new messages), `chatHistory` (session switch), `currentSessionId`.

  /**
   * useEffect hook to automatically adjust the height of the textarea input field.
   * It makes the textarea grow up to a maximum height (128px) as the user types,
   * then adds a scrollbar if content exceeds the max height.
   */
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'; // Reset height to recalculate scrollHeight.
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 128)}px`; // Set height, capping at 128px.
    }
  }, [inputValue]); // Dependency: re-run when `inputValue` changes.

  // --- Event Handlers for User Interactions ---

  /**
   * Handles sending a message to the AI assistant.
   * - Performs optimistic UI update by adding the user's message immediately.
   * - Calls the backend API to get an AI response.
   * - Updates chat history with the AI's response and any associated context/metadata.
   * - Handles new session creation if it's the first message in a "New Chat".
   */
  const handleSendMessage = async () => {
    // Prevent sending if input is empty, loading is in progress, or user is not logged in.
    if (!inputValue.trim() || isLoading || !user) {
      if (!user) setError("Please log in to send messages.");
      return;
    }

    setError(''); // Clear any previous error messages.
    setIsLoading(true); // Set loading state to true.

    // Create the user message object.
    const userMessage = {
      id: Date.now(), // Simple unique ID for frontend display.
      role: 'user', // Role of the sender.
      content: inputValue, // Content from the input field.
      timestamp: new Date().toISOString() // Current timestamp.
    };

    // Optimistically update the UI with the user's message.
    const updatedMessagesForUI = [...messages, userMessage];
    setMessages(updatedMessagesForUI);
    setInputValue(''); // Clear the input field immediately.

    let sessionIDForBackend = currentSessionId; // Use current session ID or empty string for new session.
    let sessionTitleForBackend = currentSessionTitle; // Use current title or derive for new session.

    try {
      const token = localStorage.getItem("token"); // Retrieve the authentication token.
      if (!token) {
        throw new Error("Authentication token not found. Please log in.");
      }

      // Prepare chat history for the backend.
      // Backend expects 'role' and 'content' for chat history.
      const chatHistoryForAPI = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // If it's a new chat, derive a session title from the first message.
      if (sessionIDForBackend === null || sessionIDForBackend === '') {
        sessionTitleForBackend = inputValue.substring(0, 50) + (inputValue.length > 50 ? '...' : '');
      }

      // Construct the payload for the chat API request.
      const requestPayload = {
        user_query: userMessage.content,
        session_id: sessionIDForBackend || '', // Send empty string for backend to create a new session.
        chat_history_json: JSON.stringify(chatHistoryForAPI), // Stringify the history array.
        session_title: sessionTitleForBackend,
      };

      // Send the message to the backend chat endpoint.
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`, // Include the JWT for authentication.
        },
        body: JSON.stringify(requestPayload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to get AI response: HTTP status ${response.status}.`);
      }

      const responseData = await response.json(); // Parse the AI's response.
      const assistantMessageContent = responseData.response; // The AI's generated text response.
      const newBackendSessionId = responseData.session_id; // The session ID returned by the backend (new or existing).

      // Create the assistant message object.
      const assistantMessage = {
        id: Date.now() + 1, // Unique ID for the assistant message.
        role: 'assistant', // Role is 'assistant'.
        content: assistantMessageContent, // Content from the AI.
        timestamp: new Date().toISOString(), // Current timestamp.
      };

      // Update context information received from the backend for the info modal.
      setLastContextUsed(responseData.context_used || '');
      setLastSourceFilenames(responseData.source_filenames || []);
      setLastRetrievedMetadata(responseData.retrieved_metadata || {});

      // Update the internal messages state with the assistant's response.
      setMessages(prevMessages => [...prevMessages, assistantMessage]);
      // Update the parent component's chat history state.
      setChatHistory(prevMessages => [...prevMessages, userMessage, assistantMessage]);

      // If a new session was created on the backend, inform the parent component.
      if (!currentSessionId && newBackendSessionId) {
        onNewSessionCreated(newBackendSessionId, sessionTitleForBackend);
      }

    } catch (err) {
      console.error("Error sending message to AI:", err);
      setError(err.message || 'An unexpected error occurred while getting AI response.');
      // Revert optimistic update and display an error message from the assistant.
      setMessages(updatedMessagesForUI.slice(0, -1)); // Remove the optimistically added user message.
      setMessages(prevMessages => [...prevMessages, {
        id: Date.now() + 2,
        role: 'assistant',
        content: `Error: ${err.message}. Please try again. If the issue persists, contact support.`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false); // Reset loading state.
    }
  };

  /**
   * Handles keyboard key presses in the input textarea.
   * Specifically, it sends the message when 'Enter' is pressed (without 'Shift').
   * 'Shift + Enter' creates a new line.
   * @param {object} e - The keyboard event object.
   */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent default Enter key behavior (new line).
      handleSendMessage(); // Call the message sending function.
    }
  };

  /**
   * Handles file uploads.
   * Sends the selected file(s) to the backend's upload endpoint.
   * @param {object} event - The file input change event.
   */
  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files); // Get selected files as an array.
    if (files.length === 0) return; // Do nothing if no files are selected.

    // Prevent upload if user is not logged in.
    if (!user || !localStorage.getItem("token")) {
      setError("You must be logged in to upload files.");
      event.target.value = ''; // Clear the file input.
      return;
    }

    setIsLoading(true); // Set loading state.
    setError(''); // Clear previous errors.

    const formData = new FormData(); // Create a FormData object to send files.
    files.forEach(file => {
      formData.append('files', file); // Append each file.
    });

    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token not found. Please log in.");
      }

      const response = await fetch(`${API_BASE_URL}/uploadfile/`, {
        method: 'POST', // Use POST method for file upload.
        headers: {
          // 'Content-Type' is automatically set to 'multipart/form-data' by FormData.
          'Authorization': `Bearer ${token}`, // Authenticate the request.
        },
        body: formData, // The FormData object.
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `File upload failed: HTTP status ${response.status}.`);
      }

      const result = await response.json(); // Parse the upload response.
      // Create an assistant message to confirm the upload status.
      const uploadConfirmationMessage = {
        id: Date.now(),
        role: 'assistant',
        content: `File "${files[0].name}" uploaded successfully! Status: ${result.status}. Message: ${result.message}.`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, uploadConfirmationMessage]); // Add to local messages.
      setChatHistory(prev => [...prev, uploadConfirmationMessage]); // Update parent history.

      // If the upload was successful, trigger document ingestion on the backend.
      if (result.status === 'success') {
        // A short delay before triggering ingestion to ensure backend has processed the upload.
        setTimeout(() => {
          triggerDocumentIngestion();
        }, 1000); // 1 second delay
      }

    } catch (err) {
      console.error("File upload error:", err);
      setError(err.message || 'An unexpected error occurred during file upload.');
      // Display an error message in the chat.
      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        role: 'assistant',
        content: `File upload failed: ${err.message}.`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false); // Reset loading state.
      event.target.value = ''; // Clear the file input field to allow re-uploading the same file.
    }
  };

  /**
   * Triggers the document ingestion process on the backend.
   * This typically means processing newly uploaded files and adding them to the RAG knowledge base.
   */
  const triggerDocumentIngestion = async () => {
    setIsLoading(true); // Set loading state for ingestion.
    setError(''); // Clear previous errors.
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token not found. Please log in.");
      }

      const response = await fetch(`${API_BASE_URL}/ingest_processed_documents/`, {
        method: 'POST', // Use POST method to trigger the ingestion action.
        headers: {
          'Authorization': `Bearer ${token}`, // Authenticate the request.
          'Content-Type': 'application/json', // Specify content type.
        },
        body: JSON.stringify({}) // Send an empty JSON body if no specific payload is needed.
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Document ingestion failed: HTTP status ${response.status}.`);
      }
      const responseData = await response.json();
      // Display confirmation message in chat.
      const ingestionConfirmationMessage = {
        id: Date.now(),
        role: 'assistant',
        content: `Document ingestion triggered successfully! Message: ${responseData.message}.`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, ingestionConfirmationMessage]);
      setChatHistory(prev => [...prev, ingestionConfirmationMessage]);

    } catch (err) {
      console.error('Error during document ingestion:', err);
      setError(err.message || 'An unexpected error occurred during document ingestion.');
      // Display error message in chat.
      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        role: 'assistant',
        content: `Document ingestion failed: ${err.message}.`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false); // Reset loading state.
    }
  };

  // --- Render Logic ---
  return (
    // Main container for the chat window. Uses flexbox for column layout (header, messages, input).
    // `h-full` ensures it takes full height of its parent. `bg-background` for theme.
    <div className="flex flex-col h-full bg-background relative">
      {/* Chat Header Section */}
      <div className="flex-shrink-0 flex items-center justify-between p-4 border-b border-border bg-card shadow-sm">
        <div className="flex items-center space-x-3">
          {/* Assistant Avatar in Header */}
          <div className="w-8 h-8 bg-gradient-to-br from-pyrotech-500 to-pyrotech-600 rounded-full flex items-center justify-center">
            <Bot className="h-4 w-4 text-white" />
          </div>
          <div>
            {/* Current Session Title */}
            <h2 className="font-semibold text-foreground text-lg">{currentSessionTitle}</h2>
            <p className="text-xs text-muted-foreground">Ready to help with your documents</p>
          </div>
        </div>

        {/* Right-aligned Header Controls */}
        <div className="flex items-center space-x-2">
          {/* AI Model Selector Dropdown */}
          <div className="relative">
            <button
              onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
              className="flex items-center space-x-2 px-3 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors duration-200 text-sm font-medium text-foreground"
              disabled={isLoading || !user} // Disable if loading or not logged in.
              aria-expanded={isModelDropdownOpen}
              aria-haspopup="true"
            >
              <span>{models.find(m => m.id === selectedModel)?.name}</span>
              <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${isModelDropdownOpen ? 'rotate-180' : ''}`} />
            </button>

            {isModelDropdownOpen && (
              // Dropdown content for model selection.
              <div className="absolute right-0 top-full mt-2 w-64 bg-popover border border-border rounded-lg shadow-lg z-50">
                <div className="p-2">
                  {models.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        setSelectedModel(model.id);
                        setIsModelDropdownOpen(false); // Close dropdown after selection.
                      }}
                      className={`w-full text-left p-3 rounded-lg hover:bg-accent transition-colors duration-200 ${
                        selectedModel === model.id ? 'bg-accent text-accent-foreground' : 'text-popover-foreground'
                      }`}
                      role="menuitem"
                    >
                      <div className="font-medium">{model.name}</div>
                      <div className="text-xs text-muted-foreground mt-1">{model.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Context Info Button (visible only if context was used in the last response) */}
          {lastContextUsed && (
            <button
              onClick={() => setShowContextInfo(true)}
              className="p-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200 shadow-sm"
              title="View Context Used for Last Response"
              aria-label="View context information"
            >
              <Info className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Chat Messages Area */}
      {/* `flex-1` makes this area take all available vertical space. `overflow-y-auto` enables scrolling. */}
      {/* `min-h-0` is crucial for flex items with `overflow` to behave correctly. */}
      {/* `paddingBottom` is added to ensure content is not hidden by the fixed input area. */}
      <div ref={chatMessagesContainerRef} className="flex-1 overflow-y-auto custom-scrollbar bg-background min-h-0 px-4 py-2" style={{ paddingBottom: '100px' }}>
        <div className="flex flex-col space-y-6 max-w-4xl mx-auto py-4"> {/* Added vertical padding */}
          {messages.map((message) => (
            // Render each message using the ChatMessage component.
            <ChatMessage key={message.id} message={message} />
          ))}
          
          {isLoading && (
            // Display loading dots when an AI response is pending.
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-pyrotech-500 to-pyrotech-600 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="h-4 w-4 text-white" />
              </div>
              <div className="bg-card border border-border rounded-lg p-4 max-w-3xl shadow-sm">
                <LoadingDots />
              </div>
            </div>
          )}
          
          {/* A ref for auto-scrolling to the end of messages. */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Chat Input Area (Fixed at the bottom) */}
      {/* `fixed bottom-0 left-0 right-0` positions it at the bottom of the viewport. */}
      {/* `z-10` ensures it stays above other content. */}
      <div className="fixed bottom-0 left-0 right-0 z-10 bg-card border-t border-border p-4 flex justify-center shadow-lg">
        <div className="max-w-4xl w-full"> {/* Constrains width and centers the input elements. */}
          {error && (
            // Display error message above the input field if an error occurs.
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded-lg relative mb-4 text-sm font-medium" role="alert">
              <XCircle className="inline-block h-4 w-4 mr-2" />
              {error}
            </div>
          )}
          <div className="flex items-end space-x-3">
            {/* File Upload Button */}
            <label htmlFor="file-upload"
              className="flex-shrink-0 p-3 rounded-lg bg-muted hover:bg-muted/80 transition-colors duration-200 cursor-pointer shadow-sm"
              aria-label="Upload document"
              title="Upload Documents (PDF, DOC, TXT, Images)"
            >
              <Upload className="h-5 w-5 text-muted-foreground" />
              {/* Hidden file input element */}
              <input id="file-upload" type="file" className="hidden" onChange={handleFileUpload} disabled={isLoading || !user} multiple />
            </label>

            {/* Message Input Textarea */}
            <div className="flex-1 relative">
              <textarea
                ref={inputRef} // Ref for height adjustment.
                value={inputValue} // Controlled component value.
                onChange={(e) => setInputValue(e.target.value)} // Update state on change.
                onKeyDown={handleKeyDown} // Handle Enter key for sending.
                placeholder={user ? "Ask me anything about your documents..." : "Please log in to chat."}
                className="w-full p-4 pr-12 bg-background border border-input rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200 min-h-[56px] max-h-32 text-foreground placeholder-muted-foreground shadow-sm"
                rows={1} // Initial rows.
                style={{
                  height: 'auto', // Auto height calculation.
                  minHeight: '56px' // Minimum height.
                }}
                onInput={(e) => { // Redundant with useEffect, but kept for direct DOM manipulation fallback.
                  const target = e.target;
                  target.style.height = 'auto';
                  target.style.height = `${Math.min(target.scrollHeight, 128)}px`;
                }}
                disabled={isLoading || !user} // Disable if loading or not logged in.
              />
              
              {/* Send Message Button */}
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading || !user} // Disable if input is empty, loading, or not logged in.
                className="absolute right-3 bottom-3 p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-md"
                aria-label="Send message"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
          
          {/* Input Area Footer (Hints) */}
          <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
            <div className="flex items-center space-x-4">
              <span>Press ⏎ to send, Shift + ⏎ for new line</span>
            </div>
            <div className="flex items-center space-x-2">
              <Paperclip className="h-3 w-3" />
              <span>Supports PDF, DOC, TXT, and Image files</span>
            </div>
          </div>
        </div>
      </div>

      {/* Context Information Modal */}
      {showContextInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-card text-foreground rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col border border-border">
            {/* Modal Header */}
            <div className="flex justify-between items-center p-5 border-b border-border bg-muted/20">
              <h3 className="text-xl font-semibold">Context Used for Last Response</h3>
              <button
                onClick={() => setShowContextInfo(false)}
                className="text-muted-foreground hover:text-foreground p-1 rounded-full hover:bg-accent transition-colors duration-200"
                aria-label="Close context info"
              >
                <XCircle className="h-6 w-6" />
              </button>
            </div>
            {/* Modal Content Body */}
            <div className="p-5 flex-1 overflow-y-auto custom-scrollbar">
              <h4 className="font-semibold text-lg mb-3 text-primary">Retrieved Context Snippets:</h4>
              <pre className="whitespace-pre-wrap text-sm bg-muted p-4 rounded-lg mb-5 border border-input overflow-x-auto custom-scrollbar leading-relaxed">
                {lastContextUsed || 'No relevant context was used for the last response.'}
              </pre>

              <h4 className="font-semibold text-lg mb-3 text-primary">Source Filenames:</h4>
              {lastSourceFilenames.length > 0 ? (
                <ul className="list-disc list-inside mb-5 space-y-1">
                  {lastSourceFilenames.map((filename, idx) => (
                    <li key={idx} className="text-sm text-muted-foreground flex items-center">
                      <FileText className="h-4 w-4 mr-2 flex-shrink-0 text-blue-500" />
                      <span>{filename}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground mb-5">No specific source filenames were identified from the retrieved context.</p>
              )}

              <h4 className="font-semibold text-lg mb-3 text-primary">Additional Retrieved Metadata:</h4>
              {Object.keys(lastRetrievedMetadata).length > 0 ? (
                <div className="bg-muted p-4 rounded-lg border border-input overflow-x-auto custom-scrollbar">
                  <pre className="whitespace-pre-wrap text-sm leading-relaxed">
                    {JSON.stringify(lastRetrievedMetadata, null, 2)}
                  </pre>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No additional structured metadata was retrieved.</p>
              )}
            </div>
            {/* Modal Footer */}
            <div className="p-4 border-t border-border flex justify-end bg-muted/20">
              <button
                onClick={() => setShowContextInfo(false)}
                className="bg-primary text-primary-foreground px-5 py-2 rounded-lg hover:bg-primary/90 transition-colors duration-200 font-medium shadow-md"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWindow;
