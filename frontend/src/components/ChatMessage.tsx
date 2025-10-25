import React from 'react';
// Importing icons for bot, user, copy, like, and dislike actions.
import { Bot, User, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';

/**
 * ChatMessage Component
 * Displays a single message within the chat window, distinguishing between
 * user messages and assistant messages with different styling and avatars.
 * It also provides actions like copying assistant messages.
 *
 * @param {object} props - The properties passed to the component.
 * @param {object} props.message - The message object to display.
 * Expected structure: { id: string, role: 'user' | 'assistant', content: string, timestamp: string }
 */
const ChatMessage = ({ message }) => {
  // Determine if the message was sent by the assistant.
  // Checks both 'type' (legacy/frontend-specific) and 'role' (backend-driven) for robustness.
  const isAssistant = message.type === 'assistant' || message.role === 'assistant';
  
  /**
   * Handles copying the message content to the clipboard.
   * Uses `document.execCommand('copy')` for broader compatibility within iframe environments,
   * as `navigator.clipboard.writeText()` might have restrictions.
   */
  const handleCopy = () => {
    // Create a temporary textarea element to hold the text to be copied.
    const tempTextArea = document.createElement('textarea');
    tempTextArea.value = message.content; // Set its value to the message content.
    document.body.appendChild(tempTextArea); // Append it to the document body (it doesn't need to be visible).
    tempTextArea.select(); // Select the text within the textarea.
    document.execCommand('copy'); // Execute the copy command.
    document.body.removeChild(tempTextArea); // Remove the temporary textarea.
    console.log('Message content copied to clipboard!'); // Log for debugging/feedback.
    // In a production app, you might show a small "Copied!" tooltip here.
  };

  /**
   * Formats a given ISO timestamp string into a user-friendly time string.
   * Provides fallbacks for invalid timestamps and different time granularities.
   * @param {string} timestamp - The ISO 8601 formatted timestamp string.
   * @returns {string} The formatted time string (e.g., "10:30 AM", "Yesterday 2:15 PM").
   */
  const formatTime = (timestamp) => {
    try {
      const date = new Date(timestamp); // Create a Date object from the timestamp.
      // Validate if the Date object is a valid date.
      if (isNaN(date.getTime())) {
        console.warn("Invalid timestamp received for formatting:", timestamp);
        return "Invalid Date"; // Return a fallback string for invalid dates.
      }

      const now = new Date(); // Get the current date and time.
      const diffMilliseconds = Math.abs(now.getTime() - date.getTime()); // Difference in milliseconds.
      const diffMinutes = Math.round(diffMilliseconds / (1000 * 60)); // Difference in minutes.
      const diffHours = Math.round(diffMilliseconds / (1000 * 60 * 60)); // Difference in hours.
      const diffDays = Math.round(diffMilliseconds / (1000 * 60 * 60 * 24)); // Difference in days.

      // Conditional formatting based on how recent the message is.
      if (diffMinutes < 1) {
        return 'Just now';
      } else if (diffMinutes < 60) {
        return `${diffMinutes} min ago`;
      } else if (diffHours < 24) {
        return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
      } else if (diffDays < 7) {
        // For messages within the last week, show weekday and time.
        return date.toLocaleDateString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' });
      } else {
        // For older messages, show a short date format.
        return date.toLocaleDateString([], { month: 'numeric', day: 'numeric', year: '2-digit' });
      }
    } catch (e) {
      console.error("Error formatting timestamp:", e, "Original timestamp:", timestamp);
      return "Error"; // Fallback for any parsing or formatting errors.
    }
  };

  // --- Render Logic ---
  return (
    // Main container for a single chat message.
    // Uses flexbox to align avatar and message bubble.
    // `animate-fade-in` for a subtle entrance animation.
    // `flex-row-reverse space-x-reverse` for user messages to align to the right.
    <div className={`flex items-start space-x-3 animate-fade-in ${!isAssistant ? 'flex-row-reverse space-x-reverse' : ''}`}>
      {/* Avatar Section */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center 
        ${isAssistant ? 'bg-pyrotech-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}>
        {isAssistant ? (
          // Assistant's avatar icon.
          <Bot className="h-5 w-5" />
        ) : (
          // User's avatar icon.
          <User className="h-4 w-4" />
        )}
      </div>

      {/* Message Bubble Section */}
      {/* `group` class enables hover effects on child elements. */}
      {/* `max-w-xs`, `md:max-w-md`, `lg:max-w-lg` for responsive width. */}
      <div className={`flex-1 p-3 rounded-lg max-w-xs md:max-w-md lg:max-w-lg relative group 
        ${isAssistant 
          ? 'bg-pyrotech-100 dark:bg-pyrotech-900 text-pyrotech-800 dark:text-pyrotech-100' // Assistant message styling.
          : 'bg-primary text-primary-foreground' // User message styling.
        } shadow-sm`}>
        {/* Message Content */}
        {/* `whitespace-pre-wrap` preserves whitespace and wraps text. */}
        <p className={`text-sm whitespace-pre-wrap ${
            isAssistant ? 'text-foreground' : 'text-primary-foreground'
          }`}>
          {message.content}
        </p>

        {/* Message Actions (Copy, Like, Dislike) */}
        {/* These actions are initially hidden and appear on hover (`group-hover:opacity-100`). */}
        <div className={`flex items-center space-x-2 mt-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 ${
          !isAssistant ? 'justify-end' : '' // Align actions to the right for user messages.
        }`}>
          {/* Timestamp Display */}
          <span className={`text-xs text-muted-foreground ${!isAssistant ? 'order-first' : ''}`}>
            {formatTime(message.timestamp)}
          </span>
          
          {/* Assistant-specific actions */}
          {isAssistant && (
            <div className="flex items-center space-x-1">
              {/* Copy Button */}
              <button
                onClick={handleCopy}
                className="p-1 rounded hover:bg-accent transition-colors duration-200"
                aria-label="Copy message to clipboard"
                title="Copy message"
              >
                <Copy className="h-3 w-3 text-muted-foreground" />
              </button>
              {/* Like Button (Placeholder for feedback) */}
              <button
                className="p-1 rounded hover:bg-accent transition-colors duration-200"
                aria-label="Like message"
                title="Like message"
              >
                <ThumbsUp className="h-3 w-3 text-muted-foreground" />
              </button>
              {/* Dislike Button (Placeholder for feedback) */}
              <button
                className="p-1 rounded hover:bg-accent transition-colors duration-200"
                aria-label="Dislike message"
                title="Dislike message"
              >
                <ThumbsDown className="h-3 w-3 text-muted-foreground" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
