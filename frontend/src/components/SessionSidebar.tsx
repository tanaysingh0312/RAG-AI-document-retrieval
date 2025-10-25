import React, { useState, useEffect, useCallback } from 'react';
// Importing icons for various sidebar functionalities.
import { Plus, MessageSquare, Clock, Search, Trash2, MoreHorizontal, ChevronDown, CheckCircle, XCircle } from 'lucide-react';

// Constant defining how many sessions to load per "Show More" click.
const SESSIONS_PER_PAGE = 4;

/**
 * SessionSidebar Component
 * This component displays a list of chat sessions, allowing users to:
 * - Start a new chat.
 * - Search through existing conversations.
 * - Select a conversation to view its history.
 * - Delete conversations (with confirmation).
 * - Load more conversations if available.
 *
 * @param {object} props - The properties passed to the component.
 * @param {boolean} props.isOpen - Controls the visibility of the sidebar.
 * @param {string} props.currentView - The current main application view (e.g., 'chat').
 * @param {Function} props.onViewChange - Callback to change the main application view.
 * @param {Array<object>} props.sessions - An array of chat session objects.
 * @param {Function} props.onSelectSession - Callback to select a chat session.
 * @param {Function} props.onDeleteSession - Callback to delete a chat session.
 * @param {string | null} props.currentSessionId - The ID of the currently active session.
 * @param {object} props.user - The authenticated user object.
 */
const SessionSidebar = ({ isOpen, currentView, onViewChange, sessions, onSelectSession, onDeleteSession, currentSessionId, user }) => {
  // State for the search term entered by the user.
  const [searchTerm, setSearchTerm] = useState('');
  // State to manage which session is currently in the delete confirmation state.
  // Stores the sessionId if a delete confirmation is active, otherwise null.
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  // State to control how many sessions are currently visible in the list.
  const [visibleSessionsCount, setVisibleSessionsCount] = useState(SESSIONS_PER_PAGE);

  // Ensure `sessions` prop is always treated as an array to prevent errors.
  const safeSessions = Array.isArray(sessions) ? sessions : [];

  /**
   * useEffect hook to reset the `visibleSessionsCount` whenever the `safeSessions` data changes.
   * This ensures that when new sessions are fetched or existing ones are updated/deleted,
   * the list correctly resets to the initial number of visible sessions.
   */
  useEffect(() => {
    setVisibleSessionsCount(SESSIONS_PER_PAGE);
  }, [safeSessions]); // Dependency array: re-run when `safeSessions` array reference changes.

  // Filter sessions based on the `searchTerm`.
  // Converts both the session title and search term to lowercase for case-insensitive matching.
  const filteredSessions = safeSessions.filter(session =>
    session.session_title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Determine which sessions to display based on the `visibleSessionsCount`.
  const sessionsToDisplay = filteredSessions.slice(0, visibleSessionsCount);

  // Check if there are more sessions available to load beyond the currently visible count.
  const hasMoreSessions = filteredSessions.length > visibleSessionsCount;

  // --- Event Handlers ---

  /**
   * Handles the click event for the "Show More" button.
   * Increases the `visibleSessionsCount` to display more sessions.
   */
  const handleShowMore = () => {
    setVisibleSessionsCount(prevCount => prevCount + SESSIONS_PER_PAGE);
  };

  /**
   * Initiates the delete confirmation process for a specific session.
   * Sets the `deleteConfirm` state to the ID of the session to be confirmed for deletion.
   * @param {string} sessionId - The ID of the session to mark for deletion confirmation.
   */
  const handleDeleteClick = (sessionId) => {
    setDeleteConfirm(sessionId);
  };

  /**
   * Confirms the deletion of a session.
   * Calls the `onDeleteSession` callback (passed from parent `App.jsx`) and then
   * resets the `deleteConfirm` state to hide the confirmation UI.
   * @param {string} sessionId - The ID of the session to be deleted.
   * @param {object} e - The event object to stop propagation.
   */
  const handleConfirmDelete = (sessionId, e) => {
    e.stopPropagation(); // Prevent the click from bubbling up and selecting the session.
    if (user) { // Ensure a user is logged in before attempting to delete.
      onDeleteSession(sessionId); // Call the parent's delete function.
      setDeleteConfirm(null); // Hide the confirmation UI.
    } else {
      console.warn("Attempted to delete session without a logged-in user.");
      setDeleteConfirm(null); // Hide confirmation, as action cannot proceed.
      // Optionally, display an error message to the user.
    }
  };

  /**
   * Cancels the delete confirmation process.
   * Resets the `deleteConfirm` state to `null` to hide the confirmation UI.
   * @param {object} e - The event object to stop propagation.
   */
  const handleCancelDelete = (e) => {
    e.stopPropagation(); // Prevent the click from bubbling up.
    setDeleteConfirm(null); // Hide the confirmation UI.
  };

  /**
   * Formats an ISO timestamp string into a human-readable relative time.
   * Examples: "Just now", "5 min ago", "2 hours ago", "Mon 10:30 AM", "12/25/23".
   * @param {string} isoString - The ISO 8601 formatted timestamp string.
   * @returns {string} The formatted time string.
   */
  const formatTimestamp = (isoString) => {
    try {
      const date = new Date(isoString); // Create a Date object.
      // Check for invalid date.
      if (isNaN(date.getTime())) {
        console.warn("Invalid timestamp received for formatting:", isoString);
        return "Invalid Date";
      }

      const now = new Date(); // Current date and time.
      const diffMilliseconds = now.getTime() - date.getTime(); // Difference in milliseconds.
      const diffMinutes = Math.round(diffMilliseconds / (1000 * 60));
      const diffHours = Math.round(diffMilliseconds / (1000 * 60 * 60));
      const diffDays = Math.round(diffMilliseconds / (1000 * 60 * 60 * 24));

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
      console.error("Error formatting timestamp:", e, "Original timestamp:", isoString);
      return "Error";
    }
  };

  // --- Conditional Rendering for Sidebar Visibility ---
  // If `isOpen` is false, the sidebar is not rendered (or rendered as `w-0`).
  // This helps in achieving the collapse/expand animation.
  if (!isOpen) {
    return null; // Or render a collapsed version if desired.
  }

  // --- Main Sidebar Render ---
  return (
    // Main sidebar container.
    // `h-full` makes it take full height. `flex-col` for vertical layout.
    // `min-w-[280px] max-w-[320px]` defines responsive width. `flex-shrink-0` prevents shrinking.
    <div className="h-full flex flex-col bg-card border-r border-border min-w-[280px] max-w-[320px] flex-shrink-0 rounded-lg shadow-lg">
      {/* Custom Scrollbar Styling (JSX style tag for direct CSS injection) */}
      {/* This ensures the custom scrollbar appears within the immersive environment. */}
      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px; /* Width of the scrollbar */
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: var(--color-background); /* Light gray track */
          border-radius: 10px; /* Rounded corners for the track */
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: var(--color-muted-foreground); /* Darker gray thumb */
          border-radius: 10px; /* Rounded corners for the thumb */
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: var(--color-foreground); /* Even darker gray on hover */
        }
      `}</style>

      {/* Header Section: New Chat Button */}
      {/* `sticky top-0 bg-card z-20` makes this section stick to the top when scrolling. */}
      <div className="p-4 border-b border-border flex-shrink-0 sticky top-0 bg-card z-20">
        <button
          onClick={() => onSelectSession(null)} // Calling onSelectSession with null initiates a new chat.
          className="w-full flex items-center justify-center space-x-2 bg-primary text-primary-foreground rounded-lg px-4 py-3 hover:bg-primary/90 transition-colors duration-200 font-medium shadow-md"
          aria-label="Start a new chat conversation"
          title="Start New Chat"
        >
          <Plus className="h-4 w-4" /> {/* Plus icon for new chat. */}
          <span>New Chat</span>
        </button>
      </div>

      {/* Search Input Section */}
      {/* `sticky top-[72px]` makes this section stick below the header when scrolling. */}
      <div className="p-4 border-b border-border flex-shrink-0 sticky top-[72px] bg-card z-20">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" /> {/* Search icon. */}
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)} // Update search term state.
            className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200 text-foreground placeholder-muted-foreground"
            aria-label="Search conversations"
          />
        </div>
      </div>

      {/* Sessions List Area */}
      {/* `flex-1 overflow-y-auto custom-scrollbar min-h-0 h-0` ensures this area scrolls independently and takes remaining space. */}
      <div className="flex-1 overflow-y-auto custom-scrollbar min-h-0 h-0">
        {/* Sticky header for "Recent Conversations" - only shown if there are sessions. */}
        {safeSessions.length > 0 && (
          <div className="sticky top-0 bg-card z-10 text-xs font-bold text-muted-foreground uppercase tracking-wider px-4 py-2 border-b border-border">
            Recent Conversations
          </div>
        )}

        <div className="p-2">
          <div className="space-y-1">
            {sessionsToDisplay.length > 0 ? (
              // Map through sessions to display each one.
              sessionsToDisplay.map((session) => (
                <div
                  key={session.session_id}
                  onClick={() => onSelectSession(session.session_id, session.session_title)} // Select session on click.
                  className={`group flex items-start space-x-3 p-3 rounded-lg cursor-pointer transition-all duration-200 shadow-sm
                    ${currentSessionId === session.session_id
                      ? 'bg-accent text-accent-foreground' // Active session styling.
                      : 'hover:bg-accent/50 text-muted-foreground' // Inactive hover styling.
                    }`}
                  role="button"
                  tabIndex="0"
                  aria-selected={currentSessionId === session.session_id}
                  aria-label={`Select conversation: ${session.session_title}`}
                >
                  <div className="flex-shrink-0 mt-1">
                    <MessageSquare className="h-4 w-4 text-muted-foreground" /> {/* Message icon. */}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <h3 className="text-sm font-medium truncate text-foreground">
                        {session.session_title} {/* Session title. */}
                      </h3>
                      {/* Conditional rendering for delete confirmation or delete icon. */}
                      {deleteConfirm === session.session_id ? (
                        // Confirmation buttons for deletion.
                        <div className="flex items-center space-x-1">
                          <button
                            onClick={(e) => handleConfirmDelete(session.session_id, e)}
                            className="p-1 rounded-full bg-green-500 text-white hover:bg-green-600 transition-colors duration-200"
                            title="Confirm Delete"
                            aria-label="Confirm delete session"
                          >
                            <CheckCircle className="h-4 w-4" /> {/* Check icon for confirm. */}
                          </button>
                          <button
                            onClick={(e) => handleCancelDelete(e)}
                            className="p-1 rounded-full bg-red-500 text-white hover:bg-red-600 transition-colors duration-200"
                            title="Cancel Delete"
                            aria-label="Cancel delete session"
                          >
                            <XCircle className="h-4 w-4" /> {/* X icon for cancel. */}
                          </button>
                        </div>
                      ) : (
                        // Delete button (initially hidden, appears on group hover).
                        <button
                          onClick={(e) => handleDeleteClick(session.session_id)}
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-background rounded transition-all duration-200"
                          title="Delete Conversation"
                          aria-label="Delete conversation"
                        >
                          <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" /> {/* Trash icon. */}
                        </button>
                      )}
                    </div>

                    <p className="text-xs text-muted-foreground truncate mb-2">
                      {session.last_message || "No messages yet"} {/* Last message snippet. */}
                    </p>

                    <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" /> {/* Clock icon. */}
                      <span>{formatTimestamp(session.timestamp)}</span> {/* Formatted timestamp. */}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              // Message displayed when no conversations are found.
              <p className="text-sm text-muted-foreground px-3 py-2 text-center">No conversations found. Start a new chat!</p>
            )}
          </div>

          {/* "Show More" Button */}
          {/* Only visible if there are more sessions to load beyond the current view. */}
          {hasMoreSessions && (
            <div className="text-center mt-4">
              <button
                onClick={handleShowMore}
                className="inline-flex items-center px-4 py-2 border border-input rounded-lg shadow-sm text-sm font-medium text-muted-foreground hover:bg-muted transition-colors duration-200"
                aria-label="Show more conversations"
              >
                Show More
                <ChevronDown className="ml-2 h-4 w-4" /> {/* Down arrow icon. */}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Footer Section of the Sidebar */}
      {/* `sticky bottom-0 bg-card z-20` makes this section stick to the bottom. */}
      <div className="p-4 border-t border-border flex-shrink-0 sticky bottom-0 bg-card z-20 shadow-inner">
        <div className="text-xs text-muted-foreground text-center space-y-1">
          <p>Pyrotech AI Assistant</p>
          <p className="font-semibold">Enterprise Document AI Solution</p>
          <p className="text-xxs opacity-70">Version 1.0.0</p>
        </div>
      </div>
    </div>
  );
};

export default SessionSidebar;
