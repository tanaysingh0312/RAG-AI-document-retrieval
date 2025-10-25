import React, { useState, useEffect, useCallback } from 'react';
// Importing all necessary child components for the main application structure.
// The .tsx extension is omitted as per standard React import practices,
// relying on bundler resolution.
import LoginForm from './components/LoginForm';
import HeaderBar from './components/HeaderBar';
import SessionSidebar from './components/SessionSidebar';
import ChatWindow from './components/ChatWindow';
import AdminPanel from './components/AdminPanel';

/**
 * App Component
 * This is the main entry point of the React application. It manages:
 * - Global UI states such as dark mode and sidebar visibility.
 * - User authentication status and user data.
 * - Navigation between different application views (e.g., chat, admin).
 * - Management of chat sessions and their historical messages.
 * - Communication with the backend API for core functionalities.
 */
function App() {
  // --- UI and Layout State Management ---
  // State variable to control the application's theme (dark or light mode).
  const [darkMode, setDarkMode] = useState(false);
  // State variable to control the visibility of the left-hand session sidebar.
  const [sidebarOpen, setSidebarOpen] = useState(true);
  // State variable to determine which main content panel is currently displayed.
  // Possible values are 'chat' for the conversation interface and 'admin' for the administration panel.
  const [currentView, setCurrentView] = useState('chat');

  // --- User Authentication and Profile State ---
  // State variable to hold the authenticated user's data (e.g., username, role).
  // It is null if no user is currently logged in.
  const [user, setUser] = useState(null);
  // State variable to indicate if the initial authentication check is still in progress.
  // This helps in displaying a loading spinner or preventing UI interactions before auth is confirmed.
  const [isLoadingAuth, setIsLoadingAuth] = useState(true);

  // --- Chat Session and Message History State ---
  // State variable to store an array of all chat sessions belonging to the current user.
  // Each session typically includes an ID and a title.
  const [sessions, setSessions] = useState([]);
  // State variable to hold the unique identifier of the chat session currently active and displayed.
  // It is null when a new, unsaved chat is in progress.
  const [currentSessionId, setCurrentSessionId] = useState(null);
  // State variable to store the user-friendly title of the currently active chat session.
  // Defaults to "New Chat" for unsaved conversations.
  const [currentSessionTitle, setCurrentSessionTitle] = useState("New Chat");
  // State variable to store the array of messages for the currently active chat session.
  // This array is displayed in the ChatWindow component.
  const [chatHistory, setChatHistory] = useState([]);

  // Dynamically determines the base URL for API calls.
  // This ensures flexibility whether the app is running locally (e.g., http://localhost:3000)
  // or deployed behind a proxy (e.g., https://yourdomain.com).
  const API_BASE_URL = `${window.location.origin}/api`;

  // --- Global Effects for UI and Authentication Lifecycle ---

  /**
   * useEffect hook to manage the 'dark' class on the document's root element.
   * This effect runs whenever the `darkMode` state changes, applying or removing
   * the class to toggle the global theme.
   */
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    // No explicit cleanup needed as classList operations are idempotent.
  }, [darkMode]); // Dependency array: effect re-runs when `darkMode` changes.

  /**
   * useEffect hook for performing an initial authentication check when the application loads.
   * It attempts to retrieve a JWT token from localStorage, validates its expiration,
   * and if valid, proceeds to fetch the current user's profile from the backend.
   * This ensures that users remain logged in across page refreshes if their token is valid.
   */
  useEffect(() => {
    const performInitialAuthCheck = async () => {
      setIsLoadingAuth(true); // Set loading state to true at the start of the auth check.
      const token = localStorage.getItem("token"); // Retrieve the authentication token.

      if (token) {
        try {
          // Attempt to decode the JWT token to inspect its payload, specifically the expiration time.
          const tokenParts = token.split('.');
          if (tokenParts.length !== 3) {
            throw new Error("Invalid JWT token format.");
          }
          const decodedPayload = JSON.parse(atob(tokenParts[1])); // Base64 decode the payload.
          const expirationTime = decodedPayload.exp * 1000; // Convert Unix timestamp (seconds) to milliseconds.

          // Check if the token is still valid (not expired).
          if (expirationTime > Date.now()) {
            // If the token is valid, proceed to fetch the full user profile from the backend.
            await fetchCurrentUser(token);
          } else {
            // If the token has expired, log a message, remove the invalid token, and reset user state.
            console.log("Authentication token has expired. User needs to log in again.");
            localStorage.removeItem("token");
            setUser(null);
            setIsLoadingAuth(false); // Authentication process is complete, no user logged in.
          }
        } catch (error) {
          // Catch any errors during token decoding or initial user fetch.
          console.error("Error during initial authentication check:", error);
          localStorage.removeItem("token"); // Clear potentially corrupted or invalid token.
          setUser(null);
          setIsLoadingAuth(false); // Authentication process is complete due to an error.
        }
      } else {
        // If no token is found in localStorage, the authentication process is considered complete,
        // and no user is logged in.
        setIsLoadingAuth(false);
      }
    };

    // Execute the initial authentication check when the component mounts.
    performInitialAuthCheck();
  }, [API_BASE_URL]); // Dependency array: `API_BASE_URL` is included to ensure the fetch call uses the correct base URL.

  // --- API Interaction Functions (Memoized with useCallback for performance) ---

  /**
   * Fetches the details of the currently authenticated user from the backend.
   * This function is typically called after a successful login or when validating an existing token.
   * @param {string} token - The JWT authentication token for the current user.
   */
  const fetchCurrentUser = useCallback(async (token) => {
    try {
      const response = await fetch(`${API_BASE_URL}/users/me/`, {
        method: 'GET', // Explicitly specify the GET HTTP method.
        headers: {
          'Content-Type': 'application/json', // Indicate that we expect JSON.
          'Authorization': `Bearer ${token}`, // Include the JWT for authentication.
        },
      });

      if (!response.ok) {
        // If the HTTP response status is not OK (e.g., 401, 403, 500), throw an error.
        const errorData = await response.json(); // Attempt to parse error details from response body.
        throw new Error(errorData.detail || `HTTP error! Status: ${response.status} during user fetch.`);
      }

      const userData = await response.json(); // Parse the successful JSON response.
      setUser(userData); // Update the global user state with the fetched data.
      console.log("User profile successfully loaded:", userData.username, userData.role);
      // After successfully loading the user profile, proceed to fetch their chat sessions.
      await fetchSessions(userData.username, token);
    } catch (error) {
      console.error("Failed to fetch current user profile:", error);
      localStorage.removeItem("token"); // Clear the token if fetching the user profile fails (e.g., invalid token).
      setUser(null); // Reset user state to null.
    } finally {
      setIsLoadingAuth(false); // Ensure authentication loading state is reset.
    }
  }, [API_BASE_URL, fetchSessions]); // Dependencies: `API_BASE_URL` and `fetchSessions` (because it's called here).

  /**
   * Fetches all chat sessions associated with a specific user from the backend.
   * @param {string} username - The username whose sessions are to be retrieved.
   * @param {string} token - The JWT authentication token.
   */
  const fetchSessions = useCallback(async (username, token) => {
    // Guard clause: do not proceed if essential parameters are missing.
    if (!username || !token) {
      console.warn("Cannot fetch sessions: username or authentication token is missing.");
      setSessions([]); // Ensure sessions state is empty if prerequisites are not met.
      return;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'GET', // Explicitly specify the GET HTTP method.
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`, // Include the JWT for authentication.
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! Status: ${response.status} during session fetch.`);
      }

      const fetchedSessions = await response.json(); // Parse the array of session objects.
      setSessions(fetchedSessions); // Update the global sessions state.

      // Logic to automatically select a session if none is currently active or if the active one is invalid.
      if (!currentSessionId && fetchedSessions.length > 0) {
        // If no session is selected, and sessions exist, select the most recent one.
        // Assumes backend returns sessions sorted by recency (most recent first).
        const mostRecentSession = fetchedSessions[0];
        handleSelectSession(mostRecentSession.session_id, mostRecentSession.session_title);
      } else if (currentSessionId && !fetchedSessions.some(s => s.session_id === currentSessionId)) {
        // If a session was previously selected but no longer exists in the fetched list (e.g., deleted),
        // reset to a "New Chat" state.
        handleSelectSession(null);
      }
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
      setSessions([]); // Clear sessions on any error during fetching.
    }
  }, [API_BASE_URL, currentSessionId]); // Dependencies: `API_BASE_URL` and `currentSessionId` (to react to changes in selected session).

  /**
   * Fetches the detailed message history for a specific chat session from the backend.
   * @param {string} sessionId - The unique ID of the session whose history is to be fetched.
   * @param {string} token - The JWT authentication token.
   */
  const fetchChatHistory = useCallback(async (sessionId, token) => {
    // Guard clause: clear history if session ID or token is missing.
    if (!sessionId || !token) {
      setChatHistory([]);
      return;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'GET', // Explicitly specify the GET HTTP method.
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`, // Include the JWT for authentication.
        },
      });

      if (!response.ok) {
        // Handle specific case for 404 Not Found if session doesn't exist.
        if (response.status === 404) {
          console.log(`Chat session with ID ${sessionId} not found. It might have been deleted.`);
          setChatHistory([]); // Clear history for non-existent sessions.
          return;
        }
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! Status: ${response.status} during chat history fetch.`);
      }

      const sessionData = await response.json(); // Parse the session data, which contains messages.
      // Assuming `sessionData.messages` is an array of message objects from the backend.
      // Format these messages to align with the frontend's expected message structure.
      const formattedMessages = sessionData.messages.map(msg => ({
        id: msg.id, // Unique identifier for the message.
        role: msg.role, // The role of the sender ('user' or 'assistant').
        content: msg.message, // The actual text content of the message.
        timestamp: msg.timestamp // The timestamp when the message was created.
      }));
      setChatHistory(formattedMessages); // Update the global chat history state.
    } catch (error) {
      console.error("Error fetching chat history:", error);
      setChatHistory([]); // Clear history on any error during fetching.
    }
  }, [API_BASE_URL]); // Dependencies: `API_BASE_URL`.

  // --- UI Interaction Event Handlers ---

  /**
   * Toggles the `darkMode` state. This function is passed down to child components
   * (e.g., LoginForm, HeaderBar) to allow users to switch themes.
   */
  const handleToggleDarkMode = () => {
    setDarkMode(prevMode => !prevMode); // Toggle the boolean state.
  };

  /**
   * Toggles the `sidebarOpen` state. This function is passed to the HeaderBar
   * component to control the visibility of the session sidebar.
   */
  const handleToggleSidebar = () => {
    setSidebarOpen(prev => !prev); // Toggle the boolean state.
  };

  /**
   * Callback function executed when a user successfully logs in via the LoginForm.
   * It sets the authenticated user's data, switches to the chat view, and fetches their sessions.
   * @param {object} loggedInUser - The user object containing `username` and `role`.
   */
  const handleLoginSuccess = (loggedInUser) => {
    setUser(loggedInUser); // Update the global user state.
    setCurrentView('chat'); // Automatically navigate to the chat view after login.
    // Fetch chat sessions for the newly logged-in user.
    fetchSessions(loggedInUser.username, localStorage.getItem("token"));
  };

  /**
   * Handles the user logout process. This involves:
   * - Clearing the authentication token from local storage.
   * - Resetting all user-related and session-related states to their initial values.
   * - Navigating back to the login screen implicitly by clearing the `user` state.
   */
  const handleLogout = () => {
    localStorage.removeItem("token"); // Remove the JWT token.
    setUser(null); // Clear the user object.
    setSessions([]); // Clear all sessions.
    setCurrentSessionId(null); // Reset current session ID.
    setCurrentSessionTitle("New Chat"); // Reset session title.
    setChatHistory([]); // Clear chat messages.
    setCurrentView('chat'); // Reset view to default chat.
    setSidebarOpen(true); // Ensure sidebar is open on next login.
  };

  /**
   * Changes the active main content view of the application.
   * This is used by the HeaderBar to switch between 'chat' and 'admin' panels.
   * @param {string} view - The new view to set ('chat' or 'admin').
   */
  const handleViewChange = (view) => {
    setCurrentView(view); // Update the current view state.
    // If switching to the chat view and no sessions are loaded for the current user,
    // trigger a fetch for sessions.
    if (view === 'chat' && user && sessions.length === 0) {
      fetchSessions(user.username, localStorage.getItem("token"));
    }
  };

  /**
   * Handles the selection of a chat session from the SessionSidebar or initiates a new chat.
   * When `sessionId` is `null`, it signifies starting a new conversation.
   * @param {string | null} sessionId - The ID of the session to select, or `null` for a new chat.
   * @param {string} [sessionTitle="New Chat"] - The title of the selected session. Defaults to "New Chat".
   */
  const handleSelectSession = useCallback((sessionId, sessionTitle = "New Chat") => {
    setCurrentSessionId(sessionId); // Update the current session ID.
    setCurrentSessionTitle(sessionTitle); // Update the current session title.
    setChatHistory([]); // Clear the current chat history immediately to provide visual feedback
                       // (e.g., showing a blank screen before new history loads or initial message appears).

    if (sessionId && user) {
      // If a valid session ID is provided and a user is logged in, fetch the chat history for that session.
      fetchChatHistory(sessionId, localStorage.getItem("token"));
    } else if (sessionId === null) {
      // If `sessionId` is null, it means a new chat is being started.
      // Populate the chat history with an initial greeting from the assistant.
      setChatHistory([
        {
          id: 'initial-assistant-message', // A unique, static ID for this initial message.
          role: 'assistant', // The sender is the assistant.
          content: 'Hello! I\'m your Pyrotech AI Document Assistant. I can help you analyze documents, answer questions, and provide insights. How can I assist you today?',
          timestamp: new Date().toISOString() // Current timestamp for the message.
        }
      ]);
    }
  }, [user, fetchChatHistory]); // Dependencies: `user` (to check authentication) and `fetchChatHistory` (to call it).

  /**
   * Handles the deletion of a chat session.
   * Sends a DELETE request to the backend API and updates the local state.
   * @param {string} sessionIdToDelete - The unique ID of the session to be deleted.
   */
  const handleDeleteSession = async (sessionIdToDelete) => {
    // Prevent deletion if the user is not authenticated or the token is missing.
    if (!user || !localStorage.getItem("token")) {
      console.warn("Cannot delete session: User not authenticated or token missing.");
      // Optionally, show a user-friendly error message here.
      return;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionIdToDelete}`, {
        method: 'DELETE', // Use the HTTP DELETE method.
        headers: {
          'Authorization': `Bearer ${localStorage.getItem("token")}`, // Authenticate the request.
        },
      });

      if (!response.ok) {
        // If the server responds with an error status, parse and throw.
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! Status: ${response.status}, Failed to delete session.`);
      }
      console.log(`Chat session with ID ${sessionIdToDelete} deleted successfully.`);
      // After successful deletion, re-fetch the list of sessions to update the sidebar.
      await fetchSessions(user.username, localStorage.getItem("token"));
      // If the deleted session was the one currently active, reset to a "New Chat" state.
      if (currentSessionId === sessionIdToDelete) {
        handleSelectSession(null);
      }
    } catch (error) {
      console.error("Error deleting chat session:", error);
      // Provide user feedback about the failure to delete the session.
      window.alert(`Failed to delete session: ${error.message}`); // Using window.alert for simplicity, a custom modal is recommended.
    }
  };

  /**
   * Callback function invoked by the ChatWindow component when a new chat session
   * is successfully created (e.g., when the first message is sent in a "New Chat").
   * This updates the `App` component's state to reflect the new session.
   * @param {string} newSessionId - The unique ID of the newly created session.
   * @param {string} newSessionTitle - The title of the newly created session.
   */
  const handleNewSessionCreated = useCallback((newSessionId, newSessionTitle) => {
    setCurrentSessionId(newSessionId); // Update the current session ID to the new one.
    setCurrentSessionTitle(newSessionTitle); // Update the current session title.
    // Trigger a re-fetch of sessions to ensure the new session appears in the sidebar.
    if (user) {
      fetchSessions(user.username, localStorage.getItem("token"));
    }
    // The chat history for this new session will be automatically loaded by the
    // `useEffect` hook that watches `currentSessionId`.
  }, [user, fetchSessions]); // Dependencies: `user` (for fetching sessions) and `fetchSessions`.

  // --- Conditional Rendering for Application Loading State ---

  // If `isLoadingAuth` is true, display a full-screen loading spinner.
  if (isLoadingAuth) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-gray-50 to-gray-200 dark:from-gray-900 dark:to-gray-800 text-foreground">
        <div className="flex flex-col items-center space-y-4 p-8 rounded-lg shadow-xl bg-white dark:bg-gray-800">
          <div className="w-20 h-20 border-4 border-pyrotech-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-xl font-semibold text-gray-700 dark:text-gray-300">Loading Pyrotech AI Application...</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Please wait while we prepare your workspace.</p>
        </div>
      </div>
    );
  }

  // --- Main Application Layout Render ---
  return (
    // The outermost container for the entire application.
    // Sets full viewport height, uses flexbox for a column layout, and applies global theme/font.
    <div className={`flex flex-col h-screen ${darkMode ? 'dark' : ''} font-inter bg-background text-foreground`}>
      {!user ? (
        // If no user is authenticated, display the LoginForm.
        // The login form is centered on the screen.
        <div className="flex items-center justify-center flex-grow min-h-screen bg-gradient-to-br from-gray-50 to-gray-200 dark:from-gray-900 dark:to-gray-800">
          <LoginForm
            onLogin={handleLoginSuccess} // Pass the callback for successful login.
            darkMode={darkMode} // Pass the current dark mode state.
            onToggleDarkMode={handleToggleDarkMode} // Pass the function to toggle dark mode.
          />
        </div>
      ) : (
        // If a user is authenticated, render the main application interface.
        <>
          {/* Header Bar Component: Displays application title, navigation, user info, and theme toggle. */}
          {/* It remains at the top and does not scroll with content. */}
          <HeaderBar
            darkMode={darkMode} // Current dark mode state.
            onToggleDarkMode={handleToggleDarkMode} // Function to toggle dark mode.
            sidebarOpen={sidebarOpen} // Current sidebar visibility state.
            onToggleSidebar={handleToggleSidebar} // Function to toggle sidebar visibility.
            currentView={currentView} // Current active view ('chat' or 'admin').
            onViewChange={handleViewChange} // Function to change the active view.
            onLogout={handleLogout} // Function to handle user logout.
            user={user} // The authenticated user object.
          />
          {/* Main content area: A flexible container that holds the sidebar and the main content panel. */}
          {/* `flex-grow` allows it to take up all remaining vertical space. `overflow-hidden` prevents unwanted scrollbars. */}
          <div className="flex flex-grow overflow-hidden h-full">
            {/* Session Sidebar Component: Displays and manages chat sessions. */}
            {/* Its visibility is controlled by `isOpen` prop. */}
            <SessionSidebar
              isOpen={sidebarOpen} // Controls if the sidebar is visible.
              currentView={currentView} // Current application view.
              onViewChange={handleViewChange} // Function to change views.
              sessions={sessions} // List of chat sessions for display.
              onSelectSession={handleSelectSession} // Callback for selecting a session.
              onDeleteSession={handleDeleteSession} // Callback for deleting a session.
              currentSessionId={currentSessionId} // The currently active session ID.
              user={user} // The authenticated user object.
            />

            {/* Main Content Pane: Renders either the ChatWindow or the AdminPanel. */}
            {/* `flex-1` makes it take up all remaining horizontal space. */}
            {/* `min-h-0` and `h-full` are essential for flex items to correctly manage their height with `overflow-y-auto` children. */}
            <main className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${
              // This class dynamically adjusts margin based on sidebarOpen,
              // though with the current sidebar implementation, `ml-0` is often sufficient.
              sidebarOpen ? 'ml-0' : 'ml-0'
            } min-h-0 h-full`}>
              {currentView === 'chat' && (
                // Render ChatWindow when `currentView` is 'chat'.
                <ChatWindow
                  user={user} // Pass the authenticated user.
                  currentSessionId={currentSessionId} // Pass the active session ID.
                  currentSessionTitle={currentSessionTitle} // Pass the active session title.
                  chatHistory={chatHistory} // Pass the messages for the active session.
                  setChatHistory={setChatHistory} // Allow ChatWindow to update history (e.g., for optimistic updates).
                  onNewSessionCreated={handleNewSessionCreated} // Callback for new session creation.
                  sidebarOpen={sidebarOpen} // Pass sidebar state for responsive layout adjustments.
                />
              )}
              {currentView === 'admin' && (
                // Render AdminPanel when `currentView` is 'admin'.
                <AdminPanel user={user} /> // Pass the authenticated user to AdminPanel.
              )}
            </main>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
