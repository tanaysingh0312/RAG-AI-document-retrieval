import React, { useState, useEffect, useCallback } from 'react';
// Importing various icons from 'lucide-react' for a modern and intuitive user interface.
import { Users, FileText, Settings, BarChart3, Shield, Database, Globe, Zap, Loader2 } from 'lucide-react';

/**
 * AdminPanel Component
 * This component provides a dedicated interface for administrators to:
 * - Monitor overall system health and statistics.
 * - Manage user accounts (though basic in this version, extensible).
 * - Control document ingestion and clearing operations for the RAG system.
 *
 * It ensures that only users with 'admin' role can access its functionalities.
 *
 * @param {object} props - The properties passed to the component.
 * @param {object} props.user - The authenticated user object, which includes their role.
 */
const AdminPanel = ({ user }) => {
  // State to manage the currently active tab within the admin panel.
  // Defaults to 'overview' to show system statistics first.
  const [activeTab, setActiveTab] = useState('overview');
  // State to store the list of users fetched for the 'Users' tab.
  const [users, setUsers] = useState([]);
  // State to indicate if user data is currently being loaded from the backend.
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);
  // State to store the total count of documents processed by the RAG system.
  const [totalDocuments, setTotalDocuments] = useState('N/A');
  // State to store the count of API calls made today, for analytics.
  const [apiCallsToday, setApiCallsToday] = useState('N/A');
  // State to store the number of active chat sessions.
  const [activeSessions, setActiveSessions] = useState('N/A');
  // State to indicate if general statistics (overview) are currently being loaded.
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  // State to store any error messages that need to be displayed to the user.
  const [error, setError] = useState('');
  // State to store detailed backend health information, fetched from a dedicated endpoint.
  const [backendHealth, setBackendHealth] = useState({});

  // Base URL for API calls, dynamically determined to work in various environments.
  const API_BASE_URL = `${window.location.origin}/api`;

  // Array defining the dashboard statistics cards.
  // Each object contains a label, initial value, a placeholder change, an icon, and a color gradient.
  // The `value` property will be updated dynamically by fetched data.
  const statsCardsData = [
    { label: 'Total Users', value: users.length.toLocaleString(), change: '+12%', icon: Users, color: 'from-blue-500 to-blue-600' },
    { label: 'Documents Processed', value: totalDocuments, change: '+8%', icon: FileText, color: 'from-green-500 to-green-600' },
    { label: 'API Calls Today', value: apiCallsToday, change: '+23%', icon: Zap, color: 'from-purple-500 to-purple-600' },
    { label: 'Active Sessions', value: activeSessions, change: '+5%', icon: Globe, color: 'from-orange-500 to-orange-600' },
  ];

  // Array defining the navigation tabs for the admin panel.
  // Each object has an ID, a display label, and an associated icon.
  const tabsNavigationData = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'documents', label: 'Documents', icon: FileText },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  // --- Data Fetching Functions (Memoized with useCallback) ---

  /**
   * Fetches the list of all users from the backend.
   * This function is called when the 'Users' tab is active.
   */
  const fetchUsers = useCallback(async () => {
    // Ensure only admins can fetch user data.
    if (user?.role !== 'admin') {
      setError("Access Denied: You do not have administrative privileges to view user data.");
      setUsers([]); // Clear any stale user data.
      return;
    }
    setIsLoadingUsers(true); // Set loading state for users.
    setError(''); // Clear previous errors.
    try {
      const token = localStorage.getItem("token"); // Retrieve the authentication token.
      if (!token) {
        throw new Error("Authentication token not found. Please log in.");
      }

      const response = await fetch(`${API_BASE_URL}/users`, {
        method: 'GET', // Explicitly use GET method.
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`, // Include JWT for authentication.
        },
      });

      if (!response.ok) {
        const errorData = await response.json(); // Parse error details from response.
        throw new Error(errorData.detail || `Failed to fetch users: HTTP status ${response.status}.`);
      }

      const data = await response.json();
      // Assuming backend returns an object like { "users": [...] }.
      setUsers(data.users || []); // Update the users state.
    } catch (err) {
      console.error("Error fetching users data:", err);
      setError(err.message || 'An unexpected error occurred while fetching users.');
      setUsers([]); // Clear users on error.
    } finally {
      setIsLoadingUsers(false); // Reset loading state.
    }
  }, [API_BASE_URL, user]); // Dependencies: `API_BASE_URL` and `user` (to react to user role changes).

  /**
   * Fetches various system statistics and backend health information.
   * This function is called when the 'Overview' tab is active.
   */
  const fetchStats = useCallback(async () => {
    // Only fetch stats if the user is an admin.
    if (user?.role !== 'admin') {
      return;
    }
    setIsLoadingStats(true); // Set loading state for stats.
    setError(''); // Clear previous errors.
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token not found. Please log in.");
      }
      const authHeaders = { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' };

      // Use Promise.all to fetch multiple data points concurrently for efficiency.
      const [
        userCountResponse,
        docCountResponse,
        apiCallsResponse,
        activeSessionsResponse,
        healthResponse,
      ] = await Promise.all([
        fetch(`${API_BASE_URL}/users/count`, { headers: authHeaders }),
        fetch(`${API_BASE_URL}/documents/count`, { headers: authHeaders }),
        fetch(`${API_BASE_URL}/analytics/api_calls_today`, { headers: authHeaders }),
        fetch(`${API_BASE_URL}/sessions/active`, { headers: authHeaders }),
        fetch(`${API_BASE_URL}/health`, { headers: authHeaders }), // Fetch detailed backend health.
      ]);

      // Process User Count
      if (userCountResponse.ok) {
        const userCountData = await userCountResponse.json();
        // Update the `users` state with a dummy array of the correct length to reflect total count in `statsCardsData`.
        // The actual user list for the 'Users' tab is handled by `fetchUsers`.
        setUsers(Array(userCountData.count).fill({}));
        // Update the value in the `statsCardsData` array (by direct modification or creating a new array).
        // For simplicity and direct update, we're modifying the `statsCardsData` array directly here.
        statsCardsData[0].value = userCountData.count.toLocaleString();
      } else {
        console.error("Failed to fetch user count:", await userCountResponse.json());
        statsCardsData[0].value = 'Error';
      }

      // Process Document Count
      if (docCountResponse.ok) {
        const docCountData = await docCountResponse.json();
        setTotalDocuments(docCountData.count.toLocaleString());
        statsCardsData[1].value = docCountData.count.toLocaleString();
      } else {
        console.error("Failed to fetch document count:", await docCountResponse.json());
        setTotalDocuments('Error');
        statsCardsData[1].value = 'Error';
      }

      // Process API Calls Today
      if (apiCallsResponse.ok) {
        const apiCallsData = await apiCallsResponse.json();
        setApiCallsToday(apiCallsData.count.toLocaleString());
        statsCardsData[2].value = apiCallsData.count.toLocaleString();
      } else {
        console.error("Failed to fetch API calls today:", await apiCallsResponse.json());
        setApiCallsToday('Error');
        statsCardsData[2].value = 'Error';
      }

      // Process Active Sessions
      if (activeSessionsResponse.ok) {
        const activeSessionsData = await activeSessionsResponse.json();
        setActiveSessions(activeSessionsData.count.toLocaleString());
        statsCardsData[3].value = activeSessionsData.count.toLocaleString();
      } else {
        console.error("Failed to fetch active sessions:", await activeSessionsResponse.json());
        setActiveSessions('Error');
        statsCardsData[3].value = 'Error';
      }

      // Process Backend Health
      if (healthResponse.ok) {
        const healthData = await healthResponse.json();
        setBackendHealth(healthData.details || {}); // Store detailed health info.
      } else {
        console.error("Failed to fetch backend health:", await healthResponse.json());
        setBackendHealth({ status: 'Error', message: 'Failed to fetch health data.' });
      }

    } catch (e) {
      console.error("Error fetching dashboard statistics:", e);
      setError(e.message || "Failed to load some dashboard statistics.");
    } finally {
      setIsLoadingStats(false); // Reset loading state for stats.
    }
  }, [API_BASE_URL, user, statsCardsData]); // Dependencies: `API_BASE_URL`, `user`, and `statsCardsData` (for direct updates).

  // --- Effects for Tab-specific Data Loading ---

  /**
   * useEffect hook to trigger data fetching based on the active tab.
   * - If 'users' tab is active, `fetchUsers` is called.
   * - If 'overview' tab is active, `fetchStats` is called.
   */
  useEffect(() => {
    if (user?.role === 'admin') { // Ensure only admins trigger these fetches.
      if (activeTab === 'users') {
        fetchUsers();
      } else if (activeTab === 'overview') {
        fetchStats();
      }
    }
  }, [activeTab, user, fetchUsers, fetchStats]); // Dependencies: `activeTab`, `user`, and the memoized fetch functions.

  // --- Document Management Actions ---

  /**
   * Triggers the document ingestion process on the backend.
   * This re-indexes all processed documents into the RAG knowledge base.
   */
  const triggerDocumentIngestion = async () => {
    // Ensure only admins can perform this action.
    if (user?.role !== 'admin') {
      setError("Access Denied: You do not have administrative privileges to trigger document ingestion.");
      return;
    }
    // Confirmation dialog for critical action.
    const isConfirmed = window.confirm("Are you sure you want to trigger document ingestion? This will re-index all processed documents and might take some time.");
    if (!isConfirmed) {
      return; // Abort if not confirmed.
    }

    setError(''); // Clear previous errors.
    setIsLoadingStats(true); // Show loading indicator for the action.
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token not found. Please log in.");
      }

      const response = await fetch(`${API_BASE_URL}/ingest_processed_documents/`, {
        method: "POST", // Use POST method for triggering an action.
        headers: {
          "Authorization": `Bearer ${token}`, // Authenticate the request.
          "Content-Type": "application/json" // Specify content type.
        },
        body: JSON.stringify({}) // Send an empty JSON body if no specific payload is needed.
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to trigger document ingestion: HTTP status ${response.status}.`);
      }
      const responseData = await response.json();
      // Using window.alert for simplicity as per previous code, but a custom modal is recommended for better UX.
      window.alert(`Document ingestion triggered successfully! Message: ${responseData.message}`);
      fetchStats(); // Refresh stats after ingestion to see updated document counts.
    } catch (err) {
      console.error("Error triggering document ingestion:", err);
      setError(err.message || 'An unexpected error occurred during document ingestion.');
      window.alert(`Document ingestion failed: ${err.message}`);
    } finally {
      setIsLoadingStats(false); // Hide loading indicator.
    }
  };

  /**
   * Clears all documents from the RAG knowledge base on the backend.
   * This is a destructive action and requires user confirmation.
   */
  const clearAllDocuments = async () => {
    // Ensure only admins can perform this action.
    if (user?.role !== 'admin') {
      setError("Access Denied: You do not have administrative privileges to clear documents.");
      return;
    }
    // Critical confirmation dialog for a destructive action.
    const isConfirmed = window.confirm("WARNING: Are you absolutely sure you want to clear ALL documents from the RAG knowledge base? This action cannot be undone and will remove all indexed data.");
    if (!isConfirmed) {
      return; // Abort if not confirmed.
    }

    setError(''); // Clear previous errors.
    setIsLoadingStats(true); // Show loading indicator.
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token not found. Please log in.");
      }

      const response = await fetch(`${API_BASE_URL}/clear_documents/`, {
        method: "DELETE", // Use DELETE method for clearing resources.
        headers: {
          "Authorization": `Bearer ${token}`, // Authenticate the request.
          "Content-Type": "application/json"
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to clear documents: HTTP status ${response.status}.`);
      }
      const responseData = await response.json();
      // Using window.alert for simplicity.
      window.alert(`All documents cleared successfully! Message: ${responseData.message}`);
      fetchStats(); // Refresh stats after clearing to reflect zero documents.
    } catch (err) {
      console.error("Error clearing documents:", err);
      setError(err.message || 'An unexpected error occurred while clearing documents.');
      window.alert(`Clearing documents failed: ${err.message}`);
    } finally {
      setIsLoadingStats(false); // Hide loading indicator.
    }
  };

  // --- Access Control Rendering ---

  // If the user is not logged in or does not have 'admin' role, display an access denied message.
  if (!user || user.role !== 'admin') {
    return (
      <div className="flex flex-col items-center justify-center h-full w-full p-4 bg-background text-foreground text-center">
        <Shield className="h-20 w-20 text-destructive mb-6 animate-pulse" /> {/* Larger, animated shield icon */}
        <h2 className="text-3xl font-bold text-foreground mb-3">Access Denied</h2>
        <p className="text-lg text-muted-foreground max-w-md">
          You must be logged in as an administrator to access this panel.
          Please ensure your account has the necessary permissions.
        </p>
        <button
          onClick={() => window.location.reload()} // Simple reload to go back to login or main app.
          className="mt-8 px-6 py-3 bg-primary text-primary-foreground rounded-lg shadow-md hover:bg-primary/90 transition-colors duration-200"
        >
          Go to Login / Main App
        </button>
      </div>
    );
  }

  // --- Main Admin Panel Render ---
  return (
    // Main container for the admin panel, taking full height and enabling vertical scrolling.
    <div className="flex-1 flex flex-col p-4 md:p-6 lg:p-8 bg-background overflow-y-auto custom-scrollbar">
      {/* Dashboard Header Section */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground mb-2">Admin Dashboard</h1>
          <p className="text-muted-foreground text-base">Monitor system performance and manage application resources.</p>
        </div>
        {/* Navigation Tabs for different admin sections */}
        <div className="flex space-x-2 mt-4 md:mt-0">
          {tabsNavigationData.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center space-x-2
                ${activeTab === tab.id ? 'bg-primary text-primary-foreground shadow-sm' : 'bg-muted text-muted-foreground hover:bg-accent hover:text-foreground'}`
              }
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Global Error Display */}
      {error && (
        <div className="bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 p-4 rounded-lg mb-6 border border-red-400" role="alert">
          <p className="font-semibold mb-1">Error:</p>
          <p>{error}</p>
        </div>
      )}

      {/* Main Content Area for Tabs */}
      <div className="flex-1 bg-card rounded-lg shadow-lg p-6 border border-border">
        {/* Overview Tab Content */}
        {activeTab === 'overview' && (
          <div className="space-y-8">
            <h3 className="text-2xl font-semibold text-foreground mb-4">System Overview & Metrics</h3>
            {/* Statistics Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {statsCardsData.map((stat, index) => (
                <div key={index} className={`bg-gradient-to-br ${stat.color} text-white p-6 rounded-xl shadow-lg flex flex-col justify-between h-36`}>
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-sm opacity-90 font-medium">{stat.label}</p>
                    <stat.icon className="h-7 w-7 opacity-70" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold mb-1">
                      {isLoadingStats ? (
                        <Loader2 className="h-8 w-8 animate-spin text-white opacity-80" />
                      ) : (
                        stat.value
                      )}
                    </p>
                    {/* Optional: Display change if applicable */}
                    {/* <p className="text-xs opacity-80">{stat.change} since last month</p> */}
                  </div>
                </div>
              ))}
            </div>

            {/* Backend Health Section */}
            <div className="bg-muted/30 p-6 rounded-lg border border-border">
              <h3 className="text-xl font-semibold text-foreground mb-4 flex items-center">
                <Shield className="h-6 w-6 mr-3 text-pyrotech-500" />
                Backend Service Health Status
              </h3>
              {isLoadingStats ? (
                <div className="flex items-center justify-center h-24">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  <span className="ml-3 text-muted-foreground">Loading backend health...</span>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center justify-between p-2 bg-background rounded-md border border-input">
                    <strong>RAG System:</strong>
                    <span className={`font-medium ${backendHealth.rag_system === 'Ready' ? 'text-green-600' : 'text-red-600'}`}>
                      {backendHealth.rag_system || 'N/A'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-background rounded-md border border-input">
                    <strong>ChromaDB Path:</strong>
                    <span className="font-medium">{backendHealth.chroma_db_path || 'N/A'}</span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-background rounded-md border border-input">
                    <strong>Ollama API URL:</strong>
                    <span className="font-medium">{backendHealth.ollama_api_url || 'N/A'}</span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-background rounded-md border border-input">
                    <strong>Ollama Embedding Model:</strong>
                    <span className="font-medium">{backendHealth.ollama_embedding_model || 'N/A'}</span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-background rounded-md border border-input">
                    <strong>Reranker Model:</strong>
                    <span className="font-medium">{backendHealth.reranker_model || 'N/A'}</span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-background rounded-md border border-input">
                    <strong>Chroma Document Count:</strong>
                    <span className="font-medium">
                      {backendHealth.chroma_document_count !== undefined ? backendHealth.chroma_document_count.toLocaleString() : 'N/A'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-background rounded-md border border-input">
                    <strong>Redis Status:</strong>
                    <span className={`font-medium ${backendHealth.redis_status === 'Connected' ? 'text-green-600' : 'text-red-600'}`}>
                      {backendHealth.redis_status || 'N/A'}
                    </span>
                  </div>
                  {backendHealth.initialization_errors && backendHealth.initialization_errors.length > 0 && (
                    <div className="md:col-span-2 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 p-3 rounded-lg border border-red-400">
                      <strong className="flex items-center mb-2"><Zap className="h-4 w-4 mr-2" />Initialization Errors:</strong>
                      <ul className="list-disc list-inside ml-4 space-y-1">
                        {backendHealth.initialization_errors.map((err, i) => (
                          <li key={i}>{err}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Users Tab Content */}
        {activeTab === 'users' && (
          <div className="space-y-6">
            <h3 className="text-2xl font-semibold text-foreground mb-4">User Management</h3>
            <p className="text-muted-foreground">View and manage registered user accounts.</p>
            {isLoadingUsers ? (
              <div className="flex items-center justify-center h-32 text-muted-foreground">
                <Loader2 className="h-6 w-6 animate-spin mr-3" />
                <span>Loading user data...</span>
              </div>
            ) : users.length > 0 ? (
              <div className="overflow-x-auto rounded-lg border border-border shadow-sm">
                <table className="min-w-full divide-y divide-border">
                  <thead className="bg-muted/50">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Username
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Role
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-card divide-y divide-border">
                    {users.map((userItem, index) => (
                      <tr key={index} className="hover:bg-muted/20 transition-colors duration-150">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-foreground">
                          {userItem.username}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-muted-foreground capitalize">
                          {userItem.role}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium space-x-2">
                          {/* Placeholder for user-specific actions like edit or delete */}
                          <button className="text-blue-600 hover:text-blue-800 transition-colors duration-150">Edit</button>
                          <button className="text-red-600 hover:text-red-800 transition-colors duration-150">Delete</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-8">No user accounts found.</p>
            )}
          </div>
        )}

        {/* Documents Tab Content */}
        {activeTab === 'documents' && (
          <div className="space-y-6">
            <h3 className="text-2xl font-semibold text-foreground mb-4">Document Management</h3>
            <p className="text-muted-foreground">
              Here you can manage the documents that are processed and indexed into the RAG knowledge base.
              Use the buttons below to trigger ingestion of new documents or clear existing ones.
            </p>
            <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4 mt-6">
              <button
                onClick={triggerDocumentIngestion}
                className="flex items-center justify-center bg-secondary text-secondary-foreground px-6 py-3 rounded-lg hover:bg-secondary/80 transition-colors duration-200 font-medium shadow-md"
                disabled={isLoadingStats} // Disable button while any stats or actions are loading.
              >
                {isLoadingStats ? (
                  <Loader2 className="h-5 w-5 animate-spin inline-block mr-3" />
                ) : (
                  <FileText className="h-5 w-5 inline-block mr-3" />
                )}
                Trigger Document Ingestion
              </button>
              <button
                onClick={clearAllDocuments}
                className="flex items-center justify-center bg-destructive text-destructive-foreground px-6 py-3 rounded-lg hover:bg-destructive/80 transition-colors duration-200 font-medium shadow-md"
                disabled={isLoadingStats} // Disable button while any stats or actions are loading.
              >
                {isLoadingStats ? (
                  <Loader2 className="h-5 w-5 animate-spin inline-block mr-3" />
                ) : (
                  <Trash2 className="h-5 w-5 inline-block mr-3" />
                )}
                Clear All Documents
              </button>
            </div>
          </div>
        )}

        {/* Settings Tab Content */}
        {activeTab === 'settings' && (
          <div className="space-y-6">
            <h3 className="text-2xl font-semibold text-foreground mb-4">System Settings</h3>
            <p className="text-muted-foreground">
              This section is reserved for future system-wide configuration options.
              Currently, there are no configurable settings available here.
            </p>
            {/* Placeholder for future settings forms or controls */}
            <div className="bg-muted/20 p-6 rounded-lg border border-border text-muted-foreground">
              <p>Configuration options such as API keys, model defaults, or integration settings would be implemented here.</p>
              <p className="mt-2 text-sm">Coming soon in future updates!</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminPanel;
