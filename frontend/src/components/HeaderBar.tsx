import React from 'react';
// Importing icons for various header functionalities.
import { Menu, Sun, Moon, Settings, LogOut, MessageSquare, Shield } from 'lucide-react';

/**
 * HeaderBar Component
 * Displays the application's top navigation bar. It includes:
 * - A toggle button for the sidebar.
 * - Application branding (logo and title).
 * - Navigation links between 'Chat' and 'Admin' views (admin view only for admin users).
 * - User authentication status (logged-in username and role).
 * - Buttons for toggling dark/light mode, general settings, and logout.
 *
 * @param {object} props - The properties passed to the component.
 * @param {boolean} props.darkMode - Current state of dark mode.
 * @param {Function} props.onToggleDarkMode - Callback to toggle dark mode.
 * @param {boolean} props.sidebarOpen - Current state of sidebar visibility.
 * @param {Function} props.onToggleSidebar - Callback to toggle sidebar visibility.
 * @param {string} props.currentView - The currently active main view ('chat' or 'admin').
 * @param {Function} props.onViewChange - Callback to change the main application view.
 * @param {Function} props.onLogout - Callback to handle user logout.
 * @param {object | null} props.user - The authenticated user object (null if not logged in).
 * Expected structure: { username: string, role: string }
 */
const HeaderBar = ({
  darkMode,
  onToggleDarkMode,
  sidebarOpen,
  onToggleSidebar,
  currentView,
  onViewChange, // Function to change the main view.
  onLogout,
  user // The authenticated user object.
}) => {
  // --- Render Logic ---
  return (
    // Main header container. Fixed height, border, flex layout for content alignment.
    // `relative z-10` ensures it sits above other content.
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-4 lg:px-6 relative z-10 shadow-sm">
      {/* Left Section: Sidebar Toggle and App Branding */}
      <div className="flex items-center space-x-4">
        {/* Sidebar Toggle Button */}
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg hover:bg-accent transition-colors duration-200"
          aria-label="Toggle sidebar visibility"
          title="Toggle Sidebar"
        >
          <Menu className="h-5 w-5 text-muted-foreground" />
        </button>
        
        {/* Application Branding (Logo and Title) */}
        <div className="flex items-center space-x-3">
          {/* Logo/Icon */}
          <div className="w-8 h-8 bg-gradient-to-br from-pyrotech-500 to-pyrotech-600 rounded-lg flex items-center justify-center shadow-md">
            <span className="text-white font-bold text-sm">P</span>
          </div>
          {/* App Name and Tagline */}
          <div>
            <h1 className="text-lg font-semibold text-foreground">Pyrotech AI</h1>
            <p className="text-xs text-muted-foreground">Document Assistant</p>
          </div>
        </div>
      </div>

      {/* Center Section: Main Navigation Tabs */}
      {/* Hidden on small screens, displayed as a flexible row on medium and larger screens. */}
      <div className="hidden md:flex items-center space-x-1 bg-muted/50 rounded-lg p-1 shadow-inner">
        {/* Chat View Button */}
        <button
          onClick={() => onViewChange('chat')} // Call onViewChange to switch to 'chat' view.
          className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
            ${currentView === 'chat'
              ? 'bg-background text-primary shadow-sm' // Active state styling.
              : 'text-muted-foreground hover:text-foreground hover:bg-background/50' // Inactive state styling.
            }`}
          aria-label="Switch to Chat view"
          title="Go to Chat"
        >
          <MessageSquare className="h-4 w-4" />
          <span>Chat</span>
        </button>
        {/* Admin View Button (Conditionally rendered only if user has 'admin' role) */}
        {user?.role === 'admin' && (
          <button
            onClick={() => onViewChange('admin')} // Call onViewChange to switch to 'admin' view.
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
              ${currentView === 'admin'
                ? 'bg-background text-primary shadow-sm' // Active state styling.
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50' // Inactive state styling.
              }`}
            aria-label="Switch to Admin view"
            title="Go to Admin Panel"
          >
            <Shield className="h-4 w-4" />
            <span>Admin</span>
          </button>
        )}
      </div>

      {/* Right Section: User Info and Utility Buttons */}
      <div className="flex items-center space-x-3">
        {/* User Information Display (visible only if a user is logged in) */}
        {user && (
          <span className="text-sm text-muted-foreground mr-2 hidden sm:inline-block"> {/* Hidden on extra small screens */}
            Logged in as: <span className="font-semibold text-foreground">{user.username}</span> (<span className="capitalize">{user.role}</span>)
          </span>
        )}
        {/* Dark Mode Toggle Button */}
        <button
          onClick={onToggleDarkMode}
          className="p-2 rounded-lg hover:bg-accent transition-colors duration-200"
          aria-label="Toggle dark mode"
          title="Toggle Dark/Light Mode"
        >
          {darkMode ? (
            <Sun className="h-5 w-5 text-muted-foreground" /> // Sun icon for dark mode.
          ) : (
            <Moon className="h-5 w-5 text-muted-foreground" /> // Moon icon for light mode.
          )}
        </button>
        
        {/* Settings Button (Placeholder) */}
        <button
          className="p-2 rounded-lg hover:bg-accent transition-colors duration-200"
          aria-label="Application settings"
          title="Settings"
        >
          <Settings className="h-5 w-5 text-muted-foreground" />
        </button>
        
        {/* Logout Button */}
        <button
          onClick={onLogout}
          className="p-2 rounded-lg hover:bg-accent transition-colors duration-200"
          aria-label="Logout from account"
          title="Logout"
        >
          <LogOut className="h-5 w-5 text-muted-foreground" />
        </button>
      </div>
    </header>
  );
};

export default HeaderBar;
