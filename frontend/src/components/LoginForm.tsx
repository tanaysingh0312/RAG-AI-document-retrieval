import React, { useState } from 'react';
// Importing icons for input fields and theme toggle.
import { Mail, Lock, Eye, EyeOff, Sun, Moon } from 'lucide-react';

/**
 * LoginForm Component
 * Provides the user authentication interface, allowing users to sign in
 * with a username and password. It handles form submission, API calls
 * for authentication, token storage, and displays loading/error states.
 *
 * @param {object} props - The properties passed to the component.
 * @param {Function} props.onLogin - Callback function executed upon successful login,
 * passing the authenticated user object to the parent component.
 * @param {boolean} props.darkMode - Current state of dark mode.
 * @param {Function} props.onToggleDarkMode - Callback to toggle dark mode.
 */
const LoginForm = ({ onLogin, darkMode, onToggleDarkMode }) => {
  // State to manage the form input values (username and password).
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  // State to toggle the visibility of the password input field.
  const [showPassword, setShowPassword] = useState(false);
  // State to indicate if the login process is currently in progress.
  const [isLoading, setIsLoading] = useState(false);
  // State to store and display any error messages from the login attempt.
  const [error, setError] = useState('');

  // Base URL for API calls, dynamically determined.
  const API_BASE_URL = `${window.location.origin}/api`;

  /**
   * Handles the form submission event.
   * Prevents default form submission, initiates the login API call,
   * handles success/failure, and updates loading/error states.
   * @param {object} e - The form submission event object.
   */
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent the default browser form submission.
    setError(''); // Clear any previous error messages.
    setIsLoading(true); // Set loading state to true.

    try {
      // Make an API call to the backend's token endpoint for authentication.
      // The 'Content-Type' is 'application/x-www-form-urlencoded' as expected by FastAPI's OAuth2PasswordRequestForm.
      const response = await fetch(`${API_BASE_URL}/token`, {
        method: "POST", // Use POST method for authentication.
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        // Convert formData to URL-encoded format for the request body.
        body: new URLSearchParams(formData).toString()
      });

      if (!response.ok) {
        // If the response is not OK (e.g., 400, 401), parse the error details.
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Login failed. Please check your credentials.');
      }

      const data = await response.json(); // Parse the successful response (contains access_token).
      localStorage.setItem("token", data.access_token); // Store the JWT token securely in localStorage.

      // Decode the JWT token to extract user information (username and role).
      // The JWT is typically in the format: header.payload.signature.
      const tokenParts = data.access_token.split('.');
      if (tokenParts.length !== 3) {
        throw new Error("Received invalid token format from server.");
      }
      const decodedPayload = JSON.parse(atob(tokenParts[1])); // Base64 decode the payload part.

      // Construct the user object from the decoded payload.
      const authenticatedUser = {
        username: decodedPayload.sub, // 'sub' (subject) field usually holds the username.
        role: decodedPayload.role || 'user', // 'role' field holds the user's role, default to 'user'.
      };
      
      onLogin(authenticatedUser); // Call the parent's onLogin callback with the authenticated user data.
    } catch (err) {
      console.error("Login process error:", err);
      setError(err.message || 'An unexpected error occurred during login. Please try again later.');
    } finally {
      setIsLoading(false); // Reset loading state regardless of success or failure.
    }
  };

  /**
   * Handles changes in the form input fields (username and password).
   * Updates the `formData` state as the user types.
   * @param {object} e - The input change event object.
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value // Update the specific field in formData.
    }));
  };

  // --- Render Logic ---
  return (
    // Main container for the login form, centered and styled as a card.
    <div className="w-full max-w-md mx-auto p-4">
      {/* Dark Mode Toggle Button - positioned at the top right of the card. */}
      <div className="flex justify-end mb-6">
        <button
          onClick={onToggleDarkMode}
          className="p-2 rounded-lg hover:bg-white/10 transition-colors duration-200"
          aria-label="Toggle dark mode"
          title="Toggle Dark/Light Mode"
        >
          {darkMode ? (
            <Sun className="h-5 w-5 text-gray-400" /> // Sun icon for dark mode.
          ) : (
            <Moon className="h-5 w-5 text-gray-600" /> // Moon icon for light mode.
          )}
        </button>
      </div>

      {/* Login Card Container */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8 border border-gray-200 dark:border-gray-700">
        {/* Header Section of the Login Form */}
        <div className="text-center mb-8">
          {/* Application Logo/Icon */}
          <div className="w-16 h-16 bg-gradient-to-br from-pyrotech-500 to-pyrotech-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
            <span className="text-white font-bold text-2xl">P</span>
          </div>
          {/* Welcome Message */}
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome Back to Pyrotech AI
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-base">
            Sign in to your account to access your document assistant.
          </p>
        </div>

        {/* Error Message Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative mb-4 text-sm font-medium" role="alert">
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Username Input Field */}
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Username (e.g., admin or user)
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Mail className="h-5 w-5 text-gray-400" /> {/* Mail icon */}
              </div>
              <input
                id="username"
                name="username"
                type="text" // Type is 'text' as username might not be email.
                required // Mark as required.
                value={formData.username} // Controlled component value.
                onChange={handleInputChange} // Handle input changes.
                className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-pyrotech-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-all duration-200 shadow-sm"
                placeholder="Enter your username"
                disabled={isLoading} // Disable input during loading.
                aria-label="Username input field"
              />
            </div>
          </div>

          {/* Password Input Field */}
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-gray-400" /> {/* Lock icon */}
              </div>
              <input
                id="password"
                name="password"
                type={showPassword ? 'text' : 'password'} // Toggle type for password visibility.
                required // Mark as required.
                value={formData.password} // Controlled component value.
                onChange={handleInputChange} // Handle input changes.
                className="w-full pl-10 pr-12 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-pyrotech-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-all duration-200 shadow-sm"
                placeholder="Enter your password"
                disabled={isLoading} // Disable input during loading.
                aria-label="Password input field"
              />
              {/* Toggle Password Visibility Button */}
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors duration-200"
                disabled={isLoading} // Disable button during loading.
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? (
                  <EyeOff className="h-5 w-5" /> // EyeOff icon when password is visible.
                ) : (
                  <Eye className="h-5 w-5" /> // Eye icon when password is hidden.
                )}
              </button>
            </div>
          </div>

          {/* Remember Me Checkbox and Forgot Password Link */}
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                className="h-4 w-4 text-pyrotech-500 focus:ring-pyrotech-500 border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600"
                disabled={isLoading} // Disable checkbox during loading.
              />
              <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                Remember me
              </label>
            </div>
            <button
              type="button"
              className="text-sm text-pyrotech-500 hover:text-pyrotech-600 font-medium transition-colors duration-200"
              disabled={isLoading} // Disable button during loading.
            >
              Forgot password?
            </button>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading} // Disable button during loading.
            className="w-full bg-gradient-to-r from-pyrotech-500 to-pyrotech-600 text-white py-3 px-4 rounded-lg font-medium hover:from-pyrotech-600 hover:to-pyrotech-700 focus:outline-none focus:ring-2 focus:ring-pyrotech-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-[1.02] shadow-lg"
            aria-label="Sign In button"
          >
            {isLoading ? (
              // Display a loading spinner and text when loading.
              <div className="flex items-center justify-center space-x-2">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Signing in...</span>
              </div>
            ) : (
              'Sign In' // Default button text.
            )}
          </button>
        </form>

        {/* Footer Section - Sign Up Link */}
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Don't have an account?{' '}
            <button
              type="button"
              className="text-pyrotech-500 hover:text-pyrotech-600 font-medium transition-colors duration-200"
              disabled={isLoading} // Disable button during loading.
            >
              Contact your administrator
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;
