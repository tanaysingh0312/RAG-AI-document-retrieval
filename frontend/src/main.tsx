import { createRoot } from 'react-dom/client'; // Import createRoot for React 18+ rendering.
import App from './App.jsx'; // Import the main App component, which is the root of our application.
import './index.css'; // Import global CSS styles, typically including Tailwind CSS directives.

/**
 * Main entry point of the React application.
 * This file is responsible for mounting the React application to the DOM.
 */

// Get the DOM element with the ID 'root'. This is where our React app will be injected.
const rootElement = document.getElementById("root");

// Check if the root element exists to prevent errors if the HTML structure is missing it.
if (rootElement) {
  // Create a React root. This is the modern way to render React applications in React 18+.
  // It enables concurrent features and better performance.
  const root = createRoot(rootElement);

  // Render the main App component into the created React root.
  // The <App /> component and its children will be managed by React from this point.
  root.render(<App />);
} else {
  // Log an error if the root element is not found, indicating a problem with the HTML structure.
  console.error("Error: Root element with ID 'root' not found in the document. React application cannot be mounted.");
}
