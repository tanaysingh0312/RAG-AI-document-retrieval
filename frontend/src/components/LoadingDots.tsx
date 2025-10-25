import React from 'react';

/**
 * LoadingDots Component
 * Displays a simple animated "thinking" indicator with three pulsing dots.
 * This component is typically used to provide visual feedback when the AI
 * is processing a request and a response is pending.
 */
const LoadingDots = () => {
  // --- Render Logic ---
  return (
    // Container for the loading message and dots. Uses flexbox for horizontal alignment.
    <div className="flex items-center space-x-2"> {/* Increased space-x for better visual separation */}
      {/* Text indicating that the AI is thinking. */}
      <span className="text-muted-foreground text-sm font-medium">AI is thinking</span>
      {/* Container for the three pulsing dots. */}
      <div className="flex space-x-1">
        {/* Individual dot elements. */}
        {/* Each dot has a slight animation delay to create a sequential pulsing effect. */}
        <div className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-pulse-dot" style={{ animationDelay: '0ms' }}></div>
        <div className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-pulse-dot" style={{ animationDelay: '150ms' }}></div>
        <div className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-pulse-dot" style={{ animationDelay: '300ms' }}></div>
      </div>
    </div>
  );
};

export default LoadingDots;
