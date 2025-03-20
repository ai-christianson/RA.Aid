import React, { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Send } from "lucide-react";
import { cn } from "../utils";
import { useSessionStore } from "../store";

interface InputSectionProps {
  sessionId?: number;
  onSubmit?: (message: string) => void;
  className?: string;
  isDrawerOpen?: boolean; // Prop to check if drawer is open
  isNewSession?: boolean; // New prop to indicate if this is for a new session
}

export const InputSection: React.FC<InputSectionProps> = ({ 
  sessionId,
  onSubmit,
  className,
  isDrawerOpen = false, // Default to false
  isNewSession = false  // Default to false
}) => {
  const [message, setMessage] = useState("");
  const [isMobile, setIsMobile] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Get session store state and actions for new session handling
  const { 
    newSession, 
    updateNewSessionMessage,
    submitNewSession
  } = useSessionStore();
  
  // Check for mobile screen size on client-side only
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    // Initial check
    checkMobile();
    
    // Add event listener for resize
    window.addEventListener('resize', checkMobile);
    
    // Cleanup
    return () => window.removeEventListener('resize', checkMobile);
  }, []);
  
  // If no sessionId is provided and not in new session mode, or drawer is open on mobile, don't render
  // Also, don't render if we're in new session mode but the newSession state is submitting
  if ((!sessionId && !isNewSession) || 
      (isDrawerOpen && isMobile) || 
      (isNewSession && newSession?.isSubmitting)) return null;
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim()) return;
    
    setIsSubmitting(true);
    
    try {
      if (isNewSession && newSession) {
        // For new sessions, update the message in the store and submit
        updateNewSessionMessage(message);
        await submitNewSession();
        // No need to clear the input here as the component will be unmounted
        // when the newSession state is cleared in the store
      } else if (onSubmit) {
        // For existing sessions, use the provided callback
        await onSubmit(message);
        setMessage(""); // Clear the input after submission
      }
    } catch (error) {
      console.error("Error submitting message:", error);
      // Error handling is done in the store for new sessions
      // and by the parent component for existing sessions
    } finally {
      // Only reset isSubmitting for existing sessions
      // For new sessions, the submitting state is managed by the store
      if (!isNewSession) {
        setIsSubmitting(false);
      }
    }
  };
  
  return (
    <div className={cn("fixed bottom-0 left-0 right-0 z-10 pointer-events-none md:left-[280px] lg:left-[320px] xl:left-[350px]", className)}>
      <div className="px-4 pb-4 pointer-events-none">
        <div className="relative rounded-lg border border-input bg-background/95 backdrop-blur-sm shadow-md pointer-events-auto">
          <form onSubmit={handleSubmit}>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Type your message..."
              className="flex w-full resize-none rounded-lg border-0 bg-transparent px-3 py-2 pr-12 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50"
              rows={3}
              disabled={isSubmitting}
            />
            <Button 
              type="submit" 
              disabled={!message.trim() || isSubmitting} 
              variant="ghost"
              size="sm"
              className="absolute bottom-1.5 right-1.5 h-9 w-9 rounded-md hover:bg-accent hover:text-accent-foreground"
            >
              {isSubmitting ? (
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
};