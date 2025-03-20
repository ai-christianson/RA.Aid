import React, { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Send, X } from "lucide-react";
import { cn } from "../utils";
import { useSessionStore } from "../store";

interface InputSectionProps {
  sessionId?: number;
  onSubmit?: (message: string) => void;
  className?: string;
  isDrawerOpen?: boolean; // Prop to check if drawer is open
  isNewSession?: boolean; // Prop to indicate if this is for a new session
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
    submitNewSession,
    cancelNewSession
  } = useSessionStore();
  
  // Sync local message state with newSession message when in new session mode
  useEffect(() => {
    if (isNewSession && newSession) {
      setMessage(newSession.message);
    }
  }, [isNewSession, newSession]);
  
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
        // Default to regular mode (not research-only)
        await submitNewSession(false);
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

  // Handle research-only submissions
  const handleResearchOnlySubmit = async (e: React.MouseEvent) => {
    e.preventDefault();
    
    if (!message.trim() || !isNewSession || !newSession) return;
    
    setIsSubmitting(true);
    
    try {
      updateNewSessionMessage(message);
      // Set research-only mode to true
      await submitNewSession(true);
    } catch (error) {
      console.error("Error submitting research-only message:", error);
    }
  };

  // New session can have different placeholder text and actions
  const placeholder = isNewSession 
    ? "What would you like help with today?" 
    : "Type your message...";
  
  return (
    <div className={cn(
      "fixed bottom-0 left-0 right-0 z-20 pointer-events-none md:left-[280px] lg:left-[320px] xl:left-[350px]", 
      className
    )}>
      <div className="px-4 pb-4 pointer-events-none">
        <div className="relative rounded-lg border border-input bg-background/95 backdrop-blur-sm shadow-md pointer-events-auto">
          {isNewSession && (
            <div className="flex justify-between items-center pt-2 px-3">
              <div className="text-sm font-medium">Create New Session</div>
              {newSession?.error && (
                <div className="text-xs text-destructive max-w-[70%] truncate">{newSession.error}</div>
              )}
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={cancelNewSession}
                className="h-7 w-7 p-0 flex-shrink-0"
                disabled={isSubmitting}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}
          <form onSubmit={handleSubmit}>
            <textarea
              value={message}
              onChange={(e) => {
                setMessage(e.target.value);
                // If in new session mode, also update store state
                if (isNewSession) {
                  updateNewSessionMessage(e.target.value);
                }
              }}
              placeholder={placeholder}
              className="flex w-full resize-none rounded-lg border-0 bg-transparent px-3 py-2 pr-12 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50"
              rows={isNewSession ? 4 : 3}
              disabled={isSubmitting}
            />
            {isNewSession ? (
              <div className="flex items-center space-x-2 px-3 py-2 border-t border-border/30">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleResearchOnlySubmit}
                  disabled={!message.trim() || isSubmitting}
                  className="text-xs h-8"
                >
                  {isSubmitting && newSession?.isSubmitting ? (
                    <span className="flex items-center">
                      <span className="h-3 w-3 mr-1 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      <span className="hidden sm:inline">Processing...</span>
                      <span className="sm:hidden">Processing</span>
                    </span>
                  ) : (
                    <>
                      <span className="hidden sm:inline">Research Only</span>
                      <span className="sm:hidden">Research</span>
                    </>
                  )}
                </Button>
                <div className="flex-1"></div>
                <Button 
                  type="submit" 
                  disabled={!message.trim() || isSubmitting} 
                  variant="default"
                  size="sm"
                  className="text-xs h-8"
                >
                  {isSubmitting && newSession?.isSubmitting ? (
                    <span className="flex items-center">
                      <span className="h-3 w-3 mr-1 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      <span className="hidden sm:inline">Creating...</span>
                      <span className="sm:hidden">Creating</span>
                    </span>
                  ) : (
                    <>
                      <span className="hidden sm:inline">Create Session</span>
                      <span className="sm:hidden">Create</span>
                    </>
                  )}
                </Button>
              </div>
            ) : (
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
            )}
          </form>
        </div>
      </div>
    </div>
  );
};