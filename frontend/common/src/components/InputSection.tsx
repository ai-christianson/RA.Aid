
import React, { useState, useEffect, useRef } from "react";
import { Button } from "./ui/button";
import { Send, X } from "lucide-react";
import { cn } from "../utils";
import { useSessionStore } from "../store";
import { EnterKeySvg } from './ui/EnterKeySvg';

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
  const textareaRef = useRef<HTMLTextAreaElement>(null); // Ref for the textarea

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

  // Autofocus textarea when isNewSession becomes true
  useEffect(() => {
    if (isNewSession && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isNewSession]); // Dependency: only run when isNewSession changes

  // If no sessionId is provided and not in new session mode, or drawer is open on mobile, don't render
  // Also, don't render if we're in new session mode but the newSession state is submitting
  // Adjusted logic: Should render if isNewSession is true, regardless of sessionId
  if ((!isNewSession && !sessionId) || (isDrawerOpen && isMobile) || (isNewSession && newSession?.isSubmitting)) {
      // Exception: if it *is* a new session, we *should* render it.
      if (!isNewSession) {
        return null;
      }
  }


  const handleSubmit = async (e?: React.FormEvent) => { // Made event optional for direct calls
    if (e) e.preventDefault(); // Prevent default only if event is passed

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
  const handleResearchOnlySubmit = async (e?: React.MouseEvent | React.KeyboardEvent) => { // Accept KeyboardEvent too
    if (e) e.preventDefault();

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

  // --- Ctrl+Enter Shortcut Handler ---
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Check for Ctrl+Shift+Enter for research-only submission
    if (event.ctrlKey && event.shiftKey && event.code === 'Enter') {
      if (!message.trim() || isSubmitting || !isNewSession) return; // Prevent empty/duplicate submission/outside new session
      event.preventDefault(); // Prevent default textarea behavior (newline)
      handleResearchOnlySubmit(event as any); // Trigger the research-only action
      return; // Stop further processing in this handler
    }

    // Check for Ctrl+Enter for regular submission
    if (event.ctrlKey && event.code === 'Enter') { // Use event.code for reliability
      event.preventDefault(); // Prevent default textarea behavior (e.g., new line)
      // Trigger submit only if not already submitting and message has content
      if (!isSubmitting && message.trim()) {
        handleSubmit(event as any); // Pass event as required by task description
      }
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
              ref={textareaRef} // Attach the ref here
              value={message}
              onChange={(e) => {
                setMessage(e.target.value);
                // If in new session mode, also update store state
                if (isNewSession) {
                  updateNewSessionMessage(e.target.value);
                }
              }}
              onKeyDown={handleKeyDown} // <-- Add keydown handler
              placeholder={placeholder}
              className="flex w-full resize-none rounded-lg border-0 bg-transparent px-3 py-2 pr-12 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50"
              rows={isNewSession ? 4 : 3}
              disabled={isSubmitting}
            />
            {isNewSession ? (
              <div className="flex items-center justify-end space-x-2 px-3 py-2 border-t border-border/30">{/* Changed justify-end */}
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
                    <span className="flex items-center justify-center">
                      <span className="hidden sm:inline -mt-0.5">Research Only</span>
                      <span className="sm:hidden -mt-0.5">Research</span>
                      <span className="flex items-center justify-around ml-4 border rounded bg-muted/50 text-muted-foreground">
                        <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 font-mono text-xs">
                          <strong>Ctrl</strong>
                        </kbd>
                        <span className="mx-0 -mt-1">+</span>
                         <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 font-mono text-xs">
                          <strong>Shift</strong>
                        </kbd>
                        <span className="mx-0 -mt-1">+</span>
                        <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 font-mono text-xs">
                          <EnterKeySvg className="h-3 w-4" />
                        </kbd>
                      </span>
                    </span>
                  )}
                </Button>
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
                    <span className="flex items-center justify-center">
                      <span className="hidden sm:inline -mt-0.5">Create Session</span>
                      <span className="sm:hidden -mt-0.5">Create</span>
                      <span className="flex items-center justify-around ml-4 border rounded bg-muted/15 text-gray-900">
                        <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 font-mono text-xs">
                          <strong>Ctrl</strong>
                        </kbd>
                        <span className="mx-0 -mt-1">+</span>
                        <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 font-mono text-xs">
                          <EnterKeySvg className="h-3 w-4" />
                        </kbd>
                      </span>
                    </span>
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
