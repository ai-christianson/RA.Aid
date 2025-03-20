import React, { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Send } from "lucide-react";
import { cn } from "../utils";

interface InputSectionProps {
  sessionId?: number;
  onSubmit?: (message: string) => void;
  className?: string;
  isDrawerOpen?: boolean; // New prop to check if drawer is open
}

export const InputSection: React.FC<InputSectionProps> = ({ 
  sessionId,
  onSubmit,
  className,
  isDrawerOpen = false // Default to false
}) => {
  const [message, setMessage] = useState("");
  const [isMobile, setIsMobile] = useState(false);
  
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
  
  // If no sessionId is provided, or drawer is open on mobile, don't render the component
  if (!sessionId || (isDrawerOpen && isMobile)) return null;
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (message.trim() && onSubmit) {
      onSubmit(message);
    }
    
    setMessage(""); // Clear the input after submission
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
            />
            <Button 
              type="submit" 
              disabled={!message.trim()} 
              variant="ghost"
              size="sm"
              className="absolute bottom-1.5 right-1.5 h-9 w-9 rounded-md hover:bg-accent hover:text-accent-foreground"
            >
              <Send className="h-5 w-5" />
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
};