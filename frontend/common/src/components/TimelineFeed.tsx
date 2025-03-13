import React, { useState } from 'react';
import { ScrollArea } from './ui/scroll-area';
import { TimelineStep } from './TimelineStep';
import { AgentStep } from '../utils/sample-data';

interface TimelineFeedProps {
  steps: AgentStep[];
  maxHeight?: string;
  filter?: {
    types?: string[];
    status?: string[];
  };
  sortOrder?: 'asc' | 'desc';
}

export const TimelineFeed: React.FC<TimelineFeedProps> = ({
  steps,
  maxHeight = '500px',
  filter,
  sortOrder = 'desc'
}) => {
  // State for filtered and sorted steps
  const [activeFilter, setActiveFilter] = useState(filter);
  const [activeSortOrder, setActiveSortOrder] = useState<'asc' | 'desc'>(sortOrder);

  // Apply filters and sorting
  const filteredSteps = steps.filter(step => {
    if (!activeFilter) return true;
    
    const typeMatch = !activeFilter.types || activeFilter.types.length === 0 || 
                      activeFilter.types.includes(step.type);
    
    const statusMatch = !activeFilter.status || activeFilter.status.length === 0 || 
                        activeFilter.status.includes(step.status);
    
    return typeMatch && statusMatch;
  });

  // Sort steps
  const sortedSteps = [...filteredSteps].sort((a, b) => {
    if (activeSortOrder === 'asc') {
      return a.timestamp.getTime() - b.timestamp.getTime();
    } else {
      return b.timestamp.getTime() - a.timestamp.getTime();
    }
  });

  // Toggle sort order
  const toggleSortOrder = () => {
    setActiveSortOrder(prevOrder => prevOrder === 'asc' ? 'desc' : 'asc');
  };

  // Filter by type
  const filterTypes = [
    'all',
    'tool-execution',
    'thinking',
    'planning',
    'implementation',
    'user-input'
  ];

  const handleFilterChange = (type: string) => {
    if (type === 'all') {
      setActiveFilter({
        ...activeFilter,
        types: []
      });
    } else {
      setActiveFilter({
        ...activeFilter,
        types: [type]
      });
    }
  };

  return (
    <div className="w-full border border-border rounded-md bg-background">
      <div className="p-3 border-b border-border">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-medium">Timeline Feed</h3>
          <button 
            onClick={toggleSortOrder}
            className="text-xs bg-secondary hover:bg-secondary/80 text-secondary-foreground px-2 py-1 rounded"
          >
            {activeSortOrder === 'asc' ? '⬆️ Oldest first' : '⬇️ Newest first'}
          </button>
        </div>
        
        <div className="flex gap-2 overflow-x-auto pb-2 text-xs">
          {filterTypes.map(type => (
            <button
              key={type}
              onClick={() => handleFilterChange(type)}
              className={`px-2 py-1 rounded whitespace-nowrap ${
                type === 'all' && (!activeFilter?.types || activeFilter.types.length === 0) || 
                activeFilter?.types?.includes(type)
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary/50 text-secondary-foreground hover:bg-secondary/80'
              }`}
            >
              {type === 'all' ? 'All types' : type}
            </button>
          ))}
        </div>
      </div>
      
      <ScrollArea className="h-full" style={{ maxHeight }}>
        <div className="p-3">
          {sortedSteps.length > 0 ? (
            sortedSteps.map((step) => (
              <TimelineStep key={step.id} step={step} />
            ))
          ) : (
            <div className="text-center text-muted-foreground py-8">
              No steps to display
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};