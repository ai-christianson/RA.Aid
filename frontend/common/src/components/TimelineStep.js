"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TimelineStep = void 0;
var react_1 = require("react");
var collapsible_1 = require("./ui/collapsible");
var TimelineStep = function (_a) {
    var step = _a.step;
    // Get status color
    var getStatusColor = function (status) {
        switch (status) {
            case 'completed':
                return 'bg-green-500';
            case 'in-progress':
                return 'bg-blue-500';
            case 'error':
                return 'bg-red-500';
            case 'pending':
                return 'bg-yellow-500';
            default:
                return 'bg-gray-500';
        }
    };
    // Get icon based on step type
    var getTypeIcon = function (type) {
        switch (type) {
            case 'tool-execution':
                return '🛠️';
            case 'thinking':
                return '💭';
            case 'planning':
                return '📝';
            case 'implementation':
                return '💻';
            case 'user-input':
                return '👤';
            default:
                return '▶️';
        }
    };
    // Format timestamp
    var formatTime = function (timestamp) {
        return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };
    return (<collapsible_1.Collapsible className="w-full mb-4 border border-border rounded-md overflow-hidden transition-all duration-200">
      <collapsible_1.CollapsibleTrigger className="w-full flex items-center justify-between p-3 text-left hover:bg-accent/50 cursor-pointer">
        <div className="flex items-center">
          <div className={"w-3 h-3 rounded-full ".concat(getStatusColor(step.status), " mr-3")}/>
          <div className="mr-2">{getTypeIcon(step.type)}</div>
          <div>
            <div className="font-medium">{step.title}</div>
            <div className="text-sm text-muted-foreground truncate max-w-xs">
              {step.type === 'tool-execution' ? 'Run tool' : step.content.substring(0, 60)}
              {step.content.length > 60 ? '...' : ''}
            </div>
          </div>
        </div>
        <div className="text-xs text-muted-foreground flex flex-col items-end">
          <span>{formatTime(step.timestamp)}</span>
          {step.duration && (<span className="mt-1">{(step.duration / 1000).toFixed(1)}s</span>)}
        </div>
      </collapsible_1.CollapsibleTrigger>
      <collapsible_1.CollapsibleContent>
        <div className="p-4 bg-card border-t border-border">
          <div className="text-sm whitespace-pre-wrap">
            {step.content}
          </div>
          {step.duration && (<div className="mt-3 pt-3 border-t border-border">
              <div className="text-xs text-muted-foreground">
                Duration: {(step.duration / 1000).toFixed(1)} seconds
              </div>
            </div>)}
        </div>
      </collapsible_1.CollapsibleContent>
    </collapsible_1.Collapsible>);
};
exports.TimelineStep = TimelineStep;
