"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SessionSidebar = void 0;
var react_1 = require("react");
var scroll_area_1 = require("./ui/scroll-area");
var sample_data_1 = require("../utils/sample-data");
var SessionSidebar = function (_a) {
    var onSelectSession = _a.onSelectSession, currentSessionId = _a.currentSessionId, _b = _a.sessions, sessions = _b === void 0 ? (0, sample_data_1.getSampleAgentSessions)() : _b, _c = _a.className, className = _c === void 0 ? '' : _c;
    // Get status color
    var getStatusColor = function (status) {
        switch (status) {
            case 'active':
                return 'bg-blue-500';
            case 'completed':
                return 'bg-green-500';
            case 'error':
                return 'bg-red-500';
            default:
                return 'bg-gray-500';
        }
    };
    // Format timestamp
    var formatDate = function (date) {
        return date.toLocaleDateString([], {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };
    return (<div className={"hidden md:block w-[250px] lg:w-[300px] h-screen border-r border-border ".concat(className)}>
      <div className="p-4 border-b border-border">
        <h3 className="font-medium text-lg">Sessions</h3>
      </div>
      <scroll_area_1.ScrollArea className="h-[calc(100vh-5rem)]">
        <div className="p-4 space-y-4">
          {sessions.map(function (session) { return (<button key={session.id} onClick={function () { return onSelectSession === null || onSelectSession === void 0 ? void 0 : onSelectSession(session.id); }} className={"w-full flex items-start p-3 text-left rounded-md transition-colors hover:bg-accent/50 ".concat(currentSessionId === session.id ? 'bg-accent' : '')}>
              <div className={"w-3 h-3 rounded-full ".concat(getStatusColor(session.status), " mt-1.5 mr-3 flex-shrink-0")}/>
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{session.name}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {session.steps.length} steps • {formatDate(session.updated)}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  <span className="capitalize">{session.status}</span>
                </div>
              </div>
            </button>); })}
        </div>
      </scroll_area_1.ScrollArea>
    </div>);
};
exports.SessionSidebar = SessionSidebar;
