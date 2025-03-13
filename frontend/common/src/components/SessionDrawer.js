"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SessionDrawer = void 0;
var react_1 = require("react");
var lucide_react_1 = require("lucide-react");
var sheet_1 = require("./ui/sheet");
var button_1 = require("./ui/button");
var scroll_area_1 = require("./ui/scroll-area");
var sample_data_1 = require("../utils/sample-data");
var SessionDrawer = function (_a) {
    var onSelectSession = _a.onSelectSession, currentSessionId = _a.currentSessionId, _b = _a.sessions, sessions = _b === void 0 ? (0, sample_data_1.getSampleAgentSessions)() : _b;
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
    return (<sheet_1.Sheet>
      <sheet_1.SheetTrigger asChild>
        <button_1.Button variant="ghost" size="icon" className="md:hidden">
          <lucide_react_1.Menu className="h-5 w-5"/>
          <span className="sr-only">Toggle navigation</span>
        </button_1.Button>
      </sheet_1.SheetTrigger>
      <sheet_1.SheetContent side="left" className="w-[85%] sm:max-w-md">
        <sheet_1.SheetHeader>
          <sheet_1.SheetTitle>Sessions</sheet_1.SheetTitle>
        </sheet_1.SheetHeader>
        <scroll_area_1.ScrollArea className="h-[calc(100vh-5rem)] mt-6">
          <div className="space-y-4">
            {sessions.map(function (session) { return (<sheet_1.SheetClose key={session.id} asChild>
                <button onClick={function () { return onSelectSession === null || onSelectSession === void 0 ? void 0 : onSelectSession(session.id); }} className={"w-full flex items-start p-3 text-left rounded-md transition-colors hover:bg-accent/50 ".concat(currentSessionId === session.id ? 'bg-accent' : '')}>
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
                </button>
              </sheet_1.SheetClose>); })}
          </div>
        </scroll_area_1.ScrollArea>
      </sheet_1.SheetContent>
    </sheet_1.Sheet>);
};
exports.SessionDrawer = SessionDrawer;
