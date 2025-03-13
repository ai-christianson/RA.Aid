"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.TimelineFeed = void 0;
var react_1 = require("react");
var scroll_area_1 = require("./ui/scroll-area");
var TimelineStep_1 = require("./TimelineStep");
var TimelineFeed = function (_a) {
    var steps = _a.steps, _b = _a.maxHeight, maxHeight = _b === void 0 ? '500px' : _b, filter = _a.filter, _c = _a.sortOrder, sortOrder = _c === void 0 ? 'desc' : _c;
    // State for filtered and sorted steps
    var _d = (0, react_1.useState)(filter), activeFilter = _d[0], setActiveFilter = _d[1];
    var _e = (0, react_1.useState)(sortOrder), activeSortOrder = _e[0], setActiveSortOrder = _e[1];
    // Apply filters and sorting
    var filteredSteps = steps.filter(function (step) {
        if (!activeFilter)
            return true;
        var typeMatch = !activeFilter.types || activeFilter.types.length === 0 ||
            activeFilter.types.includes(step.type);
        var statusMatch = !activeFilter.status || activeFilter.status.length === 0 ||
            activeFilter.status.includes(step.status);
        return typeMatch && statusMatch;
    });
    // Sort steps
    var sortedSteps = __spreadArray([], filteredSteps, true).sort(function (a, b) {
        if (activeSortOrder === 'asc') {
            return a.timestamp.getTime() - b.timestamp.getTime();
        }
        else {
            return b.timestamp.getTime() - a.timestamp.getTime();
        }
    });
    // Toggle sort order
    var toggleSortOrder = function () {
        setActiveSortOrder(function (prevOrder) { return prevOrder === 'asc' ? 'desc' : 'asc'; });
    };
    // Filter by type
    var filterTypes = [
        'all',
        'tool-execution',
        'thinking',
        'planning',
        'implementation',
        'user-input'
    ];
    var handleFilterChange = function (type) {
        if (type === 'all') {
            setActiveFilter(__assign(__assign({}, activeFilter), { types: [] }));
        }
        else {
            setActiveFilter(__assign(__assign({}, activeFilter), { types: [type] }));
        }
    };
    return (<div className="w-full border border-border rounded-md bg-background">
      <div className="p-3 border-b border-border">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-medium">Timeline Feed</h3>
          <button onClick={toggleSortOrder} className="text-xs bg-secondary hover:bg-secondary/80 text-secondary-foreground px-2 py-1 rounded">
            {activeSortOrder === 'asc' ? '⬆️ Oldest first' : '⬇️ Newest first'}
          </button>
        </div>
        
        <div className="flex gap-2 overflow-x-auto pb-2 text-xs">
          {filterTypes.map(function (type) {
            var _a;
            return (<button key={type} onClick={function () { return handleFilterChange(type); }} className={"px-2 py-1 rounded whitespace-nowrap ".concat(type === 'all' && (!(activeFilter === null || activeFilter === void 0 ? void 0 : activeFilter.types) || activeFilter.types.length === 0) ||
                    ((_a = activeFilter === null || activeFilter === void 0 ? void 0 : activeFilter.types) === null || _a === void 0 ? void 0 : _a.includes(type))
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary/50 text-secondary-foreground hover:bg-secondary/80')}>
              {type === 'all' ? 'All types' : type}
            </button>);
        })}
        </div>
      </div>
      
      <scroll_area_1.ScrollArea className="h-full" style={{ maxHeight: maxHeight }}>
        <div className="p-3">
          {sortedSteps.length > 0 ? (sortedSteps.map(function (step) { return (<TimelineStep_1.TimelineStep key={step.id} step={step}/>); })) : (<div className="text-center text-muted-foreground py-8">
              No steps to display
            </div>)}
        </div>
      </scroll_area_1.ScrollArea>
    </div>);
};
exports.TimelineFeed = TimelineFeed;
