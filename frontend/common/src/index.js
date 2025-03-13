"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.hello = exports.getSampleAgentSteps = exports.getSampleAgentSessions = void 0;
// Entry point for @ra-aid/common package
require("./styles/global.css");
// Direct imports from sample-data
var sample_data_1 = require("./utils/sample-data");
Object.defineProperty(exports, "getSampleAgentSessions", { enumerable: true, get: function () { return sample_data_1.getSampleAgentSessions; } });
Object.defineProperty(exports, "getSampleAgentSteps", { enumerable: true, get: function () { return sample_data_1.getSampleAgentSteps; } });
// Export utility functions
__exportStar(require("./utils"), exports);
// Export all UI components
__exportStar(require("./components/ui"), exports);
// Export timeline components
__exportStar(require("./components/TimelineStep"), exports);
__exportStar(require("./components/TimelineFeed"), exports);
// Export session navigation components
__exportStar(require("./components/SessionDrawer"), exports);
__exportStar(require("./components/SessionSidebar"), exports);
// Export the hello function (temporary example)
var hello = function () {
    console.log("Hello from @ra-aid/common");
};
exports.hello = hello;
