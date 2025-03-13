"use strict";
var __rest = (this && this.__rest) || function (s, e) {
    var t = {};
    for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0)
        t[p] = s[p];
    if (s != null && typeof Object.getOwnPropertySymbols === "function")
        for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
            if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
                t[p[i]] = s[p[i]];
        }
    return t;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CollapsibleContent = exports.CollapsibleTrigger = exports.Collapsible = void 0;
var React = require("react");
var CollapsiblePrimitive = require("@radix-ui/react-collapsible");
var utils_1 = require("../../utils");
var Collapsible = CollapsiblePrimitive.Root;
exports.Collapsible = Collapsible;
var CollapsibleTrigger = CollapsiblePrimitive.Trigger;
exports.CollapsibleTrigger = CollapsibleTrigger;
var CollapsibleContent = React.forwardRef(function (_a, ref) {
    var className = _a.className, children = _a.children, props = __rest(_a, ["className", "children"]);
    return (<CollapsiblePrimitive.Content ref={ref} className={(0, utils_1.cn)("overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down", className)} {...props}>
    {children}
  </CollapsiblePrimitive.Content>);
});
exports.CollapsibleContent = CollapsibleContent;
CollapsibleContent.displayName = "CollapsibleContent";
