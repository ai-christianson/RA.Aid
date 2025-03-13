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
exports.SheetDescription = exports.SheetTitle = exports.SheetFooter = exports.SheetHeader = exports.SheetContent = exports.SheetClose = exports.SheetTrigger = exports.Sheet = void 0;
var React = require("react");
var SheetPrimitive = require("@radix-ui/react-dialog");
var class_variance_authority_1 = require("class-variance-authority");
var lucide_react_1 = require("lucide-react");
var utils_1 = require("../../utils");
var Sheet = SheetPrimitive.Root;
exports.Sheet = Sheet;
var SheetTrigger = SheetPrimitive.Trigger;
exports.SheetTrigger = SheetTrigger;
var SheetClose = SheetPrimitive.Close;
exports.SheetClose = SheetClose;
var SheetPortal = SheetPrimitive.Portal;
var SheetOverlay = React.forwardRef(function (_a, ref) {
    var className = _a.className, props = __rest(_a, ["className"]);
    return (<SheetPrimitive.Overlay className={(0, utils_1.cn)("fixed inset-0 z-50 bg-background/80 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0", className)} {...props} ref={ref}/>);
});
SheetOverlay.displayName = SheetPrimitive.Overlay.displayName;
var sheetVariants = (0, class_variance_authority_1.cva)("fixed z-50 gap-4 bg-background p-6 shadow-lg transition ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-300 data-[state=open]:duration-500", {
    variants: {
        side: {
            top: "inset-x-0 top-0 border-b data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top",
            right: "inset-y-0 right-0 h-full w-3/4 border-l data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right sm:max-w-sm",
            bottom: "inset-x-0 bottom-0 border-t data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom",
            left: "inset-y-0 left-0 h-full w-3/4 border-r data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left sm:max-w-sm",
        },
    },
    defaultVariants: {
        side: "right",
    },
});
var SheetContent = React.forwardRef(function (_a, ref) {
    var _b = _a.side, side = _b === void 0 ? "right" : _b, className = _a.className, children = _a.children, props = __rest(_a, ["side", "className", "children"]);
    return (<SheetPortal>
    <SheetOverlay />
    <SheetPrimitive.Content ref={ref} className={(0, utils_1.cn)(sheetVariants({ side: side }), className)} {...props}>
      {children}
      <SheetPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-secondary">
        <lucide_react_1.X className="h-4 w-4"/>
        <span className="sr-only">Close</span>
      </SheetPrimitive.Close>
    </SheetPrimitive.Content>
  </SheetPortal>);
});
exports.SheetContent = SheetContent;
SheetContent.displayName = SheetPrimitive.Content.displayName;
var SheetHeader = function (_a) {
    var className = _a.className, props = __rest(_a, ["className"]);
    return (<div className={(0, utils_1.cn)("flex flex-col space-y-2 text-center sm:text-left", className)} {...props}/>);
};
exports.SheetHeader = SheetHeader;
SheetHeader.displayName = "SheetHeader";
var SheetFooter = function (_a) {
    var className = _a.className, props = __rest(_a, ["className"]);
    return (<div className={(0, utils_1.cn)("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2", className)} {...props}/>);
};
exports.SheetFooter = SheetFooter;
SheetFooter.displayName = "SheetFooter";
var SheetTitle = React.forwardRef(function (_a, ref) {
    var className = _a.className, props = __rest(_a, ["className"]);
    return (<SheetPrimitive.Title ref={ref} className={(0, utils_1.cn)("text-lg font-semibold text-foreground", className)} {...props}/>);
});
exports.SheetTitle = SheetTitle;
SheetTitle.displayName = SheetPrimitive.Title.displayName;
var SheetDescription = React.forwardRef(function (_a, ref) {
    var className = _a.className, props = __rest(_a, ["className"]);
    return (<SheetPrimitive.Description ref={ref} className={(0, utils_1.cn)("text-sm text-muted-foreground", className)} {...props}/>);
});
exports.SheetDescription = SheetDescription;
SheetDescription.displayName = SheetPrimitive.Description.displayName;
