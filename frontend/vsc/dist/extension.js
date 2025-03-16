"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/extension.ts
var extension_exports = {};
__export(extension_exports, {
  activate: () => activate,
  deactivate: () => deactivate
});
module.exports = __toCommonJS(extension_exports);
var vscode = __toESM(require("vscode"));
var RAWebviewViewProvider = class {
  constructor(_extensionUri) {
    this._extensionUri = _extensionUri;
  }
  /**
   * Called when a view is first created to initialize the webview
   */
  resolveWebviewView(webviewView, context, _token) {
    webviewView.webview.options = {
      // Enable JavaScript in the webview
      enableScripts: true,
      // Restrict the webview to only load resources from the extension's directory
      localResourceRoots: [this._extensionUri]
    };
    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
  }
  /**
   * Creates HTML content for the webview with proper security policies
   */
  _getHtmlForWebview(webview) {
    const logoUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "assets", "RA.png"));
    const nonce = getNonce();
    return `<!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} https:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RA.Aid</title>
        <style>
          body {
            padding: 0;
            color: var(--vscode-foreground);
            font-size: var(--vscode-font-size);
            font-weight: var(--vscode-font-weight);
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
          }
          .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            text-align: center;
          }
          .logo {
            width: 100px;
            height: 100px;
            margin-bottom: 20px;
          }
          h1 {
            color: var(--vscode-editor-foreground);
            font-size: 1.3em;
            margin-bottom: 15px;
          }
          p {
            color: var(--vscode-foreground);
            margin-bottom: 10px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <img src="${logoUri}" alt="RA.Aid Logo" class="logo">
          <h1>RA.Aid</h1>
          <p>Your research and development assistant.</p>
          <p>More features coming soon!</p>
        </div>
      </body>
      </html>`;
  }
};
function getNonce() {
  let text = "";
  const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
function activate(context) {
  console.log('Congratulations, your extension "ra-aid" is now active!');
  const provider = new RAWebviewViewProvider(context.extensionUri);
  const viewRegistration = vscode.window.registerWebviewViewProvider(
    "ra-aid.view",
    // Must match the view id in package.json
    provider
  );
  context.subscriptions.push(viewRegistration);
  const disposable = vscode.commands.registerCommand("ra-aid.helloWorld", () => {
    vscode.window.showInformationMessage("Hello World from RA.Aid!");
  });
  context.subscriptions.push(disposable);
}
function deactivate() {
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  activate,
  deactivate
});
//# sourceMappingURL=extension.js.map
