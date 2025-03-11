// The module 'vscode' contains the VS Code extensibility API
import * as vscode from 'vscode';

/**
 * WebviewViewProvider implementation for the RA.Aid panel
 */
class RAWebviewViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly _extensionUri: vscode.Uri) {}

  /**
   * Called when a view is first created to initialize the webview
   */
  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    // Set options for the webview
    webviewView.webview.options = {
      // Enable JavaScript in the webview
      enableScripts: true,
      // Restrict the webview to only load resources from the extension's directory
      localResourceRoots: [this._extensionUri]
    };

    // Set the HTML content of the webview
    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
  }

  /**
   * Creates HTML content for the webview with proper security policies
   */
  private _getHtmlForWebview(webview: vscode.Webview): string {
    // Create a URI to the extension's assets directory
    const logoUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'assets', 'RA.png'));

    // Create a URI to the script file
    // const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'dist', 'webview.js'));

    // Use a nonce to whitelist scripts
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
}

/**
 * Generates a random nonce for CSP
 */
function getNonce() {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

// This method is called when your extension is activated
export function activate(context: vscode.ExtensionContext) {
  // Use the console to output diagnostic information (console.log) and errors (console.error)
  console.log('Congratulations, your extension "ra-aid" is now active!');

  // Register the WebviewViewProvider
  const provider = new RAWebviewViewProvider(context.extensionUri);
  const viewRegistration = vscode.window.registerWebviewViewProvider(
    'ra-aid.view', // Must match the view id in package.json
    provider
  );
  context.subscriptions.push(viewRegistration);

  // The command has been defined in the package.json file
  // Now provide the implementation of the command with registerCommand
  // The commandId parameter must match the command field in package.json
  const disposable = vscode.commands.registerCommand('ra-aid.helloWorld', () => {
    // The code you place here will be executed every time your command is executed
    // Display a message box to the user
    vscode.window.showInformationMessage('Hello World from RA.Aid!');
  });

  context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
export function deactivate() {}