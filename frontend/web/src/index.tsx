import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import { 
  hello, 
  Button, 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription,
  CardContent,
  CardFooter,
  Input,
  Switch
} from '@ra-aid/common';
// The CSS import happens through the common package's index.ts

hello();

const App = () => {
  const [inputValue, setInputValue] = useState('');
  const [switchState, setSwitchState] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLoadingClick = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 2000);
  };

  return (
    <div className="min-h-screen bg-background text-foreground dark">
      <div className="container mx-auto py-10 px-4">
        <header className="mb-10">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 inline-block text-transparent bg-clip-text">shadcn/ui Components Demo</h1>
          <p className="text-muted-foreground">A showcase of UI components from the common package</p>
        </header>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle>Button Component</CardTitle>
              <CardDescription>Various button variants from shadcn/ui</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-2">
                <Button variant="default">Default</Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="destructive">Destructive</Button>
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="link">Link</Button>
              </div>
              <div className="pt-2">
                <h3 className="text-sm font-medium mb-2">Button Sizes</h3>
                <div className="flex items-center gap-2">
                  <Button size="sm">Small</Button>
                  <Button>Default</Button>
                  <Button size="lg">Large</Button>
                </div>
              </div>
              <div className="pt-2">
                <h3 className="text-sm font-medium mb-2">Button States</h3>
                <div className="flex items-center gap-2">
                  <Button disabled>Disabled</Button>
                  <Button onClick={handleLoadingClick} disabled={loading}>
                    {loading ? 'Loading...' : 'Click to Load'}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle>Input Component</CardTitle>
              <CardDescription>Text input with various states</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium block mb-1">Standard Input</label>
                <Input 
                  placeholder="Type something..." 
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                />
                {inputValue && (
                  <p className="text-sm mt-1 text-muted-foreground">You typed: {inputValue}</p>
                )}
              </div>
              <div>
                <label className="text-sm font-medium block mb-1">Disabled Input</label>
                <Input disabled placeholder="Disabled input" />
              </div>
              <div>
                <label className="text-sm font-medium block mb-1">With Icon</label>
                <div className="relative">
                  <Input placeholder="Search..." />
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground">
                      <circle cx="11" cy="11" r="8"></circle>
                      <path d="m21 21-4.3-4.3"></path>
                    </svg>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle>Switch Component</CardTitle>
              <CardDescription>Toggle switch with controlled state</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Toggle theme</label>
                <Switch 
                  checked={switchState}
                  onCheckedChange={setSwitchState}
                />
              </div>
              <p className="text-sm text-muted-foreground">
                Current theme: <span className="font-medium">{switchState ? 'Dark' : 'Light'}</span>
              </p>
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Disabled switch</label>
                <Switch disabled />
              </div>
            </CardContent>
            <CardFooter>
              <p className="text-xs text-muted-foreground">Click the switch to toggle its state</p>
            </CardFooter>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle>Card Component</CardTitle>
              <CardDescription>Card with header, content, and footer sections</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-card-foreground">Cards are versatile containers that group related content and actions.</p>
              <div className="mt-4 p-3 bg-muted rounded-md">
                <code className="text-xs text-pink-400">
                  {'<Card>'}<br/>
                  {'  <CardHeader>'}<br/>
                  {'    <CardTitle>Title</CardTitle>'}<br/>
                  {'    <CardDescription>Description</CardDescription>'}<br/>
                  {'  </CardHeader>'}<br/>
                  {'  <CardContent>Content</CardContent>'}<br/>
                  {'  <CardFooter>Footer</CardFooter>'}<br/>
                  {'</Card>'}
                </code>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between border-t border-border pt-4">
              <Button variant="outline" size="sm">Cancel</Button>
              <Button size="sm">Save</Button>
            </CardFooter>
          </Card>
        </div>

        <footer className="mt-12 text-center text-muted-foreground text-sm">
          <p>Built with shadcn/ui components from the RA-Aid common package</p>
        </footer>
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);