import React from "react";
import { 
  ThemeProvider, 
  ThemeToggle,
  Button,
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent
} from "../index";

/**
 * ShadcnDemo Component
 * 
 * Demonstrates various shadcn UI components including:
 * - Cards with different content
 * - Buttons with different variants and sizes
 * - Theme toggling between light and dark modes
 * - Responsive layout using Tailwind CSS
 */
export function ShadcnDemo() {
  return (
    <ThemeProvider defaultTheme="system">
      <div className="min-h-screen bg-background text-foreground p-6">
        <div className="container mx-auto">
          <header className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold">Shadcn UI Demo</h1>
            <ThemeToggle />
          </header>

          <section className="mb-10">
            <h2 className="text-2xl font-semibold mb-4">Button Variants</h2>
            <div className="flex flex-wrap gap-4">
              <Button variant="default">Default</Button>
              <Button variant="secondary">Secondary</Button>
              <Button variant="outline">Outline</Button>
              <Button variant="destructive">Destructive</Button>
              <Button variant="ghost">Ghost</Button>
              <Button variant="link">Link</Button>
            </div>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-semibold mb-4">Button Sizes</h2>
            <div className="flex flex-wrap items-center gap-4">
              <Button size="sm">Small</Button>
              <Button size="default">Default</Button>
              <Button size="lg">Large</Button>
              <Button size="icon">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-4 w-4"
                >
                  <path d="M12 5v14" />
                  <path d="M5 12h14" />
                </svg>
              </Button>
            </div>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-semibold mb-4">Cards</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Feature Card */}
              <Card>
                <CardHeader>
                  <CardTitle>Feature Highlight</CardTitle>
                  <CardDescription>Showcasing a key product feature</CardDescription>
                </CardHeader>
                <CardContent>
                  <p>This card demonstrates a clean way to highlight important features or content in your application.</p>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" className="w-full">Learn More</Button>
                </CardFooter>
              </Card>

              {/* Stats Card */}
              <Card>
                <CardHeader>
                  <CardTitle>Statistics</CardTitle>
                  <CardDescription>Key metrics and data points</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Total Users</span>
                    <span className="font-medium">1,234</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Active Sessions</span>
                    <span className="font-medium">567</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Conversion Rate</span>
                    <span className="font-medium">12.5%</span>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button variant="secondary" className="w-full">View Report</Button>
                </CardFooter>
              </Card>

              {/* CTA Card */}
              <Card className="bg-primary text-primary-foreground">
                <CardHeader>
                  <CardTitle>Get Started</CardTitle>
                  <CardDescription className="text-primary-foreground/80">Begin your journey today</CardDescription>
                </CardHeader>
                <CardContent>
                  <p>Sign up now and get access to all premium features for the first 30 days.</p>
                </CardContent>
                <CardFooter>
                  <Button variant="secondary" className="w-full">Sign Up</Button>
                </CardFooter>
              </Card>
            </div>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-semibold mb-4">Responsive Demo</h2>
            <Card>
              <CardContent className="p-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-muted p-4 rounded-md text-center">
                    <p className="mb-2 font-medium">Mobile View</p>
                    <p className="text-xs text-muted-foreground">Default</p>
                  </div>
                  <div className="hidden sm:block bg-muted p-4 rounded-md text-center">
                    <p className="mb-2 font-medium">Tablet View</p>
                    <p className="text-xs text-muted-foreground">sm:block</p>
                  </div>
                  <div className="hidden lg:block bg-muted p-4 rounded-md text-center">
                    <p className="mb-2 font-medium">Desktop View</p>
                    <p className="text-xs text-muted-foreground">lg:block</p>
                  </div>
                  <div className="hidden lg:block bg-muted p-4 rounded-md text-center">
                    <p className="mb-2 font-medium">Desktop View</p>
                    <p className="text-xs text-muted-foreground">lg:block</p>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="ghost">Reset</Button>
                <Button>Save Layout</Button>
              </CardFooter>
            </Card>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Interactive States</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Button States</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button className="w-full">Normal</Button>
                  <Button className="w-full hover:bg-primary/80" disabled>Disabled</Button>
                  <Button className="w-full focus:ring-2">Focus (Click me)</Button>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Interactive Example</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="mb-4">Toggle between these options:</p>
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline">Option 1</Button>
                    <Button variant="outline">Option 2</Button>
                    <Button variant="outline">Option 3</Button>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button variant="default" className="w-full">Apply Selection</Button>
                </CardFooter>
              </Card>
            </div>
          </section>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default ShadcnDemo;