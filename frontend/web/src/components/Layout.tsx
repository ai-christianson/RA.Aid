import React, { ReactNode } from 'react';

interface LayoutProps {
  header: ReactNode;
  sidebar?: ReactNode;
  drawer?: ReactNode;
  children: ReactNode;
}

/**
 * Layout component using CSS Grid with named areas
 * This component creates a responsive layout with:
 * - Sticky header at the top (z-index 30)
 * - Sidebar on desktop (hidden on mobile)
 * - Main content area with proper positioning
 */
export const Layout: React.FC<LayoutProps> = ({ header, sidebar, drawer, children }) => {
  return (
    <>
      <style>{`
        .layout-grid {
          display: grid;
          min-height: 100vh;
          grid-template-areas:
            "header"
            "main";
          grid-template-rows: 64px 1fr;
          grid-template-columns: 1fr;
        }
        
        @media (min-width: 768px) {
          .layout-grid {
            grid-template-areas:
              "header header"
              "sidebar main";
            grid-template-columns: 250px 1fr;
            grid-template-rows: 64px 1fr;
          }
        }
        
        @media (min-width: 1024px) {
          .layout-grid {
            grid-template-columns: 300px 1fr;
          }
        }
        
        .layout-header {
          grid-area: header;
          position: sticky;
          top: 0;
          z-index: 30;
          height: 64px;
          display: flex;
          align-items: center;
        }
        
        .layout-sidebar {
          grid-area: sidebar;
          position: sticky;
          top: 64px;
          height: calc(100vh - 64px);
          overflow-y: auto;
          z-index: 20;
          display: none;
        }
        
        @media (min-width: 768px) {
          .layout-sidebar {
            display: block;
          }
        }
        
        .layout-main {
          grid-area: main;
          overflow-y: auto;
        }
      `}</style>
      
      <div className="layout-grid bg-background text-foreground">
        <header className="layout-header bg-background border-b border-border">
          {header}
        </header>
        
        {sidebar && (
          <aside className="layout-sidebar bg-background border-r border-border">
            {sidebar}
          </aside>
        )}
        
        {/* Mobile drawer - rendered outside grid */}
        {drawer}
        
        <main className="layout-main p-4">
          {children}
        </main>
      </div>
    </>
  );
};