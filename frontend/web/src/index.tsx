import React from 'react';
import ReactDOM from 'react-dom/client';
import { hello } from '@ra-aid/common';

hello();

const App = () => (
  <div>
    <h1>Hello from @ra-aid/web using Vite</h1>
  </div>
);

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
