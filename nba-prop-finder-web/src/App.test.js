import { render, screen } from '@testing-library/react';
import App from './App';

test('renders prop finder tab', () => {
  render(<App />);
  expect(screen.getByText(/prop finder/i)).toBeInTheDocument();
});
