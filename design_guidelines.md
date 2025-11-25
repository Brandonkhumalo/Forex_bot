# Design Guidelines: ML Trading AI Dashboard

## Design Approach
**Reference-Based Approach** inspired by modern trading platforms (TradingView, Robinhood, Webull) combined with data-dashboard best practices. Focus on data density, real-time information clarity, and professional financial interface standards.

---

## Typography System

**Font Families:**
- Primary: Inter (via Google Fonts CDN) - Interface text, data labels
- Monospace: JetBrains Mono - Numerical data, prices, balances

**Hierarchy:**
- Dashboard Title: text-3xl font-bold
- Section Headers: text-xl font-semibold
- Card Titles: text-lg font-medium
- Body/Labels: text-base font-normal
- Data Values: text-base font-mono
- Small Labels: text-sm
- Metrics/Stats: text-2xl to text-4xl font-bold font-mono

---

## Layout System

**Spacing Primitives:** Use Tailwind units of 2, 4, 6, 8, 12, 16
- Component padding: p-4, p-6
- Section gaps: gap-4, gap-6
- Margins: m-4, m-6, m-8

**Grid Structure:**
- Dashboard uses sidebar + main content layout
- Sidebar: fixed w-64, left-aligned navigation
- Main content area: flex-1 with max-w-7xl container
- Analytics page: Grid-based layouts for charts (grid-cols-1 md:grid-cols-2 lg:grid-cols-3)

---

## Core Components

### Navigation
**Sidebar Navigation (Fixed Left):**
- Logo/Brand at top (h-16)
- Navigation links with icons (Heroicons)
- Active state indication with subtle treatment
- AI Status indicator (running/stopped) prominently displayed
- Account balance display at bottom of sidebar
- Logout button at very bottom

### Dashboard Layout
**Main Trading Dashboard:**
- Top bar: Account summary cards in grid (grid-cols-1 md:grid-cols-4)
  - Total Balance
  - Available Capital
  - Total P/L
  - Win Rate
- AI Control Center: Card with large Start/Stop button + current status
- Active Trades: Table with columns: Pair, Entry Price, Current Price, P/L, Time
- Recent Trades: Compact list, last 10 trades
- Quick Stats: Small metric cards showing key performance indicators

### Analytics Page
**Comprehensive Data Visualization:**
- Performance Overview: Large stat cards (grid-cols-2 lg:grid-cols-4)
  - Total Trades
  - Win Rate %
  - Best Pair
  - ML Accuracy
- Chart Section (grid-cols-1 lg:grid-cols-2):
  - P/L Over Time (line chart)
  - Win/Loss Distribution (pie/donut chart)
  - Strategy Performance Breakdown (bar chart)
  - Trade Volume by Pair (horizontal bar)
- Trade History Table: Sortable, filterable, paginated
- ML Model Metrics: Separate card showing training status, accuracy, last retrain

### Data Tables
- Clean, dense presentation
- Alternating row treatment for readability
- Fixed headers for long tables
- Sortable columns with visual indicators
- Monospace font for all numerical data
- Status badges for trade outcomes (Win/Loss/Open)

### Cards & Containers
- Consistent border treatment
- Subtle shadow for elevation
- Padding: p-6
- Rounded corners: rounded-lg
- Header with title + optional action button

### Buttons & Controls
**Primary Actions:**
- Large AI control button: px-8 py-4 text-lg rounded-lg
- Standard buttons: px-4 py-2 rounded-md
- Icon buttons: p-2 rounded-md

**States:**
- Disabled state with reduced opacity
- Loading state with spinner icon

### Forms (Auth Pages)
- Centered layout with max-w-md
- Input fields: p-3 rounded-lg border
- Labels: text-sm font-medium mb-2
- Error messages: text-sm below inputs
- Submit button: full width, py-3

---

## Component Structure Details

### Status Indicators
- AI Running: Pulsing indicator + "Active" text
- AI Stopped: Static indicator + "Inactive" text
- Trade Status: Badge components (Open/Closed/Win/Loss)

### Real-time Updates
- Live price updates with subtle flash animation on change
- Auto-refreshing data (visual indicator in corner)
- Timestamp display: "Last updated: X seconds ago"

---

## Accessibility
- All interactive elements have focus states (ring-2 ring-offset-2)
- Form inputs include proper labels and ARIA attributes
- Tables use proper semantic HTML (thead, tbody, th, td)
- Icon-only buttons include aria-label
- Minimum touch target size: 44x44px for mobile

---

## Icons
**Library:** Heroicons (via CDN)
**Usage:**
- Navigation: outline style, size-6
- Cards/Stats: outline style, size-8
- Buttons: solid style, size-5
- Tables: outline style, size-4
- Status indicators: custom SVG for pulse effect

---

## Responsive Behavior
- Mobile (< 768px): Stack sidebar as bottom nav or hamburger menu, single column layouts
- Tablet (768px - 1024px): Collapsible sidebar, 2-column grids
- Desktop (> 1024px): Full sidebar visible, 3-4 column grids for analytics

---

## Authentication Pages
- Clean, centered forms on plain background
- Logo centered above form
- Card container: max-w-md with p-8
- Social proof optional: "Trusted by X traders" text below form
- Link to alternate auth page (Login ↔ Register)

---

## Data Visualization
- Use Chart.js or Recharts library for all charts
- Consistent chart styling across analytics page
- Legend placement: bottom or right side
- Tooltip on hover with detailed information
- Responsive charts that scale with container

---

## Images
**No hero images** - this is a dashboard application focused on data and functionality. All visual content is data-driven (charts, tables, metrics).