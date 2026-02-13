# FPR Framework — Online Playbook Design Brief

## Google AI Studio Prompt

Use the prompt below in Google AI Studio (Gemini) to generate the playbook design. Copy everything between the `---START PROMPT---` and `---END PROMPT---` markers.

---START PROMPT---

You are a senior front-end designer and developer. I need you to create a complete, single-page HTML playbook website for the "Future-Proof Rating (FPR) Framework" — a methodology for evaluating whether companies survive the AI transition. The full content is provided at the end of this prompt.

## Design Direction

Create a dark-mode-first, editorial-grade web experience inspired by the aesthetics of Stripe Press, Linear.app, and Vercel documentation. The tone is authoritative, data-driven, and slightly urgent — this is about corporate survival, not a marketing brochure.

## Design System Specifications

### Color Palette
- **Background**: Near-black (#0A0A0B) with subtle noise texture or dot-grid pattern
- **Surface cards**: Dark gray (#141416) with 1px border (#1E1E22) and subtle frosted glass effect (backdrop-filter: blur)
- **Primary accent**: Amber/Orange (#F59E0B) — signals urgency and importance
- **Secondary accent**: Cyan (#06B6D4) — for data points and links
- **Danger/Circuit breaker**: Red (#EF4444) with glow effect for circuit breaker sections
- **Success**: Emerald (#10B981) for high-scoring items
- **Text primary**: #E5E5E5
- **Text secondary**: #8A8A8A
- **Text emphasis**: White (#FFFFFF) used sparingly for key terms

### Typography
- **Headings**: Inter (or system sans-serif) — Bold/Black weight, tight letter-spacing (-0.02em)
- **Body**: Inter — Regular weight, 1.7 line-height, 16-18px base
- **Data/Scores/Labels**: JetBrains Mono (or system monospace) — used for scores, percentages, dimension codes (MCS, CPS, etc.), and technical labels
- **Section numbers**: Large monospace numerals as decorative elements

### Layout Principles
- Max content width: 900px centered, with full-bleed sections for visual breaks
- Generous vertical spacing: 120-160px between major sections
- Card-based layout for grouped information (dimensions, quadrants, etc.)
- Sticky left sidebar for navigation on desktop, hamburger on mobile
- Responsive: works on desktop (1200px+), tablet (768px), and mobile (375px)

## Component Design Requirements

### 1. Hero Section
- Full-viewport height
- Large, bold headline: "Future-Proof Rating Framework"
- Subtitle below in secondary text color
- The three key stats (12 dimensions, 4 circuit breakers, 4 temporal horizons) displayed as a horizontal row of monospace-styled stat blocks
- Subtle animated background: slow-moving gradient mesh or particle field (CSS only, no JS library)
- Scroll-down indicator arrow at bottom

### 2. Three Audience Mode Cards
- Horizontal row of three glass-effect cards
- Each card has an icon (use simple Unicode or SVG), title, description
- Hover effect: card lifts with increased glow on border
- Cards: Self-Assessment, Supply Chain Risk, Investment Due Diligence

### 3. AI Disruption Thesis Section
- Two-column layout for "Client Extinction Problem" and "Chain Collapse Problem"
- "Wrong Questions vs Right Questions" displayed as a contrasting comparison:
  - Left side: red-tinted card with strikethrough text for wrong questions
  - Right side: green-tinted card with checkmark bullets for right questions

### 4. AI Impact Quadrants
- 2x2 grid that actually looks like a quadrant diagram
- Each quadrant is a card with distinct background tint:
  - AI Windfall (top-left): green-tinted
  - AI Pivot (top-right): amber-tinted
  - AI Neutral (bottom-left): gray/neutral-tinted
  - AI Terminal (bottom-right): red-tinted
- Axis labels on the edges: "AI Leverage" on Y-axis, "Value Chain Replaceability" on X-axis
- Each card includes the example company description

### 5. Three Assessment Layers
- Three concentric-ring visual or three stacked horizontal bars showing the layers
- External Pressure (40%) — largest/outermost
- Internal Resilience (35%) — middle
- Adaptive Capacity (25%) — innermost/smallest
- Each layer lists its dimensions with codes and weights
- Use a donut chart or stacked bar visualization (CSS-only)

### 6. 12 Rating Dimensions (Accordion Section)
- Collapsible accordion for each dimension
- Each accordion header shows: dimension code (monospace, colored), name, weight percentage as a small pill badge, and a "Circuit Breaker" badge in red if applicable
- When expanded, show:
  - The key question in italic
  - Scoring criteria table (score range, description, examples) with color-coded rows
  - Micah Interview Mapping section with target roles and question clusters
  - LLM Enrichment Logic note
- "Expand All / Collapse All" toggle at section top
- Group dimensions under their layer headings (External Pressure, Internal Resilience, Adaptive Capacity) with distinct section backgrounds

### 7. Circuit Breakers Section
- High-impact visual treatment — this should feel alarming
- Red accent borders, subtle red glow
- Four circuit breaker rules displayed as large, bold conditional statements
- Format: "IF [condition] → Max Score: [cap] ([rating])"
- Brief explanation below each rule
- Implementation note in a distinct callout box

### 8. Chain Collapse Model
- Vertical flowchart showing the 5 chain links: Upstream → Company → Downstream → Lateral → End Consumer
- Connected by lines/arrows
- Each step is a numbered card with description
- Chain Collapse Score Modifier displayed as a horizontal gradient bar from red (0.70x) to green (1.10x)

### 9. Scoring Engine
- Step-by-step pipeline visualization (numbered steps 1-5)
- Each step is a card connected by arrows or a progress-line
- Use visual hierarchy to show the flow: Raw Scores → Weighted → Chain Modifier → Circuit Breakers → Rating

### 10. Sector-Adaptive Weights Table
- Clean data table with monospace numbers
- Alternating row backgrounds
- Highlighted cells where sector weight differs significantly from base
- Column headers styled distinctly

### 11. Rating Scale
- Eight rating levels displayed as a vertical stack or horizontal gradient
- Each level is a colored card/bar:
  - A+ (90-100): Bright green
  - A (80-89): Green
  - B+ (70-79): Light green
  - B (60-69): Yellow-green
  - C+ (50-59): Yellow/amber
  - C (40-49): Orange
  - D (25-39): Dark orange/red
  - F (0-24): Deep red
- Each includes the label (e.g., "AI Accelerated"), score range, and description

### 12. Micah Deep Integration
- Pipeline visualization similar to Scoring Engine but focused on interview flow
- Interview Coverage Matrix as a styled table with filled/empty circles (use Unicode ●○)
- Clean, readable table with proper column alignment

### 13. Temporal Analysis
- Four time horizons displayed as a horizontal timeline: T0 → T1 → T2 → T3
- Each point on the timeline expands into a card with description
- Competitive Half-Life section as a horizontal bar/gradient with time markers

### 14. Implementation Roadmap
- Five phases displayed as overlapping horizontal Gantt-style bars or a vertical timeline
- Each phase is a card with week range, title, and bullet list of deliverables
- Show the overlapping nature of phases (e.g., Phase 2 starts during Phase 1)

### 15. MCP Tool Specification
- Grid of tool cards (2-3 columns)
- Each card shows: tool name in monospace as a "function call" style, arrow, description
- Subtle code-editor aesthetic for this section

### 16. Footer
- Dark, minimal footer
- "FPR Framework v1.0" branding
- "Powered by Micah (Nexa Staff) · Designed for YVL Capital"
- Key stats repeated: 12 Dimensions · 4 Circuit Breakers · 4 Temporal Horizons · 3 Audience Modes

## Technical Requirements

- **Single HTML file** with embedded CSS and minimal vanilla JavaScript (for accordions, sticky nav, smooth scrolling)
- **No external dependencies** except Google Fonts (Inter + JetBrains Mono)
- **CSS Grid and Flexbox** for layout
- **CSS custom properties** (variables) for the design system
- **Smooth scroll behavior** and scroll-snap for major sections
- **Intersection Observer** for fade-in animations as sections enter viewport
- **Mobile responsive**: hamburger nav, stacked cards, smaller typography
- **Print-friendly**: @media print styles that switch to light background
- **Accessibility**: proper heading hierarchy, ARIA labels on interactive elements, sufficient color contrast
- **Performance**: no heavy animations, lazy approach for below-fold content

## Interaction Details

- Sticky navigation sidebar (desktop) or top bar (mobile) with section links
- Active section highlighted in nav based on scroll position
- Accordion sections toggle with smooth height animation
- Cards have subtle hover effects (transform, border glow)
- Smooth scroll to section when nav link clicked
- "Back to top" floating button that appears after first scroll

## FULL CONTENT TO INCLUDE:

[PASTE THE ENTIRE FPR FRAMEWORK CONTENT FROM THE ORIGINAL DOCUMENT HERE]

Please generate the complete, production-ready HTML file.

---END PROMPT---

## How to Use This Prompt

1. Open [Google AI Studio](https://aistudio.google.com/)
2. Select Gemini 2.5 Pro (or latest available model)
3. Copy the prompt between the markers above
4. Replace `[PASTE THE ENTIRE FPR FRAMEWORK CONTENT FROM THE ORIGINAL DOCUMENT HERE]` with the full FPR Framework text
5. Submit and wait for the HTML output
6. Save the output as `index.html`
7. Open in a browser to preview
8. Iterate: paste the HTML back with specific change requests

## Iteration Prompts (for refinement passes)

After the first generation, use these follow-up prompts:

### Pass 2 — Animation Polish
```
Review the HTML file. Add these micro-interactions:
- Staggered fade-in for card grids (each card delayed by 100ms)
- Number counter animation for stats in the hero section
- Parallax-lite effect on section backgrounds (CSS transform on scroll)
- Subtle pulse animation on circuit breaker badges
Keep everything CSS + vanilla JS only.
```

### Pass 3 — Data Visualization
```
Enhance the visual data representations:
- Convert the Assessment Layers into an animated SVG donut chart
- Add a CSS-only horizontal bar chart for dimension weights
- Create a visual 2x2 quadrant with plotted axes and gradient backgrounds
- Add sparkline-style indicators next to temporal projections
```

### Pass 4 — Responsive & Polish
```
Do a full responsive audit:
- Test at 375px, 768px, 1024px, 1440px breakpoints
- Ensure tables scroll horizontally on mobile with a fade edge indicator
- Stack all card grids to single column below 768px
- Increase touch targets for accordion headers on mobile
- Add a table of contents drawer on mobile
- Verify all text remains readable and no content overflows
```

## Design Inspiration References

| Reference | URL | What to Borrow |
|---|---|---|
| Stripe Press | press.stripe.com | Editorial typography, dark sections, content rhythm |
| Linear | linear.app | Dark mode, glassmorphism, animation quality |
| Vercel Docs | vercel.com/docs | Sidebar navigation, clean information architecture |
| Tailwind UI | tailwindui.com | Component patterns, card designs, responsive layouts |
| a16z Big Ideas | a16z.com/big-ideas | Tech editorial tone, data visualization style |
| Pitch | pitch.com | Presentation-quality web design, visual storytelling |
| Raycast | raycast.com | Dark UI, monospace accents, developer-tool aesthetic |
