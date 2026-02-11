# Copilot Instructions for nanochat-wiki

## Project Overview

This is the **nanochat Wiki** — a VitePress-powered documentation site for [nanochat](https://github.com/karpathy/nanochat), Andrej Karpathy's minimal full-stack ChatGPT clone. The wiki was generated using the [Deep-Wiki plugin](https://github.com/nicepkg/vitepress-plugin-mermaid) for GitHub Copilot CLI.

## Tech Stack

- **VitePress** (v1.6+) — static site generator built on Vue 3 + Vite
- **Mermaid** — diagrams rendered via `vitepress-plugin-mermaid`
- **medium-zoom** — click-to-zoom on images
- Custom dark theme with GitHub-inspired color palette

## Project Structure

```
├── .vitepress/
│   ├── config.mts          # VitePress configuration (sidebar, nav, mermaid theme)
│   ├── theme/
│   │   ├── index.ts         # Custom theme setup (image zoom, mermaid zoom)
│   │   └── custom.css       # Dark theme styles
│   └── public/              # Static assets (logo.svg)
├── 01-getting-started/      # Overview & installation docs
├── 02-architecture/         # System architecture & GPT model docs
├── 03-core-components/      # Tokenizer, dataloader, optimizer, attention, FP8
├── 04-training/             # Pretraining, SFT, RL docs
├── 05-evaluation/           # CORE metric, loss eval, task framework
├── 06-inference/            # Engine, web server, tool use docs
├── 07-infrastructure/       # Checkpoints, distributed training, reports
├── index.md                 # Homepage (VitePress frontmatter hero)
├── onboarding-guide.md      # Principal-level architecture deep-dive
└── zero-to-hero-guide.md    # Beginner-friendly learning path
```

## Conventions

- **Markdown files** use standard VitePress markdown with Mermaid code blocks for diagrams
- **Mermaid diagrams** must use dark-mode color variables (see existing diagrams for the palette)
- **Sidebar** is defined in `.vitepress/config.mts` — update it when adding/removing pages
- **No trailing slashes** in internal links (use `/01-getting-started/overview` not `/01-getting-started/overview/`)
- Pages are numbered by section (`01-`, `02-`, etc.) for logical ordering

## Commands

```bash
npm run dev      # Local dev server with hot reload
npm run build    # Production build (output in .vitepress/dist/)
npm run preview  # Preview production build locally
```

## Adding New Pages

1. Create the `.md` file in the appropriate section directory
2. Add a sidebar entry in `.vitepress/config.mts` under the correct section
3. Use existing pages as templates for frontmatter and Mermaid diagram styling

## Mermaid Diagram Guidelines

- Always use the dark theme color palette defined in `config.mts`
- Key colors: `#2d333b` (node bg), `#6d5dfc` (brand/border), `#e6edf3` (text), `#161b22` (cluster bg)
- Diagrams are click-to-zoom enabled via the custom theme

## Deployment

The site is deployed to GitHub Pages via GitHub Actions (`.github/workflows/deploy.yml`). Pushes to `main` trigger automatic builds and deployment.
