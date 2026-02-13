# 📚 nanochat Wiki

Comprehensive documentation for [**nanochat**](https://github.com/karpathy/nanochat) — Andrej Karpathy's minimal full-stack ChatGPT clone that trains a GPT-2 grade model for ~$72 in ~3 hours.

> 🌐 **Live site:** [https://thegovind.github.io/nanochat-wiki/](https://thegovind.github.io/nanochat-wiki/)

## What's Inside

This wiki covers every stage of the nanochat pipeline:

| Section | Topics |
|---------|--------|
| **Getting Started** | Overview, philosophy, installation & setup |
| **Architecture** | System architecture, GPT model (RoPE, GQA, sliding window) |
| **Core Components** | Tokenizer, dataloader, MuonAdamW optimizer, Flash Attention 3, FP8 training |
| **Training** | Pretraining, supervised fine-tuning (SFT), reinforcement learning (RL) |
| **Evaluation** | CORE metric, BPB loss evaluation, task framework |
| **Inference & Serving** | KV cache engine, FastAPI web server, sandboxed tool use |
| **Infrastructure** | Checkpoint management, distributed training, report generation |
| **Onboarding** | Principal-level architecture guide + zero-to-hero learning path |

## Local Development

```bash
npm install        # Install dependencies
npm run dev        # Start dev server at http://localhost:5173
npm run build      # Production build
npm run preview    # Preview production build
```

## Built With

- [VitePress](https://vitepress.dev/) — Static site generator powered by Vue 3 + Vite
- [vitepress-plugin-mermaid](https://github.com/emersonbottero/vitepress-plugin-mermaid) — Mermaid diagram rendering
- [medium-zoom](https://github.com/francoischalifour/medium-zoom) — Click-to-zoom images
- Custom dark theme with GitHub-inspired color palette

## How This Wiki Was Generated

This documentation was generated using the **[Deep-Wiki plugin](https://github.com/nicepkg/deep-wiki)** for [GitHub Copilot CLI](https://githubnext.com/projects/copilot-cli). Deep-Wiki analyzes the source repository, traces code paths, and produces structured documentation with Mermaid diagrams — all packaged as a VitePress site.

## Deployment

The site is automatically deployed to GitHub Pages on every push to `main` via the workflow in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).

## License

This wiki documentation is provided under the MIT License.
