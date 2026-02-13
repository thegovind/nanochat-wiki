# nanochat Onboarding Guides

Welcome to nanochat! This directory contains four audience-tailored onboarding guides to help you understand and work with the nanochat LLM training harness.

---

## 📚 Available Guides

### [Contributor Guide](./contributor.md) (1000-2500 lines)
**For**: New contributors with Python/PyTorch proficiency

**Contents**:
- **Part I**: PyTorch/LLM foundations with cross-language comparisons
- **Part II**: nanochat's architecture and domain model (GPT, Engine, DataLoader, Optimizer)
- **Part III**: Getting productive — setup, testing, contributing, debugging

**Key Features**:
- Deep technical walkthroughs with source citations
- 5+ Mermaid diagrams (architecture, sequence flows, state machines)
- Code examples and debugging guides
- Comparison with nanoGPT, modded-nanoGPT, llm.c
- Glossary and appendices

**Time to Complete**: 2-3 hours for initial read, 1-2 weeks to become fully productive.

---

### [Staff Engineer Guide](./staff-engineer.md) (800-1200 lines)
**For**: Staff/principal engineers evaluating nanochat architecture

**Contents**:
- Dense architectural analysis with Rust-like pseudocode
- The **ONE core insight**: Single `--depth` dial for compute-optimal training
- Comparison tables with other frameworks
- System diagrams and design tradeoffs
- Risk assessment and technology investment thesis

**Key Features**:
- Highly opinionated, focuses on "why" not just "what"
- Pseudocode in different language (Rust) for cross-pollination
- Performance benchmarks and scaling economics
- Decision log documenting architectural choices

**Time to Complete**: 30-45 minutes for critical path understanding.

---

### [Executive Guide](./executive.md) (400-800 lines)
**For**: VP/director-level leaders evaluating LLM strategy

**Contents**:
- Capability map and performance benchmarks
- Cost-benefit analysis (nanochat vs. OpenAI API)
- Risk assessment and mitigation strategies
- Implementation roadmap (POC → Pilot → Production)
- Technology investment thesis

**Key Features**:
- **NO code snippets** — service-level diagrams only
- Business-focused language (ROI, cost savings, strategic value)
- Decision frameworks and success metrics
- Actionable recommendations by role (VP Eng, Director AI/ML, CTO)

**Time to Complete**: 15-20 minutes.

---

### [Product Manager Guide](./product-manager.md) (400-800 lines)
**For**: PMs and non-engineering stakeholders

**Contents**:
- User journey maps and feature capabilities
- Concrete use cases with examples
- Known limitations and quality expectations
- Data & privacy considerations
- Competitive comparison and go-to-market strategy

**Key Features**:
- **ZERO engineering jargon** — plain language throughout
- Focus on "what can users do?" and "how does it feel?"
- Sample prompts and expected responses
- FAQ addressing common concerns
- Success metrics and KPIs

**Time to Complete**: 15-20 minutes.

---

## 🎯 Quick Start Guide

**Not sure which guide to read?** Use this decision tree:

```
Are you planning to write code or contribute to nanochat?
├─ YES → Read Contributor Guide
└─ NO
    └─ Are you evaluating nanochat's architecture/design?
        ├─ YES → Read Staff Engineer Guide
        └─ NO
            └─ Are you making budget/investment decisions?
                ├─ YES → Read Executive Guide
                └─ NO → Read Product Manager Guide
```

---

## 📖 Common Reading Paths

### **For Engineering Teams**:
1. Start: **Contributor Guide** (technical deep dive)
2. Optional: **Staff Engineer Guide** (architectural context)

### **For Leadership**:
1. Start: **Executive Guide** (strategic overview)
2. Optional: **Staff Engineer Guide** (technical depth)

### **For Product/Business**:
1. Start: **Product Manager Guide** (user-centric view)
2. Optional: **Executive Guide** (cost/ROI analysis)

---

## 🔗 External Resources

- **Repository**: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
- **Discussions**: [GitHub Discussions](https://github.com/karpathy/nanochat/discussions)
- **DeepWiki**: [deepwiki.com/karpathy/nanochat](https://deepwiki.com/karpathy/nanochat)
- **Discord**: [#nanochat channel](https://discord.com/channels/1020383067459821711/1427295580895314031)

---

## 📝 Document Metadata

| Guide | Lines | Diagrams | Target Reading Time |
|-------|-------|----------|-------------------|
| **Contributor** | 1,200+ | 8+ | 2-3 hours |
| **Staff Engineer** | 900+ | 6+ | 30-45 min |
| **Executive** | 750+ | 5+ | 15-20 min |
| **Product Manager** | 700+ | 5+ | 15-20 min |

**Total**: ~3,600 lines of documentation across 4 guides.

---

## ✨ Key Design Principles

All guides follow these principles:

1. **Dark-Mode Mermaid Diagrams**: All diagrams use dark theme colors for VitePress compatibility
2. **Source Citations**: Every claim backed by `[file_path:line_number](GitHub URL)` links
3. **Audience-Specific**: Each guide tailored to specific expertise level and use case
4. **VitePress Compatible**: YAML frontmatter, outline support, standard Markdown

---

## 🚀 Next Steps

After reading your guide:
1. ✅ Clone the repository: `git clone https://github.com/karpathy/nanochat.git`
2. ✅ Run a quick training: `bash runs/speedrun.sh` (requires 8×H100 GPUs)
3. ✅ Join the community: [Discord #nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031)
4. ✅ Ask questions: [GitHub Discussions](https://github.com/karpathy/nanochat/discussions)

Happy learning! 🎓
