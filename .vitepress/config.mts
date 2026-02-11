import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'nanochat Wiki',
    description: 'Documentation for nanochat — the minimal full-stack ChatGPT clone',
    base: '/nanochat-wiki/',
    ignoreDeadLinks: true,
    appearance: 'dark',
    head: [
      ['link', { rel: 'icon', href: '/logo.svg' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
      ['link', { href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap', rel: 'stylesheet' }],
    ],
    markdown: {
      lineNumbers: true,
    },
    themeConfig: {
      logo: '/logo.svg',
      outline: { level: [2, 3] },
      nav: [
        { text: 'Home', link: '/' },
        { text: 'GitHub', link: 'https://github.com/karpathy/nanochat' },
      ],
      sidebar: [
        {
          text: '🚀 ONBOARDING',
          collapsed: false,
          items: [
            { text: 'Principal-Level Guide', link: '/onboarding-guide' },
            { text: 'Zero to Hero', link: '/zero-to-hero-guide' },
          ],
        },
        {
          text: '01 · Getting Started',
          collapsed: false,
          items: [
            { text: 'Overview & Philosophy', link: '/01-getting-started/overview' },
            { text: 'Installation & Setup', link: '/01-getting-started/installation' },
          ],
        },
        {
          text: '02 · Architecture',
          collapsed: false,
          items: [
            { text: 'System Architecture', link: '/02-architecture/system-architecture' },
            { text: 'GPT Model', link: '/02-architecture/gpt-model' },
          ],
        },
        {
          text: '03 · Core Components',
          collapsed: false,
          items: [
            { text: 'Tokenizer', link: '/03-core-components/tokenizer' },
            { text: 'Dataloader', link: '/03-core-components/dataloader' },
            { text: 'Optimizer (MuonAdamW)', link: '/03-core-components/optimizer' },
            { text: 'Flash Attention', link: '/03-core-components/flash-attention' },
            { text: 'FP8 Training', link: '/03-core-components/fp8-training' },
          ],
        },
        {
          text: '04 · Training',
          collapsed: true,
          items: [
            { text: 'Pretraining', link: '/04-training/pretraining' },
            { text: 'Supervised Fine-Tuning', link: '/04-training/supervised-finetuning' },
            { text: 'Reinforcement Learning', link: '/04-training/reinforcement-learning' },
          ],
        },
        {
          text: '05 · Evaluation',
          collapsed: true,
          items: [
            { text: 'CORE Metric', link: '/05-evaluation/core-metric' },
            { text: 'Loss Evaluation (BPB)', link: '/05-evaluation/loss-evaluation' },
            { text: 'Task Framework', link: '/05-evaluation/task-framework' },
          ],
        },
        {
          text: '06 · Inference & Serving',
          collapsed: true,
          items: [
            { text: 'Engine & KV Cache', link: '/06-inference/engine-kv-cache' },
            { text: 'Web Server', link: '/06-inference/web-server' },
            { text: 'Tool Use & Execution', link: '/06-inference/tool-use' },
          ],
        },
        {
          text: '07 · Infrastructure',
          collapsed: true,
          items: [
            { text: 'Checkpoint Management', link: '/07-infrastructure/checkpoint-management' },
            { text: 'Distributed Training', link: '/07-infrastructure/distributed-training' },
            { text: 'Report Generation', link: '/07-infrastructure/report-generation' },
          ],
        },
      ],
    },
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    mermaid: {
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#0d1117',
        primaryColor: '#2d333b',
        primaryTextColor: '#e6edf3',
        primaryBorderColor: '#6d5dfc',
        secondaryColor: '#1c2333',
        secondaryTextColor: '#e6edf3',
        secondaryBorderColor: '#6d5dfc',
        tertiaryColor: '#161b22',
        tertiaryTextColor: '#e6edf3',
        tertiaryBorderColor: '#30363d',
        lineColor: '#8b949e',
        textColor: '#e6edf3',
        mainBkg: '#2d333b',
        nodeBkg: '#2d333b',
        nodeBorder: '#6d5dfc',
        nodeTextColor: '#e6edf3',
        clusterBkg: '#161b22',
        clusterBorder: '#30363d',
        titleColor: '#e6edf3',
        edgeLabelBackground: '#1c2333',
        actorBkg: '#2d333b',
        actorTextColor: '#e6edf3',
        actorBorder: '#6d5dfc',
        actorLineColor: '#8b949e',
        signalColor: '#e6edf3',
        signalTextColor: '#e6edf3',
        labelBoxBkgColor: '#2d333b',
        labelBoxBorderColor: '#6d5dfc',
        labelTextColor: '#e6edf3',
        loopTextColor: '#e6edf3',
        activationBorderColor: '#6d5dfc',
        activationBkgColor: '#1c2333',
        sequenceNumberColor: '#e6edf3',
        noteBkgColor: '#2d333b',
        noteTextColor: '#e6edf3',
        noteBorderColor: '#6d5dfc',
        classText: '#e6edf3',
        labelColor: '#e6edf3',
        altBackground: '#161b22',
      },
    },
  })
)
