import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'nanochat',
    description: 'The simplest experimental harness for training LLMs',
    base: '/nanochat-wiki/',
    cleanUrls: true,
    lastUpdated: true,

    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: '/nanochat-wiki/logo.svg' }],
    ],

    themeConfig: {
      logo: '/logo.svg',
      siteTitle: 'nanochat',

      nav: [
        { text: 'Getting Started', link: '/getting-started/overview' },
        { text: 'Deep Dive', link: '/deep-dive/architecture/gpt-model' },
        { text: 'Onboarding', link: '/onboarding/contributor' },
        { text: 'GitHub', link: 'https://github.com/karpathy/nanochat' },
      ],

      sidebar: {
        '/getting-started/': [
          {
            text: 'Getting Started',
            items: [
              { text: 'Overview', link: '/getting-started/overview' },
              { text: 'Installation', link: '/getting-started/installation' },
              { text: 'GPT-2 Speedrun', link: '/getting-started/speedrun-walkthrough' },
              { text: 'Quick Reference', link: '/getting-started/quick-reference' },
            ],
          },
        ],
        '/deep-dive/': [
          {
            text: 'Architecture',
            collapsed: false,
            items: [
              { text: 'GPT Transformer Model', link: '/deep-dive/architecture/gpt-model' },
              { text: 'Attention Mechanisms', link: '/deep-dive/architecture/attention-mechanisms' },
              { text: 'MLP & Blocks', link: '/deep-dive/architecture/mlp-and-blocks' },
              { text: 'Scaling & Config', link: '/deep-dive/architecture/scaling-and-configuration' },
            ],
          },
          {
            text: 'Data Pipeline',
            collapsed: false,
            items: [
              { text: 'Tokenizer', link: '/deep-dive/data-pipeline/tokenizer' },
              { text: 'Dataset', link: '/deep-dive/data-pipeline/dataset' },
              { text: 'Dataloader', link: '/deep-dive/data-pipeline/dataloader' },
            ],
          },
          {
            text: 'Training Stages',
            collapsed: false,
            items: [
              { text: 'Pretraining', link: '/deep-dive/training/pretraining' },
              { text: 'Supervised Fine-Tuning', link: '/deep-dive/training/supervised-finetuning' },
              { text: 'Reinforcement Learning', link: '/deep-dive/training/reinforcement-learning' },
            ],
          },
          {
            text: 'Optimization',
            collapsed: false,
            items: [
              { text: 'Muon/AdamW Optimizer', link: '/deep-dive/optimization/muon-adamw' },
              { text: 'Learning Rate Schedule', link: '/deep-dive/optimization/learning-rate-schedule' },
              { text: 'FP8 Training', link: '/deep-dive/optimization/fp8-training' },
            ],
          },
          {
            text: 'Evaluation',
            collapsed: false,
            items: [
              { text: 'CORE Metric', link: '/deep-dive/evaluation/core-metric' },
              { text: 'Evaluation Tasks', link: '/deep-dive/evaluation/evaluation-tasks' },
              { text: 'Bits Per Byte', link: '/deep-dive/evaluation/bits-per-byte' },
            ],
          },
          {
            text: 'Inference & Deployment',
            collapsed: false,
            items: [
              { text: 'Inference Engine', link: '/deep-dive/inference/inference-engine' },
              { text: 'Chat UI', link: '/deep-dive/inference/chat-ui' },
              { text: 'Code Execution', link: '/deep-dive/inference/code-execution' },
            ],
          },
        ],
        '/onboarding/': [
          {
            text: 'Onboarding Guides',
            items: [
              { text: 'Contributor', link: '/onboarding/contributor' },
              { text: 'Staff Engineer', link: '/onboarding/staff-engineer' },
              { text: 'Executive', link: '/onboarding/executive' },
              { text: 'Product Manager', link: '/onboarding/product-manager' },
            ],
          },
        ],
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/karpathy/nanochat' },
      ],

      search: {
        provider: 'local',
      },

      outline: {
        level: 'deep',
      },

      editLink: {
        pattern: 'https://github.com/karpathy/nanochat/edit/master/wiki/:path',
        text: 'Edit this page on GitHub',
      },
    },

    mermaid: {
      theme: 'dark',
    },
    mermaidPlugin: {
      class: 'mermaid',
    },

    markdown: {
      math: true,
    },
  })
)
