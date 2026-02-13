import DefaultTheme from 'vitepress/theme'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'

function initMermaidZoom() {
  document.querySelectorAll('.mermaid').forEach((el) => {
    if (el.dataset.zoomBound) return
    el.dataset.zoomBound = 'true'
    el.addEventListener('click', (e) => {
      e.stopPropagation()
      // Create modal overlay
      const overlay = document.createElement('div')
      overlay.className = 'mermaid-zoom-overlay'
      const clone = el.cloneNode(true)
      clone.classList.add('mermaid-zoomed-content')
      overlay.appendChild(clone)
      document.body.appendChild(overlay)
      // Close on click
      overlay.addEventListener('click', () => {
        overlay.remove()
      })
      // Close on Escape
      const onKey = (ev) => {
        if (ev.key === 'Escape') {
          overlay.remove()
          document.removeEventListener('keydown', onKey)
        }
      }
      document.addEventListener('keydown', onKey)
    })
  })
}

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()
    onMounted(() => {
      setTimeout(initMermaidZoom, 1000)
    })
    watch(
      () => route.path,
      () => nextTick(() => setTimeout(initMermaidZoom, 1000))
    )
  },
}
