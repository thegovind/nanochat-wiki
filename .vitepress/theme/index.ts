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
      let scale = 1

      const overlay = document.createElement('div')
      overlay.className = 'mermaid-zoom-overlay'

      // Controls bar
      const controls = document.createElement('div')
      controls.className = 'mermaid-zoom-controls'
      controls.innerHTML = `
        <button class="mermaid-zoom-btn" data-action="out">−</button>
        <span class="mermaid-zoom-level">100%</span>
        <button class="mermaid-zoom-btn" data-action="in">+</button>
        <button class="mermaid-zoom-btn" data-action="reset">Reset</button>
        <button class="mermaid-zoom-btn mermaid-zoom-close" data-action="close">✕</button>
      `

      const container = document.createElement('div')
      container.className = 'mermaid-zoom-container'
      const clone = el.cloneNode(true)
      clone.classList.add('mermaid-zoomed-content')
      container.appendChild(clone)

      overlay.appendChild(controls)
      overlay.appendChild(container)
      document.body.appendChild(overlay)
      document.body.style.overflow = 'hidden'

      const levelEl = controls.querySelector('.mermaid-zoom-level')

      function setScale(s) {
        scale = Math.max(0.25, Math.min(5, s))
        clone.style.transform = `scale(${scale})`
        levelEl.textContent = `${Math.round(scale * 100)}%`
      }

      controls.addEventListener('click', (ev) => {
        const action = ev.target.dataset.action
        if (action === 'in') setScale(scale + 0.25)
        else if (action === 'out') setScale(scale - 0.25)
        else if (action === 'reset') setScale(1)
        else if (action === 'close') close()
        ev.stopPropagation()
      })

      // Scroll to zoom
      container.addEventListener('wheel', (ev) => {
        ev.preventDefault()
        setScale(scale + (ev.deltaY < 0 ? 0.1 : -0.1))
      }, { passive: false })

      function close() {
        overlay.remove()
        document.body.style.overflow = ''
        document.removeEventListener('keydown', onKey)
      }

      // Click overlay background to close
      overlay.addEventListener('click', (ev) => {
        if (ev.target === overlay || ev.target === container) close()
      })

      const onKey = (ev) => {
        if (ev.key === 'Escape') close()
        else if (ev.key === '+' || ev.key === '=') setScale(scale + 0.25)
        else if (ev.key === '-') setScale(scale - 0.25)
        else if (ev.key === '0') setScale(1)
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
