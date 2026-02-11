import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import mediumZoom from 'medium-zoom'
import './custom.css'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()

    const initImageZoom = () => {
      mediumZoom('.vp-doc img:not(.no-zoom)', {
        background: 'rgba(0, 0, 0, 0.92)',
      })
    }

    const initMermaidZoom = (retries = 20) => {
      const svgs = document.querySelectorAll('.mermaid svg')
      if (svgs.length === 0 && retries > 0) {
        setTimeout(() => initMermaidZoom(retries - 1), 500)
        return
      }

      svgs.forEach((svg) => {
        if (svg.dataset.zoomEnabled) return
        svg.dataset.zoomEnabled = 'true'
        svg.style.cursor = 'pointer'

        svg.addEventListener('click', () => {
          const overlay = document.createElement('div')
          overlay.className = 'mermaid-zoom-overlay'

          const controls = document.createElement('div')
          controls.className = 'mermaid-zoom-controls'
          controls.innerHTML = `
            <button class="zoom-btn" data-action="in">+</button>
            <button class="zoom-btn" data-action="out">−</button>
            <button class="zoom-btn" data-action="reset">Reset</button>
            <button class="zoom-btn" data-action="close">✕</button>
          `

          const container = document.createElement('div')
          container.className = 'mermaid-zoom-container'

          const clone = svg.cloneNode(true) as SVGSVGElement
          if (!clone.getAttribute('viewBox')) {
            const bbox = (svg as SVGSVGElement).getBBox()
            clone.setAttribute('viewBox', `${bbox.x} ${bbox.y} ${bbox.width} ${bbox.height}`)
          }
          clone.style.width = '90vw'
          clone.style.height = '80vh'
          clone.style.maxWidth = 'none'
          container.appendChild(clone)

          overlay.appendChild(controls)
          overlay.appendChild(container)
          document.body.appendChild(overlay)
          document.body.style.overflow = 'hidden'

          let scale = 1
          let translateX = 0
          let translateY = 0
          let isPanning = false
          let startX = 0
          let startY = 0

          const updateTransform = () => {
            container.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`
          }

          controls.addEventListener('click', (e) => {
            const target = e.target as HTMLElement
            const action = target.dataset.action
            if (action === 'in') { scale *= 1.3; updateTransform() }
            else if (action === 'out') { scale /= 1.3; updateTransform() }
            else if (action === 'reset') { scale = 1; translateX = 0; translateY = 0; updateTransform() }
            else if (action === 'close') { close() }
          })

          container.addEventListener('wheel', (e) => {
            e.preventDefault()
            scale *= e.deltaY < 0 ? 1.1 : 0.9
            scale = Math.max(0.1, Math.min(10, scale))
            updateTransform()
          }, { passive: false })

          container.addEventListener('mousedown', (e) => {
            isPanning = true
            startX = e.clientX - translateX
            startY = e.clientY - translateY
            container.style.cursor = 'grabbing'
          })

          document.addEventListener('mousemove', (e) => {
            if (!isPanning) return
            translateX = e.clientX - startX
            translateY = e.clientY - startY
            updateTransform()
          })

          document.addEventListener('mouseup', () => {
            isPanning = false
            container.style.cursor = 'grab'
          })

          const close = () => {
            overlay.remove()
            document.body.style.overflow = ''
            document.removeEventListener('keydown', keyHandler)
          }

          overlay.addEventListener('click', (e) => {
            if (e.target === overlay) close()
          })

          const keyHandler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') close()
            else if (e.key === '+' || e.key === '=') { scale *= 1.3; updateTransform() }
            else if (e.key === '-') { scale /= 1.3; updateTransform() }
            else if (e.key === '0') { scale = 1; translateX = 0; translateY = 0; updateTransform() }
          }
          document.addEventListener('keydown', keyHandler)
        })
      })
    }

    onMounted(() => {
      initImageZoom()
      initMermaidZoom()
    })

    watch(
      () => route.path,
      () =>
        nextTick(() => {
          initImageZoom()
          initMermaidZoom()
        })
    )
  },
} satisfies Theme
