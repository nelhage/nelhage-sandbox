(function () {
  const tip = document.getElementById('tip');
  let activeEl = null;

  function place(el) {
    const html = TIPS[el.getAttribute('data-tip')];
    if (!html) return;
    tip.innerHTML = html;
    tip.style.display = 'block';
    // force layout then measure
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const margin = 8;
    let left = r.left + r.width / 2 - tw / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - tw - margin));
    let top = r.top - th - margin;        // prefer above
    if (top < margin) top = r.bottom + margin; // else below
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
    tip.classList.add('show');
  }
  function hide() {
    tip.classList.remove('show');
    tip.style.display = 'none';
    activeEl = null;
  }

  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (el && el !== activeEl) { activeEl = el; place(el); }
  });
  document.addEventListener('mouseout', e => {
    const el = e.target.closest('[data-tip]');
    if (el && (!e.relatedTarget || !el.contains(e.relatedTarget))) hide();
  });
  // keyboard accessibility
  document.addEventListener('focusin', e => {
    const el = e.target.closest('[data-tip]');
    if (el) { activeEl = el; place(el); }
  });
  document.addEventListener('focusout', hide);
  window.addEventListener('scroll', () => { if (activeEl) place(activeEl); }, { passive: true });

  // HP chart dots <-> floor node cross-highlight
  document.querySelectorAll('.hpdot').forEach(dot => {
    dot.addEventListener('mouseenter', () => {
      const f = dot.getAttribute('data-floor');
      const node = [...document.querySelectorAll('.node')].find(n => n.querySelector('.node-floor')?.textContent === f);
      if (node) {
        node.classList.add('focus-floor');
        const tipKey = node.getAttribute('data-tip');
        if (tipKey && TIPS[tipKey]) { activeEl = dot; place(dot); tip.innerHTML = TIPS[tipKey]; place(dot); }
      }
    });
    dot.addEventListener('mouseleave', () => {
      document.querySelectorAll('.focus-floor').forEach(n => n.classList.remove('focus-floor'));
      hide();
    });
  });
})();
