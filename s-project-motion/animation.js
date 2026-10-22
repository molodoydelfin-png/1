(() => {
  'use strict';
  const DURATION = 2.5;
  const svg = document.querySelector('#logo');
  const stage = document.querySelector('.stage');
  const lines = [...svg.querySelectorAll('[data-ray]')];
  const dots = [...svg.querySelectorAll('[data-dot]')];
  const center = svg.querySelector('[data-center]');
  const wordmark = svg.querySelector('[data-wordmark]');
  const backgroundRect = svg.querySelector('[data-background-rect]');
  const timeline = document.querySelector('#timeline');
  const timeLabel = document.querySelector('#time-label');
  const toggle = document.querySelector('#toggle');
  const loopControl = document.querySelector('#loop');
  const speedControl = document.querySelector('#speed');
  const themeControl = document.querySelector('#background');
  const params = new URLSearchParams(location.search);
  const renderMode = params.has('render');
  let time = 0, playing = false, loop = false, speed = 1, previous = null;
  let background = 'light';

  const clamp = x => Math.max(0, Math.min(1, x));
  const progress = (t, start, duration) => {
    if (t <= start) return 0;
    if (t >= start + duration - 1e-10) return 1;
    return clamp((t - start) / duration);
  };

  // Invert x on a cubic Bezier; evaluate y at the same parameter.
  // Identical easing is used by live playback, seeking and offline exports.
  function bezier(x1, y1, x2, y2) {
    const component = (u, a, b) => 3*(1-u)*(1-u)*u*a + 3*(1-u)*u*u*b + u*u*u;
    return x => {
      if (x <= 0 || x >= 1) return clamp(x);
      let lo = 0, hi = 1;
      for (let i = 0; i < 24; i++) {
        const u = (lo + hi) / 2;
        if (component(u, x1, x2) < x) lo = u; else hi = u;
      }
      return component((lo + hi) / 2, y1, y2);
    };
  }
  const rayEase = bezier(.45, 0, .25, 1);
  const textEase = bezier(.22, 1, .36, 1);
  const pointEase = bezier(0, 0, .58, 1);

  function paint() {
    const c = pointEase(progress(time, 0, .2));
    center.setAttribute('opacity', c);
    center.setAttribute('transform', `translate(203 205) scale(${.85+.15*c}) translate(-203 -205)`);
    lines.forEach((line, i) => {
      const p = rayEase(progress(time, .2 + i*.018, .738));
      line.setAttribute('stroke-dashoffset', 1-p);
      line.setAttribute('opacity', p === 0 ? 0 : 1);
    });
    dots.forEach((dot, i) => {
      const p = pointEase(progress(time, 1.1 + i*.008, .228));
      const x = dot.getAttribute('cx'), y = dot.getAttribute('cy');
      dot.setAttribute('opacity', p);
      dot.setAttribute('transform', `translate(${x} ${y}) scale(${.85+.15*p}) translate(${-x} ${-y})`);
    });
    const t = textEase(progress(time, 1.4, .7));
    wordmark.setAttribute('opacity', t);
    // Translate exactly 8 output pixels at 1920x1080, independent of the logo's 2.4 scale.
    wordmark.setAttribute('transform', `translate(${(8/2.4)*(1-t)} 0)`);
    timeline.value = time;
    timeLabel.value = `${time.toFixed(2)} / ${DURATION.toFixed(2)}`;
    toggle.textContent = playing ? 'Пауза' : 'Воспроизвести';
    toggle.setAttribute('aria-pressed', String(playing));
  }

  function setTime(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return;
    time = Math.max(0, Math.min(DURATION, number));
    previous = null;
    paint();
  }
  function play() {
    if (time >= DURATION) time = 0;
    playing = true; previous = null; paint();
  }
  function pause() { playing = false; previous = null; paint(); }
  function reset() { playing = false; setTime(0); }
  function replay() { setTime(0); play(); }
  function setBackground(value) {
    if (!['light', 'dark', 'transparent'].includes(value)) throw new Error('Unknown background');
    background = value;
    themeControl.value = value;
    stage.dataset.background = value;
    svg.setAttribute('color', value === 'dark' ? '#f7f7f7' : '#090909');
    backgroundRect.setAttribute('fill', value === 'dark' ? '#111216' : '#f7f7f7');
    backgroundRect.setAttribute('opacity', value === 'transparent' ? '0' : '1');
  }
  function frame(timestamp) {
    if (playing) {
      if (previous !== null) {
        time += Math.max(0, timestamp - previous) / 1000 * speed;
        if (time >= DURATION) {
          if (loop) time %= DURATION;
          else { time = DURATION; playing = false; }
        }
      }
      previous = timestamp;
      paint();
    } else previous = null;
    requestAnimationFrame(frame);
  }
  // A background browser tab must not skip ahead when it becomes visible again.
  document.addEventListener('visibilitychange', () => { previous = null; });
  toggle.addEventListener('click', () => playing ? pause() : play());
  document.querySelector('#replay').addEventListener('click', replay);
  document.querySelector('#reset').addEventListener('click', reset);
  timeline.addEventListener('input', () => { const value = timeline.value; pause(); setTime(value); });
  loopControl.addEventListener('change', () => { loop = loopControl.checked; });
  speedControl.addEventListener('change', () => { speed = Number(speedControl.value); previous = null; });
  themeControl.addEventListener('change', () => setBackground(themeControl.value));

  function finalSvg() {
    const clone = svg.cloneNode(true);
    clone.removeAttribute('id');
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('width', '1920'); clone.setAttribute('height', '1080');
    clone.querySelectorAll('[data-ray]').forEach(el => {
      el.setAttribute('stroke-dashoffset', '0'); el.setAttribute('opacity', '1');
    });
    clone.querySelectorAll('[data-dot], [data-center], [data-wordmark]').forEach(el => {
      el.setAttribute('opacity', '1'); el.removeAttribute('transform');
    });
    return new XMLSerializer().serializeToString(clone);
  }
  function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  const status = document.querySelector('#export-status');
  document.querySelector('#save-svg').addEventListener('click', () => {
    download(new Blob([finalSvg()], {type: 'image/svg+xml;charset=utf-8'}), `s-project-${background}.svg`);
    status.textContent = 'SVG сохранён: финальный кадр, выбранный фон.';
  });
  document.querySelector('#save-png').addEventListener('click', async () => {
    status.textContent = 'Подготовка PNG…';
    const url = URL.createObjectURL(new Blob([finalSvg()], {type:'image/svg+xml;charset=utf-8'}));
    try {
      const image = new Image();
      image.src = url; await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = 1920; canvas.height = 1080;
      canvas.getContext('2d').drawImage(image, 0, 0);
      const blob = await new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('PNG export failed')), 'image/png'));
      download(blob, `s-project-${background}.png`);
      status.textContent = 'PNG сохранён: 1920 × 1080, финальный кадр.';
    } catch (e) { status.textContent = `Не удалось сохранить PNG: ${e.message}`; }
    finally { URL.revokeObjectURL(url); }
  });

  window.logoMotion = {
    duration: DURATION, seek: setTime, play, pause, reset, replay, setBackground, finalSvg,
    getState: () => ({time, playing, loop, speed, background}),
  };
  if (renderMode) {
    document.documentElement.classList.add('render');
    document.body.classList.add('render');
  }
  setBackground(params.get('background') || 'light');
  setTime(0);
  if (!renderMode) play();
  requestAnimationFrame(frame);
})();
