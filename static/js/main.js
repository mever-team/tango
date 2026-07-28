/* TANGO project page — vanilla JS, no dependencies.
   Components: theme toggle, filmstrip scrubbers ("clips"), synchronized
   drift-race sliders, tabs, chart tooltips, BibTeX copy, scrollspy, reveals. */
(function () {
  'use strict';

  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Theme toggle ---------- */
  var toggleBtn = document.querySelector('.theme-toggle');
  if (toggleBtn) {
    var syncToggleLabel = function () {
      var light = document.documentElement.getAttribute('data-theme') === 'light';
      toggleBtn.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
      var mt = document.getElementById('meta-theme-color');
      if (mt) mt.setAttribute('content', light ? '#fafaf8' : '#0b0e12');
    };
    syncToggleLabel();
    toggleBtn.addEventListener('click', function () {
      var root = document.documentElement;
      var light = root.getAttribute('data-theme') === 'light';
      if (light) {
        root.removeAttribute('data-theme');
      } else {
        root.setAttribute('data-theme', 'light');
      }
      syncToggleLabel();
      try { localStorage.setItem('tango-theme', light ? 'dark' : 'light'); } catch (e) {}
    });
  }

  /* ---------- Clip: filmstrip scrubber ---------- */
  function makeClip(root) {
    var track = root.querySelector('.frames-track');
    var imgs = Array.prototype.slice.call(track.querySelectorAll('img'));
    var times = [];
    try { times = JSON.parse(root.getAttribute('data-times') || '[]'); } catch (e) {}
    var range = root.querySelector('input[type="range"]');
    var tc = root.querySelector('.timecode');
    var frames = root.querySelector('.frames');
    var api = {
      root: root,
      count: imgs.length,
      idx: 0,
      onUser: null,
      set: function (i) {
        i = Math.max(0, Math.min(imgs.length - 1, i));
        api.idx = i;
        for (var k = 0; k < imgs.length; k++) {
          imgs[k].classList.toggle('on', k === i);
        }
        if (range) {
          range.value = i;
          if (times[i]) range.setAttribute('aria-valuetext', times[i]);
        }
        if (tc && times[i]) tc.textContent = times[i];
      }
    };
    function userSet(i) {
      api.set(i);
      if (api.onUser) api.onUser(api.idx);
    }
    frames.addEventListener('pointermove', function (e) {
      if (e.pointerType === 'touch') return;
      var r = frames.getBoundingClientRect();
      var i = Math.floor(((e.clientX - r.left) / r.width) * imgs.length);
      userSet(i);
    });
    if (range) {
      range.addEventListener('input', function () { userSet(+range.value); });
    }
    var ui = root.querySelector('.clip-ui');
    if (ui) ui.removeAttribute('hidden');
    root.classList.add('is-ready');
    api.set(+(root.getAttribute('data-start') || 0));
    return api;
  }

  var clips = [];
  Array.prototype.forEach.call(document.querySelectorAll('.clip'), function (el) {
    clips.push(makeClip(el));
  });
  function clipsIn(container) {
    return clips.filter(function (c) { return container.contains(c.root); });
  }

  /* ---------- Race: shared slider + one-shot / looping autoplay ---------- */
  function makeRace(section) {
    var group = clipsIn(section);
    if (!group.length) return;
    var range = section.querySelector('.race-range');
    var axes = section.querySelectorAll('.seg-axis');
    var n = group[0].count;
    var loop = section.getAttribute('data-autoplay') === 'loop';
    var timer = null;
    var played = false;
    var userStopped = false;
    var times = [];
    try { times = JSON.parse(group[0].root.getAttribute('data-times') || '[]'); } catch (e) {}

    function paint(i) {
      group.forEach(function (c) { c.set(i); });
      if (range) {
        range.value = i;
        if (times[i]) range.setAttribute('aria-valuetext', times[i]);
      }
      Array.prototype.forEach.call(axes, function (axis) {
        var spans = axis.querySelectorAll('span');
        Array.prototype.forEach.call(spans, function (s, k) {
          s.classList.toggle('on', k === i);
        });
      });
    }
    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
    }
    function play() {
      if (reducedMotion || timer || userStopped) return;
      var i = group[0].idx;
      timer = setInterval(function () {
        i += 1;
        if (i >= n) {
          if (loop) {
            i = -1; /* hold one beat on last frame, restart next tick */
          } else {
            stop();
            return;
          }
        }
        if (i >= 0) paint(i);
      }, 1100);
    }

    function userStop() { userStopped = true; stop(); }
    if (range) {
      range.addEventListener('input', function () { userStop(); paint(+range.value); });
    }
    group.forEach(function (c) {
      c.onUser = function (i) { userStop(); paint(i); };
    });
    section.addEventListener('pointerdown', userStop);

    if (reducedMotion) {
      paint(n - 1); /* rest at the most informative state */
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          if (loop) {
            play();
          } else if (!played) {
            played = true;
            paint(0);
            play();
          }
        } else if (loop) {
          stop();
        }
      });
    }, { threshold: 0.35 });
    io.observe(section);
    paint(range ? +range.value : n - 1); /* initial state: axis highlight + valuetext */
  }

  Array.prototype.forEach.call(document.querySelectorAll('[data-race]'), makeRace);

  /* ---------- Tabs ---------- */
  Array.prototype.forEach.call(document.querySelectorAll('[role="tablist"]'), function (list) {
    var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
    function select(tab) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute('aria-selected', on ? 'true' : 'false');
        t.tabIndex = on ? 0 : -1;
        var panel = document.getElementById(t.getAttribute('aria-controls'));
        if (panel) panel.hidden = !on;
      });
    }
    tabs.forEach(function (t, i) {
      t.addEventListener('click', function () { select(t); });
      t.addEventListener('keydown', function (e) {
        var d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (!d) return;
        e.preventDefault();
        var next = tabs[(i + d + tabs.length) % tabs.length];
        next.focus();
        select(next);
      });
    });
    select(tabs.filter(function (t) { return t.getAttribute('aria-selected') === 'true'; })[0] || tabs[0]);
  });

  /* ---------- Chart tooltips ---------- */
  var tip = document.createElement('div');
  tip.className = 'viz-tip';
  document.body.appendChild(tip);

  function showTip(e, el) {
    var data;
    try { data = JSON.parse(el.getAttribute('data-tip')); } catch (err) { return; }
    var html = '';
    if (data.title) html += '<b>' + data.title + '</b>';
    (data.rows || []).forEach(function (r) {
      html += '<div class="row"><span>' + r[0] + '</span><span>' + r[1] + '</span></div>';
    });
    tip.innerHTML = html;
    tip.style.display = 'block';
    moveTip(e);
  }
  function moveTip(e) {
    var pad = 14;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') tip.style.display = 'none';
  });
  Array.prototype.forEach.call(document.querySelectorAll('[data-tip]'), function (el) {
    el.addEventListener('pointerenter', function (e) { showTip(e, el); });
    el.addEventListener('pointermove', moveTip);
    el.addEventListener('pointerleave', function () { tip.style.display = 'none'; });
    el.addEventListener('focus', function () {
      var r = el.getBoundingClientRect();
      showTip({ clientX: r.left + r.width / 2, clientY: r.top }, el);
    });
    el.addEventListener('blur', function () { tip.style.display = 'none'; });
  });

  /* ---------- BibTeX copy ---------- */
  var copyBtn = document.querySelector('.copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var code = document.getElementById('bibtex-code');
      var text = code ? code.textContent : '';
      function done() {
        copyBtn.classList.add('copied');
        copyBtn.querySelector('span').textContent = 'Copied';
        setTimeout(function () {
          copyBtn.classList.remove('copied');
          copyBtn.querySelector('span').textContent = 'Copy';
        }, 1800);
      }
      function legacy() {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, legacy);
      } else {
        legacy();
      }
    });
  }

  /* ---------- Hero elements -> top bar when scrolled past ---------- */
  var topnav = document.querySelector('.topnav');
  function mirrorInBar(selector, cls) {
    var el = document.querySelector(selector);
    if (!el || !topnav || !('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function (entries) {
      var e = entries[0];
      topnav.classList.toggle(cls, !e.isIntersecting && e.boundingClientRect.top < 0);
    }, { rootMargin: '-56px 0px 0px 0px' });
    io.observe(el);
  }
  mirrorInBar('.hero-links', 'show-actions');
  mirrorInBar('.hero-mark', 'show-mark');

  /* ---------- Scrollspy ---------- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.topnav-links a[href^="#"]'));
  if (navLinks.length && 'IntersectionObserver' in window) {
    var map = {};
    navLinks.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var current = null;
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          if (current) current.classList.remove('active');
          current = map[entry.target.id];
          if (current) current.classList.add('active');
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) spy.observe(el);
    });
  }

  /* ---------- Reveal on scroll + manifold draw ---------- */
  if ('IntersectionObserver' in window && !reducedMotion) {
    var rev = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          if (entry.target.classList.contains('manifold-fig')) {
            entry.target.classList.add('animate');
          }
          rev.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });
    Array.prototype.forEach.call(document.querySelectorAll('.reveal, .manifold-fig'), function (el) {
      rev.observe(el);
    });
  } else {
    Array.prototype.forEach.call(document.querySelectorAll('.reveal'), function (el) {
      el.classList.add('in');
    });
  }
})();
