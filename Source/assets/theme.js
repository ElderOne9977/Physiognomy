const toggle = element.querySelector('[role="switch"]');
const preference = window.matchMedia('(prefers-color-scheme: dark)');
const apply = (dark) => {
  document.documentElement.classList.toggle('dark', dark);
  document.body.classList.toggle('dark', dark);
  toggle.setAttribute('aria-checked', String(dark));
};
let saved;
try { saved = localStorage.getItem('physio-theme'); } catch (_) {}
apply(saved === 'dark' || (saved !== 'light' && preference.matches));
toggle.addEventListener('click', () => {
  const dark = toggle.getAttribute('aria-checked') !== 'true';
  apply(dark);
  saved = dark ? 'dark' : 'light';
  try { localStorage.setItem('physio-theme', saved); } catch (_) {}
});
preference.addEventListener('change', (event) => {
  if (!saved) apply(event.matches);
});
