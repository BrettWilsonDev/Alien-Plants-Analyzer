/**
 * app.js
 * Shared utilities: dark mode, TF/OpenCV info, province centroids
 * Included on all pages.
 */

(function () {
  // Dark mode / theme handling
  const html = document.documentElement;
  const saved = localStorage.getItem('theme');
  if (saved) html.dataset.theme = saved;

  function toggleTheme() {
    const next = html.dataset.theme === 'light' ? '' : 'light';
    html.dataset.theme = next;
    if (next) localStorage.setItem('theme', next);
    else localStorage.removeItem('theme');

    const emoji = document.querySelector('.emoji');
    if (next) {
      emoji.textContent = '🌙'; 
    } else {
      emoji.textContent = '☀️';
      
    }
  }

  document.addEventListener('click', (e) => {
    const t = e.target;
    if (t && (t.id === 'darkmode-toggle' || t.closest?.('#darkmode-toggle'))) {
      toggleTheme();
    }
  });

  // Expose TF info display function
  window.showTFInfo = async function (extraMsg = '') {
    const tfinfo = document.getElementById('tfinfo');
    if (!tfinfo) return;
    if (typeof tf === 'undefined') {
      tfinfo.innerHTML = '<p style="color:crimson">TensorFlow.js not loaded</p>';
      return;
    }
    try {
      await tf.ready();
      const mem = tf.memory();
      let html = `<ul>
        <li>TF.js Version: ${tf?.version?.tfjs ?? 'unknown'}</li>
        <li>Backend: ${tf.getBackend()}</li>
        <li>Memory: ${mem.numTensors} tensors, ${mem.numBytes} bytes</li>
      </ul>`;
      if (extraMsg) html += `<p>${extraMsg}</p>`;
      tfinfo.innerHTML = html;
    } catch (err) {
      tfinfo.innerHTML = `<p style="color:crimson">TF Info error: ${err.message}</p>`;
    }
  };

  // Expose OpenCV info function
  window.showOpenCVInfo = function () {
    const div = document.getElementById('opencvinfo');
    if (!div) return;
    if (typeof cv === 'undefined' || !cv.getBuildInformation) {
      div.innerHTML = '<p style="color:crimson">OpenCV.js not loaded or not ready</p>';
      return;
    }
    div.innerHTML = `<pre>${cv.getBuildInformation()}</pre>`;
  };

  // Province centroids (simple lookup)
  window.PROVINCE_CENTROIDS = {
    'Eastern Cape': [-32.2968, 26.4194],
    'Free State': [-28.4540, 26.7960],
    'Gauteng': [-26.2041, 28.0473],
    'KwaZulu-Natal': [-29.8739, 30.8930],
    'Limpopo': [-23.8974, 29.4428],
    'Mpumalanga': [-25.6559, 30.9250],
    'Northern Cape': [-29.0452, 22.1555],
    'North West': [-26.8706, 25.6577],
    'Western Cape': [-33.9180, 18.4232]
  };

  // make these functions visible for other scripts
  window.toggleTheme = toggleTheme;
})();

