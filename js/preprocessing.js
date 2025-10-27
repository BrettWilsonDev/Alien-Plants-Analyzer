/**
 * preprocessing.js
 * OpenCV.js image reading and filter helpers.
 * Designed to be used from predict.js (no module system).
 *
 * Exposes:
 *  - window.readToMat(imgElement)
 *  - window.applyFilterToCanvas(filter, canvasElement)
 *  - window.freeOriginalMat()
 */

(function () {
  let originalMat = null;

  // Read <img> element into OpenCV Mat (RGBA)
  function readToMat(imgEl) {
    if (!imgEl) throw new Error('readToMat: imgEl required');
    // free old mat if present
    if (originalMat) {
      try { originalMat.delete(); } catch (e) {}
      originalMat = null;
    }
    // cv.imread handles img elements and returns an RGBA mat
    originalMat = cv.imread(imgEl);
    return originalMat;
  }

  // Apply a named filter and render to a canvas element using cv.imshow
  function applyFilterToCanvas(filter, canvasEl) {
    if (!originalMat) throw new Error('applyFilterToCanvas: call readToMat first');
    if (!canvasEl) throw new Error('applyFilterToCanvas: canvas element required');

    const dst = new cv.Mat();
    try {
      switch (filter) {
        case 'gray':
          cv.cvtColor(originalMat, dst, cv.COLOR_RGBA2GRAY);
          cv.imshow(canvasEl, dst);
          break;
        case 'blur':
          // convert to gray then blur for speed
          cv.cvtColor(originalMat, dst, cv.COLOR_RGBA2GRAY);
          cv.GaussianBlur(dst, dst, new cv.Size(7, 7), 0);
          cv.imshow(canvasEl, dst);
          break;
        case 'canny':
          cv.cvtColor(originalMat, dst, cv.COLOR_RGBA2GRAY);
          cv.Canny(dst, dst, 50, 150);
          cv.imshow(canvasEl, dst);
          break;
        case 'original':
        default:
          // show original RGBA mat
          cv.imshow(canvasEl, originalMat);
      }
    } catch (err) {
      throw err;
    } finally {
      try { dst.delete(); } catch (e) {}
      // keep originalMat around for re-filtering until freed explicitly
      canvasEl.style.display = 'block';
    }
  }

  function freeOriginalMat() {
    if (originalMat) {
      try { originalMat.delete(); } catch (e) {}
      originalMat = null;
    }
  }

  // expose
  window.readToMat = readToMat;
  window.applyFilterToCanvas = applyFilterToCanvas;
  window.freeOriginalMat = freeOriginalMat;
})();
