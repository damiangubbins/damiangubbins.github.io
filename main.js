const drawCanvas = document.getElementById("draw-canvas");
const dctx = drawCanvas.getContext("2d");
let drawing = false;

dctx.lineWidth = 20;
dctx.lineCap = "round";
dctx.strokeStyle = "white";
drawCanvas.style.background = "black";

drawCanvas.onmousedown = (e) => {
  drawing = true;
  const rect = drawCanvas.getBoundingClientRect();
  dctx.beginPath();
  dctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
};

drawCanvas.onmouseup = () => (drawing = false);

drawCanvas.onmousemove = (e) => {
  if (!drawing) return;
  const rect = drawCanvas.getBoundingClientRect();
  dctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  dctx.stroke();
  dctx.beginPath();
  dctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
};

function clearCanvas() {
  dctx.fillRect(0, 0, 280, 280);
  dctx.beginPath();
  window.predicting = false;
  drawNN();
}

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });

const nnCanvas = document.getElementById("nn-canvas");
const nctx = nnCanvas.getContext("2d");

function get49BlockData() {
  const blocks = [];
  const representativeIndices = [];

  for (let r = 0; r < 7; r++) {
    for (let c = 0; c < 7; c++) {
      let sum = 0;
      for (let y = r * 4; y < (r + 1) * 4; y++) {
        for (let x = c * 4; x < (c + 1) * 4; x++) {
          sum += window.lastInputData[y * 28 + x];
        }
      }
      blocks.push(sum / 16);
      representativeIndices.push((r * 4 + 2) * 28 + (c * 4 + 2));
    }
  }
  return { blocks, representativeIndices };
}

const countInput = 49;
const countHidden = 32;
const countOutput = 10;

const gapInput = 12;
const gapHidden = 15;
const gapOutput = 30;

const leftX = 40,
  midX = nnCanvas.width / 2,
  rightX = nnCanvas.width - 40;

drawNodeColumn(leftX, countInput, "Pixels", "#666", gapInput);
drawNodeColumn(midX, countHidden, "Hidden", "#333", gapHidden);
drawNodeColumn(rightX, countOutput, "0-9", "#000", gapOutput);

window.hoveredOutputNode = null;
window.hoveredHiddenNode = null;
window.lockedOutputNode = null;
let nnCanvasMoveTimeout = null;

nnCanvas.onmousemove = (e) => {
  if (nnCanvasMoveTimeout) clearTimeout(nnCanvasMoveTimeout);
  const rect = nnCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  window.hoveredOutputNode = window.lockedOutputNode; // Keep locked node highlighted
  window.hoveredHiddenNode = null;

  for (let k = 0; k < 10; k++) {
    const nodeY = nnCanvas.height / 2 + (k - 10 / 2) * gapOutput + 10;
    const dist = Math.hypot(mx - (nnCanvas.width - 40), my - nodeY);

    if (dist < 15) {
      window.hoveredOutputNode = k;
      break;
    }
  }

  for (let k = 0; k < 32; k++) {
    const nodeY = nnCanvas.height / 2 + (k - 32 / 2) * gapHidden + 10;
    const dist = Math.hypot(mx - midX, my - nodeY);

    if (dist < 12) {
      window.hoveredHiddenNode = k;
      break;
    }
  }

  nnCanvasMoveTimeout = setTimeout(() => {
    drawNN();
  }, 20); // 20ms debounce
};

nnCanvas.onclick = () => {
  if (window.lockedOutputNode === window.hoveredOutputNode && window.lockedOutputNode !== null) {
    window.hoveredOutputNode = null;
    window.lockedOutputNode = null;
  } else {
    window.lockedOutputNode = window.hoveredOutputNode;
  }
  drawNN();
};

function drawNN() {
  nctx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

  const { blocks, representativeIndices } = get49BlockData();

  const weights0 = model.layers[0].getWeights()[0].dataSync();
  const weights1 = model.layers[1].getWeights()[0].dataSync();

  const w0p99 = percentile(weights0, 0.99);
  const w1p99 = percentile(weights1, 0.99);

  const w0p75 = percentile(weights0, 0.75);
  const w1p75 = percentile(weights1, 0.75);

  const currentPrediction = tf.tidy(() => {
    const input = tf.tensor2d(window.lastInputData, [1, 784]);
    return model.predict(input).dataSync();
  });

  const activationValues = tf.tidy(() => {
    const input = tf.tensor2d(window.lastInputData, [1, 784]);
    const hiddenLayer = model.layers[0].apply(input);
    const activations = hiddenLayer.dataSync();
    const maxActivation = Math.max(...activations);
    return activations.map((a) => a / maxActivation);
  });

  const inputHighlights = [];
  const hiddenHighlights = [];

  blocks.forEach((avg, i) => {
    if (avg < 0.1 && window.predicting) return;

    const startY = nnCanvas.height / 2 + (i - countInput / 2) * gapInput + 10;
    const pixelIdx = representativeIndices[i];

    for (let j = 0; j < countHidden; j++) {
      const endY = nnCanvas.height / 2 + (j - countHidden / 2) * gapHidden + 10;
      const weight = weights0[pixelIdx * countHidden + j];

      if (window.hoveredOutputNode !== null && Math.abs(weights1[j * countOutput + window.hoveredOutputNode]) < w1p75)
        continue;

      if (window.hoveredOutputNode !== null) hiddenHighlights.push(j);

      if (window.hoveredHiddenNode !== null && j !== window.hoveredHiddenNode) continue;
      if (window.hoveredHiddenNode !== null && Math.abs(weight) < w0p75) continue;

      if (window.hoveredOutputNode !== null && !inputHighlights.includes(i)) inputHighlights.push(i);

      drawLink(leftX, startY, midX, endY, weight, w0p99);
    }
  });

  activationValues.forEach((activation, i) => {
    const startY = nnCanvas.height / 2 + (i - countHidden / 2) * gapHidden + 10;

    if (window.hoveredHiddenNode !== null && i !== window.hoveredHiddenNode) return;

    for (let j = 0; j < countOutput; j++) {
      if (window.hoveredOutputNode !== null && j !== window.hoveredOutputNode) continue;

      const endY = nnCanvas.height / 2 + (j - countOutput / 2) * gapOutput + 10;
      const weight = weights1[i * countOutput + j];

      if (window.hoveredOutputNode !== null && Math.abs(weight) < w1p75) continue;
      if (window.hoveredHiddenNode !== null && Math.abs(weight) < w1p75) continue;

      drawLink(midX, startY, rightX, endY, weight, w1p99);
    }
  });

  if (!window.predicting) {
    drawNodeColumn(leftX, countInput, "Input", "#666", gapInput, false, [], inputHighlights);
    drawNodeColumn(midX, countHidden, "Hidden", "#333", gapHidden, false, [], hiddenHighlights);
    drawNodeColumn(rightX, countOutput, "Output", "#000", gapOutput, true, [], []);
  } else {
    drawNodeColumn(leftX, countInput, "Input", "#666", gapInput, false, blocks, inputHighlights);
    drawNodeColumn(midX, countHidden, "Hidden", "#333", gapHidden, false, activationValues, hiddenHighlights);
    drawNodeColumn(rightX, countOutput, "Output", "#000", gapOutput, true, currentPrediction, []);
  }
}

function percentile(arr, p) {
  if (!model._cachedPercentiles) model._cachedPercentiles = {};

  const cacheKey = arr.length + "-" + p;
  if (model._cachedPercentiles[cacheKey] && window.useCachedPercentiles) return model._cachedPercentiles[cacheKey];

  const sorted = Array.from(arr)
    .map(Math.abs)
    .sort((a, b) => a - b);

  const idx = Math.floor(p * sorted.length);
  const val = sorted[idx] || 0.01;
  model._cachedPercentiles[cacheKey] = val;

  return val;
}

function drawLink(x1, y1, x2, y2, weight, maxWeight) {
  // let alpha = Math.sqrt(Math.abs(weight) / (maxWeight + 1e-6));
  let alpha = Math.abs(weight) / (maxWeight + 1e-6);
  if (alpha > 1.0) alpha = 1.0;
  if (alpha < 0.08) return;

  nctx.beginPath();
  nctx.moveTo(x1, y1);
  nctx.lineTo(x2, y2);

  const visibleAlpha = Math.max(Math.pow(alpha, 2), 0.6);
  nctx.strokeStyle = weight > 0 ? `rgba(255, 50, 0, ${visibleAlpha})` : `rgba(0, 100, 255, ${visibleAlpha})`;

  nctx.lineWidth = Math.pow(alpha, 4) * 3;

  if (alpha > 0.95) {
    nctx.shadowBlur = 4;
    nctx.shadowColor = weight > 0 ? "red" : "blue";
  } else {
    nctx.shadowBlur = 0;
  }

  nctx.stroke();
  nctx.shadowBlur = 0;
}

function drawNodeColumn(x, count, label, color, gap, showDigits, scores = [], highlight = []) {
  nctx.font = "12px Arial";
  nctx.textAlign = "center";

  for (let i = 0; i < count; i++) {
    const y = nnCanvas.height / 2 + (i - count / 2) * gap + 10;
    const score = scores[i] || 0;

    let opacity = 1.0;
    if (window.hoveredOutputNode !== null) {
      if (showDigits) {
        opacity = i === window.hoveredOutputNode ? 1.0 : 0.1;
      } else {
        opacity = highlight.includes(i) ? 1.0 : 0.1;
      }
    }

    nctx.globalAlpha = opacity;
    nctx.fillStyle = color;

    const size = 3 + score * 8;
    nctx.beginPath();
    nctx.arc(x, y, size, 0, Math.PI * 2);
    nctx.fill();

    if (showDigits) {
      nctx.globalAlpha = 1.0;
      nctx.fillText(i, x + 20, y + 5);
    }
  }
  nctx.globalAlpha = 1.0;
}

async function trainOnMNIST() {
  const btn = document.getElementById("train-btn");
  const status = document.getElementById("train-status");
  btn.disabled = true;
  status.innerText = "Status: Training...";

  const set = mnist.set(500, 0).training;

  const trainXs = tf.tensor2d(set.map((item) => item.input));
  const trainYs = tf.tensor2d(set.map((item) => item.output));

  if (model._cachedPercentiles) model._cachedPercentiles = {};

  await model.fit(trainXs, trainYs, {
    epochs: 3,
    batchSize: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        const randomIdx = Math.floor(Math.random() * 100);
        const currentSample = set[randomIdx].input;

        window.lastInputData = currentSample;
        window.predicting = true;

        drawSample(currentSample);
        drawNN();

        await tf.nextFrame();
      },
    },
  });

  window.predicting = false;

  status.innerText = "Status: Training Complete! Now try drawing a digit.";
  btn.disabled = false;
  btn.innerText = "Train More (Add 500 more samples)";

  drawNN();
}

function predictDigit() {
  const tensor = tf.tidy(() => {
    let img = tf.browser.fromPixels(drawCanvas, 1);
    img = tf.image.resizeBilinear(img, [28, 28]);
    return img.div(255.0).reshape([1, 784]);
  });

  const prediction = model.predict(tensor);
  const digit = prediction.argMax(1).dataSync()[0];
  document.getElementById("prediction-text").innerText = `Guess: ${digit}`;

  window.predicting = true;
  window.lastInputData = tensor.dataSync();
  drawNN();
}

function drawSample(pixels) {
  const sCanvas = document.getElementById("sample-canvas");
  const sCtx = sCanvas.getContext("2d");
  const imgData = sCtx.createImageData(28, 28);
  for (let i = 0; i < pixels.length; i++) {
    const val = pixels[i] * 255;
    imgData.data[i * 4] = val;
    imgData.data[i * 4 + 1] = val;
    imgData.data[i * 4 + 2] = val;
    imgData.data[i * 4 + 3] = 255;
  }
  sCtx.putImageData(imgData, 0, 0);
}

window.lastInputData = new Array(784).fill(0);
window.predicting = false;
window.useCachedPercentiles = false;

clearCanvas();
drawNN();
