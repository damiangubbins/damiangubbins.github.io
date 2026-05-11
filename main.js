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
}

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });

const nnCanvas = document.getElementById("nn-canvas");
const nctx = nnCanvas.getContext("2d");

function get49BlockData(inputData) {
  const blocks = [];
  const representativeIndices = [];

  for (let r = 0; r < 7; r++) {
    for (let c = 0; c < 7; c++) {
      let sum = 0;
      for (let y = r * 4; y < (r + 1) * 4; y++) {
        for (let x = c * 4; x < (c + 1) * 4; x++) {
          sum += inputData[y * 28 + x];
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

let hoveredOutputNode = null;
let nnCanvasMoveTimeout = null;

nnCanvas.onmousemove = (e) => {
  if (nnCanvasMoveTimeout) clearTimeout(nnCanvasMoveTimeout);
  const rect = nnCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  hoveredOutputNode = null;

  for (let k = 0; k < 10; k++) {
    const nodeY = nnCanvas.height / 2 + (k - 10 / 2) * gapOutput + 10;
    const dist = Math.hypot(mx - (nnCanvas.width - 40), my - nodeY);

    if (dist < 15) {
      hoveredOutputNode = k;
      break;
    }
  }

  nnCanvasMoveTimeout = setTimeout(() => {
    drawNN(window.lastInputData);
    console.log("Redrawing NN with hoveredOutputNode:", hoveredOutputNode);
  }, 20); // 20ms debounce
};

function drawNN(inputData) {
  nctx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

  const { blocks, representativeIndices } = get49BlockData(inputData);

  const weights0 = model.layers[0].getWeights()[0].dataSync();
  const weights1 = model.layers[1].getWeights()[0].dataSync();

  let maxWeight0 = 0.01;
  for (let i = 0; i < weights0.length; i++) {
    const val = Math.abs(weights0[i]);
    if (val > maxWeight0) maxWeight0 = val;
  }

  let maxWeight1 = 0.01;
  for (let i = 0; i < weights1.length; i++) {
    const val = Math.abs(weights1[i]);
    if (val > maxWeight1) maxWeight1 = val;
  }

  const currentPrediction = tf.tidy(() => {
    const input = tf.tensor2d(inputData, [1, 784]);
    return model.predict(input).dataSync();
  });

  const activationValues = tf.tidy(() => {
    const input = tf.tensor2d(inputData, [1, 784]);
    const hiddenLayer = model.layers[0].apply(input);
    const activations = hiddenLayer.dataSync();
    const maxActivation = Math.max(...activations);
    return activations.map((a) => a / maxActivation);
  });

  blocks.forEach((avg, i) => {
    if (window.predicting && avg < 0.2) return;

    const startY = nnCanvas.height / 2 + (i - countInput / 2) * gapInput + 10;
    const pixelIdx = representativeIndices[i];

    for (let j = 0; j < countHidden; j++) {
      if (hoveredOutputNode !== null) {
        const outputWeight =
          weights1[j * countOutput + hoveredOutputNode] * (window.predicting ? activationValues[j] : 1.0);
        if (Math.abs(outputWeight) < (window.predicting ? 0.05 : 0.2)) continue;
      }

      const endY = nnCanvas.height / 2 + (j - countHidden / 2) * gapHidden + 10;
      const weight = weights0[pixelIdx * countHidden + j];
      drawLink(leftX, startY, midX, endY, weight, maxWeight0);
    }
  });

  activationValues.forEach((activation, i) => {
    if (window.predicting && activation < 0.2) return;

    const startY = nnCanvas.height / 2 + (i - countHidden / 2) * gapHidden + 10;

    for (let j = 0; j < countOutput; j++) {
      if (hoveredOutputNode !== null && j !== hoveredOutputNode) continue;

      const endY = nnCanvas.height / 2 + (j - countOutput / 2) * gapOutput + 10;
      const weight = weights1[i * countOutput + j];
      drawLink(midX, startY, rightX, endY, weight, maxWeight1);
    }
  });

  if (window.batch === 49 && !window.predicting) {
    drawNodeColumn(leftX, countInput, "Input", "#666", gapInput);
    drawNodeColumn(midX, countHidden, "Hidden", "#333", gapHidden);
    drawNodeColumn(rightX, countOutput, "Output", "#000", gapOutput, true);
  } else {
    drawNodeColumn(leftX, countInput, "Input", "#666", gapInput, false, blocks);
    drawNodeColumn(midX, countHidden, "Hidden", "#333", gapHidden, false, activationValues);
    drawNodeColumn(rightX, countOutput, "Output", "#000", gapOutput, true, currentPrediction);
  }
}

function drawLink(x1, y1, x2, y2, weight, maxWeight) {
  let alpha = Math.sqrt(Math.abs(weight) / maxWeight);

  nctx.beginPath();
  nctx.moveTo(x1, y1);
  nctx.lineTo(x2, y2);

  const visibleAlpha = Math.max(Math.pow(alpha, 2), 0.6);
  nctx.strokeStyle = weight > 0 ? `rgba(255, 50, 0, ${visibleAlpha})` : `rgba(0, 100, 255, ${visibleAlpha})`;

  nctx.lineWidth = Math.pow(alpha, 4) * 3;

  if (alpha > 0.95) {
    nctx.shadowBlur = 8;
    nctx.shadowColor = weight > 0 ? "red" : "blue";
  } else {
    nctx.shadowBlur = 0;
  }

  nctx.stroke();
  nctx.shadowBlur = 0;
}

function drawNodeColumn(x, count, label, color, gap, showDigits = false, scores = []) {
  nctx.fillStyle = color;
  nctx.font = "12px Arial";
  nctx.textAlign = "center";

  nctx.fillText(label, x, 20);

  for (let i = 0; i < count; i++) {
    const y = nnCanvas.height / 2 + (i - count / 2) * gap + 10;
    const score = scores[i] || 0;
    const size = 3 + score * 8;

    nctx.beginPath();
    nctx.arc(x, y, size, 0, Math.PI * 2);
    nctx.fill();

    if (showDigits) {
      nctx.fillText(i, x + 20, y + 5);
    }
  }
}

async function trainOnMNIST() {
  const btn = document.getElementById("train-btn");
  const status = document.getElementById("train-status");
  btn.disabled = true;
  status.innerText = "Status: Training...";

  const set = mnist.set(500, 0).training;

  const trainXs = tf.tensor2d(set.map((item) => item.input));
  const trainYs = tf.tensor2d(set.map((item) => item.output));

  window.predicting = false;

  await model.fit(trainXs, trainYs, {
    epochs: 3,
    batchSize: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        window.batch = batch;
        const randomIdx = Math.floor(Math.random() * 100);
        const currentSample = set[randomIdx].input;

        window.lastInputData = currentSample;

        drawSample(currentSample);
        drawNN(currentSample);

        await tf.nextFrame();
      },
    },
  });

  status.innerText = "Status: Training Complete! Now try drawing a digit.";
  btn.disabled = false;
  btn.innerText = "Train More (Add 500 more samples)";
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
  drawNN(window.lastInputData);
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

clearCanvas();
drawNN(window.lastInputData);
