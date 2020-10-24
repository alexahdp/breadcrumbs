import { Renderer } from './renderer.js';
import { Visualizer } from './visualizer.js';

const canvas = document.getElementById('screen');
const renderer = new Renderer(canvas);

let currentLine = null;
let nearestNode = null;
let cursorMode = ''; // draw | free

const getEventCoords = e => {
  const rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

canvas.addEventListener('dblclick', e => {
  const [x, y] = getEventCoords(e);
  renderer.addNode({ x, y, r: 10 });
});


canvas.addEventListener('mousedown', e => {
  if (!nearestNode) return;
  const [x, y] = getEventCoords(e);
  currentLine = {
    from: nearestNode,
    to: { x, y },
  };
});

canvas.addEventListener('mouseup', e => {
  if (currentLine) {
    if (nearestNode) {
      currentLine.to = nearestNode;
      renderer.addLine(currentLine);
    }
    currentLine = null;
  }
})

canvas.addEventListener('mousemove', e => {
  if (currentLine) {
    currentLine.to.x = e.offsetX;
    currentLine.to.y = e.offsetY;
  }
  const [x, y] = getEventCoords(e);
  nearestNode = renderer.getNearestNode(x, y);
});

let visualizeAnimator = null;
const startButton = document.getElementById('start');
startButton.addEventListener('click', e => {
  const visualizer = new Visualizer(renderer, renderer.nodes[0]);
  visualizeAnimator = visualizer.animator();
});

const saveButton = document.getElementById('save');
saveButton.addEventListener('click', e => {
  renderer.save();
});

const loadButton = document.getElementById('load');
loadButton.addEventListener('click', e => {
  renderer.restore();
});

renderer.addAnimation((canvas, ctx) => {
  if (currentLine) {
    renderer.drawLine(currentLine);
  }
  if (nearestNode) {
    renderer.drawNode(nearestNode, { color: 'red', stroke:  "#FF0000"});
  }
  if (visualizeAnimator) {
    const state = visualizeAnimator.next();
    
    if (!state.done) {
      renderer.drawNode(state.value.node, state.value.opts);
    }
  }
  
});

renderer.animate();
