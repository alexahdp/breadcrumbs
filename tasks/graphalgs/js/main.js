import { Controller } from './controller.js';
import { Renderer } from './renderer.js';
import { Visualizer } from './visualizer.js';

const canvas = document.getElementById('screen');
const renderer = new Renderer(canvas);
const controller = new Controller();

let currentLine = null;
let nearestNode = null;

const getEventCoords = e => {
  const rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

canvas.addEventListener('dblclick', e => {
  const [x, y] = getEventCoords(e);
  controller.addNode({ x, y, r: 10 });
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
      controller.addLine(currentLine);
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
  nearestNode = controller.getNearestNode(x, y);
});

let visualizeAnimator = null;
const startButton = document.getElementById('start');
startButton.addEventListener('click', e => {
  const visualizer = new Visualizer(renderer, controller.nodes[0]);
  // visualizeAnimator = visualizer.animator();
  let node = controller.nodes[0];
  while (node && node.to.length > 0) {
    visualizer.add(node, node.to[0].to);
    node = node.to[0].to;
  }
  visualizeAnimator = visualizer.play();
});

const saveButton = document.getElementById('save');
saveButton.addEventListener('click', e => {
  controller.save();
});

const loadButton = document.getElementById('load');
loadButton.addEventListener('click', e => {
  controller.restore();
});

renderer.addAnimation(() => {
  if (currentLine) {
    renderer.drawLine(currentLine);
  }
  if (nearestNode) {
    renderer.drawNode(nearestNode, { color: 'red', stroke:  "#FF0000"});
  }
  if (visualizeAnimator) {
    const state = visualizeAnimator.next();
    
    if (!state.done) {
      state.value.forEach(value => {
        renderer.drawNode(value.node, value.opts);
      });
    }
  }  
});

function animate() {
  renderer.animate(controller.nodes, controller.lines);
  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);
