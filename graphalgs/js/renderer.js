const genId = () => {
  return (new Date()).getTime().toString() + Math.round(Math.random()* 1000).toString();
}

export class Renderer {

  constructor(canvas) {
    this.canvas = canvas;
    this.nodes = [];
    this.lines = [];
    this.animations = [];
  }

  addAnimation(animation) {
    this.animations.push(animation);
  }

  addLine(line) {
    line.id = genId();
    line.from.to.push(line);
    line.to.from.push(line);
    this.lines.push(line);
  }

  addNode(node) {
    node.id = genId();
    node.to = [];
    node.from = [];
    // console.log(node)
    this.nodes.push(node);
  }

  getNearestNode(x, y, r = 1000) {
    let minDist = Infinity;
    let nearestNode = null;
    this.nodes.forEach(node => {
      const d = (x - node.x)**2 + (y - node.y)**2;
      if (d < r && d < minDist) {
        minDist = d;
        nearestNode = node;
      }
    });
    return nearestNode;
  }

  drawNode(node, opts = {}) {
    const ctx = this.canvas.getContext('2d');
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r, 0, Math.PI*2, true);
    if (opts.color) {
      ctx.fillStyle = "red";
      ctx.fill();
    }
    if (opts.stroke) {
      ctx.strokeStyle = "#FF0000";
      ctx.stroke();
    }
    ctx.strokeStyle = "#000";
  }

  drawLine(line) {
    const ctx = this.canvas.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(line.from.x, line.from.y);
    ctx.lineTo(line.to.x, line.to.y);
    ctx.stroke();  
  }

  animate() {
    const context = this.canvas.getContext('2d');
    const ctx = context;
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    for (const node of this.nodes) {
      context.beginPath();
      context.arc(node.x, node.y, node.r, 0, Math.PI*2, true);
      context.stroke();
    }
  
    for (const line of this.lines) {
      ctx.beginPath();
      ctx.moveTo(line.from.x, line.from.y);
      ctx.lineTo(line.to.x, line.to.y);
      ctx.stroke();  
    }

    this.animations.forEach(animation => {
      animation(this.canvas, ctx);
    });
    
    requestAnimationFrame(() => this.animate());
  }

  save() {
    const data = {
      nodes: this.nodes.map(node => {
        return {
          id: node.id,
          from: node.from.map(l => l.id),
          to: node.to.map(l => l.id),
          x: node.x,
          y: node.y,
          r: node.r,
        };
      }),
      lines: this.lines.map(line => {
        return {
          id: line.id,
          from: line.from.id,
          to: line.to.id,
        };
      }),
    };
    localStorage.setItem('data', JSON.stringify(data));
  }

  restore() {
    const dataString = localStorage.getItem('data');
    const data = JSON.parse(dataString);
    const linesMap = data.lines.reduce((o, l) => {
      o[l.id] = l;
      return o;
    }, {});
    data.nodes.forEach(node => {
      node.from = node.from.map(lid => {
        linesMap[lid].to = node;
        return linesMap[lid];
      });
      node.to = node.to.map(lid => {
        linesMap[lid].from = node;
        return linesMap[lid];
      });
    });
    this.nodes = data.nodes;
    this.lines = data.lines;
  }
}