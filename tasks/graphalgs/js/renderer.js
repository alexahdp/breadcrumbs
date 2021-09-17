

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

  animate(nodes, lines) {
    const context = this.canvas.getContext('2d');
    const ctx = context;
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    for (const node of nodes) {
      context.beginPath();
      context.arc(node.x, node.y, node.r, 0, Math.PI*2, true);
      context.stroke();
    }
  
    for (const line of lines) {
      ctx.beginPath();
      ctx.moveTo(line.from.x, line.from.y);
      ctx.lineTo(line.to.x, line.to.y);
      ctx.stroke();  
    }

    this.animations.forEach(animation => {
      animation(this.canvas, ctx);
    });
  }
}