const genId = () => {
  return (new Date()).getTime().toString() + Math.round(Math.random()* 1000).toString();
}

export class Controller {
  constructor() {
    this.nodes = [];
    this.lines = [];
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