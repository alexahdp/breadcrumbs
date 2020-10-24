const dist = ({x: x1, y: y1}, {x: x2, y: y2}) => Math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2);

export class Visualizer {
  constructor(renderer, rootNode) {
    this.renderer = renderer;
    this.rootNode = rootNode;
  }

  *animator() {
    let currentNode = this.rootNode;
    while (currentNode != null && currentNode.to.length > 0) {
      yield {
        node: currentNode,
        opts: { color: 'red', stroke:  "#FF0000"},
      };

      const x1 = currentNode.to[0].to.x;
      const y1 = currentNode.to[0].to.y;

      const vec = { x: x1 - currentNode.x, y: y1 - currentNode.y };
      const len = Math.sqrt(vec.x ** 2 + vec.y ** 2);
      vec.x = vec.x / len;
      vec.y = vec.y / len;

      const dot = {
        x: currentNode.x,
        y: currentNode.y,
        r: 10
      };

      do {
        dot.x += vec.x;
        dot.y += vec.y;
        yield {
          node: dot,
          opts: { color: 'red', stroke:  "#FF0000"},
        };
      } while(dist(dot, currentNode.to[0].to) > 5);

      currentNode = currentNode.to[0].to;
    }
  }
}