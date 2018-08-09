import Node from './binaryNode.js';

class BinaryTree {
  constructor() {
    this.root = null;
  }

  add(val) {
    if (!this.root) {
      this.root = new Node(val);
      this.root.root = this;
      return;
    }
    
    this.root.add(val, true);
  }

  remove(val) {
    this.root.remove(val);
  }

  validate() {
    this.root.validate();
  }
  
  exportCSV() {
    return this._export(this.root, null, []);
  }
  
  _export(node, parent, result) {
    result.push({
      node: node.val,
      parent: (parent ? parent.val : ''),
    });
    
    if (node.prev) result = this._export(node.prev, node, result);
    if (node.next) result = this._export(node.next, node, result);
    return result;
  }
}

export default BinaryTree;