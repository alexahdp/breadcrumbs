// https://bl.ocks.org/ravi4j/9d4871f11fa959fa6ded19a30d4b3227

class Node {
  constructor(val, parent = null) {
    this.root = null;
    this.val = val;
    this.next = null;
    this.prev = null;
    this.parent = parent;
  }
  
  replace(node, pos) {
    if (this.parent) {
      const pos = this.getPos();
      if (pos === 'next') {
        this.parent.setNext(node);
      } else {
        this.parent.setPrev(node);
      }
      this.parent = null;
    } else {
      node.parent = null;
    }

    if (this.root) {
      node.root = this.root;
      this.root.root = node;
      this.root = null;
    }

    if (pos === 'next') {
      node.setNext(this);
    }
    else if (pos === 'prev') {
      node.setPrev(this);
    }
  }

  setNext(node) {
    this.next = node;
    node.parent = this;
  }

  setPrev(node) {
    this.prev = node;
    node.parent = this;
  }

  getMax() {
    return this.next ? this.next.getMax() : this;
  }

  getMin() {
    return this.prev ? this.prev.getMin() : this;
  }

  getPos() {
    if (!this.parent) return null;
    return this.parent.prev == this ? 'prev' : 'next';
  }

  getDepth() {
    const nextDepth = this.next ? this.next.getDepth() : 0;
    const prevDepth = this.prev ? this.prev.getDepth() : 0;
    return (nextDepth > prevDepth ? nextDepth : prevDepth) + 1;
  }

  get prevDepth() {
    return this.next ? this.next.getDepth() : 0;
  }

  get nextDepth() {
    return this.prev ? this.prev.getDepth() : 0;
  }

  balance() {
    this.validate();

    // сравнить глубину левого и правого поддерева
    // если глубина одного из них больше - поставить
    // его на место себя
    const leftDepth = this.prev ? this.prev.getDepth() : 0;
    const rightDepth = this.next ? this.next.getDepth() : 0;

    if (Math.abs(leftDepth - rightDepth) <= 1) {
      if (this.parent) {
        this.parent.balance();
      }
      return;
    }

    if (leftDepth > rightDepth) {
      this.leftRotation();
    } else {
      this.rightRotation();
    }

    if (this.parent) {
        this.parent.balance();
      }
  }

  leftRotation() {
    const max = this.prev.getMax();

    if (max.prev) {
      max.replace(max.prev);
      max.prev = null;
    } else {
      max.parent[max.getPos()] = null;
    }

    this.replace(max, 'next');
    
    if (this.prev) {
      max.setPrev(this.prev);
      this.prev = null;
    }
  }

  rightRotation() {
    const min = this.next.getMin();
    
    if (min.next) {
      min.replace(min.next);
      min.next = null;
    } else {
      min.parent[min.getPos()] = null;
    }

    this.replace(min, 'prev');

    if (this.next) {
      min.setNext(this.next);
      this.next = null;
    }
  }

  add(val, balance = false) {
    let cur = this;

    while (cur) {
      if (val === cur.val) {
        cur = null;
      }
      else if (val < cur.val) {
        if (cur.prev) {
          cur = cur.prev;
        }
        else if (cur.prev) {
          const newNode = new Node(val);
          cur.prev.replace(newNode, 'prev');
          if (balance) {
            cur.balance();
          }
          return;
        }
        else {
          cur.prev = new Node(val, cur);
          if (balance && cur.parent) {
            cur.parent.balance();
          }
          return;
        }
      }
      else if (val > cur.val) {
        if (cur.next) {
          cur = cur.next;
        }
        else if (cur.next) {
          const newNode = new Node(val);
          cur.next.replace(newNode, 'next');
          if (balance) {
            cur.balance();
          }
          return;
        }
        else {
          cur.next = new Node(val, cur);
          if (balance && cur.parent) {
            cur.parent.balance();
          }
          return;
        }
      }
    }
  }

  remove(val) {
    let cur = this;

    while (cur) {
      if (val === cur.val) {
        if (cur.prev) {
          const max = cur.prev.getMax();
          max.parent[max.getPos()] = null;
          cur.replace(max);
          max.next = cur.next;
          if (max.next) {
            max.next.parent = max;
          }
          if (cur.prev !==  max) {
            max.prev = cur.prev;
            if (max.prev) {
              max.prev.parent = max;
            }
          }
          max.balance();
          return;
        }
        else if (cur.next) {
          const min = cur.next.getMin();
          min.parent[min.getPos()] = null;
          cur.replace(min);
          min.prev = cur.prev;
          if (min.prev) {
            min.prev.parent = min;
          }
          if (cur.next !== min) {
            min.next = cur.next;
            if (min.next) {
              min.next.parent = min;
            }
          }
          min.balance();
          return;
        }
        else {
          if (cur.parent) {
            cur.parent[cur.getPos()] = null;
          } else {
            cur.root = null;
          }
          return;
        }
      }
      else if (val < cur.val) {
        if (cur.prev) {
          cur = cur.prev;
        } else {
          return;
        }
      }
      else if (val > cur.val) {
        if (cur.next) {
          cur = cur.next;
        } else {
          return;
        }
      }
    }
  }

  // deep(cb) {
  //   cb(this);
  //   if (this.prev) this.prev.deep(cb);
  //   if (this.next) this.next.deep(cb);
  // }

  validate() {
    if (this.prev) {
      if (this.prev.val >= this.val) throw new Error(`Invalid tree: val: ${this.val}, prev: ${this.prev.val}`);
      if (this.prev.parent !== this) throw new Error(`Invalid tree: prev.parent not eq this, val: ${this.prev.val}`);
      this.prev.validate();
    }
    if (this.next) {
      if (this.next.val <= this.val) throw new Error(`Invalid tree: val: ${this.val}, next: ${this.next.val}`);
      if (this.next.parent !== this) throw new Error(`Invalid tree: next.parent not eq this, val: ${this.val}`);
      this.next.validate();
    }
    if (this.root) {
      if (this.parent !== null) {
        throw new Error('Invalid parent');
      }
    }
  }
}

class Tree {
  constructor() {
    this.root = null;
  }
  
  get(val) {
    return this.root.val === val;
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

if (typeof module != 'undefined' && module.exports) {
  module.exports = Node;
}
else if (window) {
  window.Tree = Tree;
}