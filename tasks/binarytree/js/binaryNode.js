
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

  balance() {
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
    if (this.val === val) return;

    if (val < this.val) {
      if (this.prev) {
        this.prev.add(val, balance);
      } else {
        this.prev = new Node(val, this);
        if (balance && this.parent) {
          this.parent.balance();
        }
      }
    } else {
      if (this.next) {
        this.next.add(val, balance);
      } else {
        this.next = new Node(val, this);
        if (balance && this.parent) {
          this.parent.balance();
        }
      }
    }
  }

  find(val) {
    if (this.val === val) {
      return this;
    }
    else if (val < this.val && this.prev) {
      return this.prev.find(val);
    }
    else if (val > this.val && this.next) {
      return this.next.find(val);
    }
    else {
      return null;
    }
  }

  remove(val, balance = false) {
    const removeNode = this.find(val);
    if ( ! removeNode) return;

    if ( ! removeNode.prev && ! removeNode.next) {
      if (removeNode.parent) {
        removeNode.parent[removeNode.getPos()] = null;
      } else {
        removeNode.root = null;
      }
    }
    else if (removeNode.prev) {
      removeNode.replace(removeNode.prev);
      if (balance && removeNode.prev.parent) {
        removeNode.prev.parent.balance();
      }
    }
    else {
      removeNode.replace(removeNode.next);
      if (balance && removeNode.next.parent) {
        removeNode.next.parent.balance();
      }
    }
  }

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

export default Node;