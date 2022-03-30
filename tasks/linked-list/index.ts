export class Node<T> {
  constructor(public value: T, public next: Node<T> | null = null) {}
}

export class List<T> {
  private head: Node<T> | null = null;
  private tail: Node<T> | null = null;

  constructor() {}

  add(value: T): Node<T> {
    const node = new Node(value);
    if (!this.head) {
      this.head = node;
      this.tail = node;
      return node;
    }
    if (!this.tail) {
      throw new Error('unconsistent state!');
    }

    this.tail.next = node;
    this.tail = node;
    return node;
  }

  preprend(value: T): Node<T> {
    const node = new Node(value);
    node.next = this.head;
    this.head = node;
    if (!this.tail) {
      this.tail = node;
    }
    return node;
  }

  search(cb: (n: Node<T>) => boolean): Node<T> | null {
    let currentNode = this.head;
    while (currentNode && !cb(currentNode)) {
      currentNode = currentNode.next;
    }
    return currentNode;
  }

  remove(cb: (n: Node<T>) => boolean): boolean {
    if (!this.head) {
      return false;
    }

    if (cb(this.head)) {
      if (this.head === this.tail) {
        this.tail = null;
        this.head = null;
      } else {
        this.head = this.head.next;
      }
      return true;
    }

    let currentNode: Node<T> = this.head;
    while (currentNode.next && cb(currentNode.next)) {
      currentNode = currentNode.next;
    }

    if (currentNode.next && cb(currentNode.next)) {
      if (currentNode.next !== this.tail) {
        currentNode.next = currentNode.next.next;
      } else {
        currentNode.next = null;
        this.tail = currentNode;
      }
      return true;
    }

    return false;
  }

  *traverse() {
    let current = this.head;
    while (current) {
      yield current.value;
      current = current.next;
    }
  }
}
