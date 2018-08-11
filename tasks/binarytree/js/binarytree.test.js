const chai = require('chai');
const expect = chai.expect;

const Node = require('./binaryNode.js');

describe('', () => {
  it('constructor', () => {
    const node = new Node(10);
    expect(node.val).to.be.equal(10);
    expect(node.next).to.be.null;
    expect(node.prev).to.be.null;
    expect(node.parent).to.be.null;
  });

  it('constructor with parent', () => {
    const node = new Node(10, 'parent');
    expect(node.val).to.be.equal(10);
    expect(node.next).to.be.null;
    expect(node.prev).to.be.null;
    expect(node.parent).to.be.equal('parent');
  });

  it('replace', () => {
    const node = new Node(20);
    const tree = {
      root: node
    };
    node.root = tree;
    
    const newNode = new Node(10);
    node.replace(newNode, 'prev');
    expect(tree.root.val).to.be.equal(10);
    expect(tree.root.prev.val).to.be.equal(20);
    expect(tree.root.next).to.be.null;
    expect(tree.root.prev.prev).to.be.null;
    expect(tree.root.prev.next).to.be.null;
    expect(tree.root.prev.parent.val).to.be.equal(10);
    expect(tree.root.parent).to.be.null;
  });

  it('replace with parent link', () => {
    const node = new Node(20);
    const tree = { root: node };
    node.root = tree;
    
    node.add(10);

    const newNode = new Node(15);
    node.prev.replace(newNode, 'prev');

    expect(tree.root.val).to.be.equal(20);
    expect(tree.root.prev.val).to.be.equal(15);
    expect(tree.root.prev.prev.val).to.be.equal(10);

    expect(tree.root.next).to.be.null;
    expect(tree.root.prev.next).to.be.null;
    expect(tree.root.prev.parent.val).to.be.equal(20);
    expect(tree.root.prev.prev.parent.val).to.be.equal(15);

    expect(tree.root.parent).to.be.null;
  });

  it('full node replace', () => {
    const node = new Node(20);
    const tree = { root: node };
    node.root = tree;
    
    node.add(10);

    const node2 = new Node(15);

    node.replace(node2, 'next', true);

    expect(tree.root.val).to.be.equal(15);
    expect(tree.root.prev).to.be.null;
    expect(tree.root.next.val).to.be.equal(20);
    expect(tree.root.next.parent.val).to.be.equal(15);
    expect(tree.root.next.prev.val).to.be.equal(10);
    expect(tree.root.next.next).to.be.null;

    expect(tree.root.parent).to.be.null;
  });

  it('replace by node', () => {
    const node = new Node(20);
    const tree = { root: node };
    node.root = tree;
    
    node.add(10);

    expect(node.prev.parent).to.be.equal(node);

    const node2 = new Node(5);
    node.prev.replace(node2);

    expect(tree.root.val).to.be.equal(20);
    expect(tree.root.prev.val).to.be.equal(5);
    expect(tree.root.prev.parent).to.be.equal(node);
    expect(tree.root.prev.prev).to.be.null;
    expect(tree.root.prev.next).to.be.null;
    expect(tree.root.parent).to.be.null;
  });

  it('add left', () => {
    const node = new Node(10);
    const tree = { root: node };
    node.root = tree;
    
    node.add(20);
    expect(node.val).to.be.equal(10);
    expect(node.prev).to.be.null;
    expect(node.next.val).to.be.equal(20);
    expect(node.next.parent).to.be.equal(node);

    node.add(5);
    expect(node.val).to.be.equal(10);
    expect(node.prev.val).to.be.equal(5);
    expect(node.prev.parent).to.be.equal(node);
    expect(node.next.val).to.be.equal(20);
    expect(node.next.parent).to.be.equal(node);

    node.add(7);
    expect(tree.root).to.be.equal(node);
    expect(tree.root.val).to.be.equal(10);
    expect(node.prev.parent.val).to.be.equal(node.val);
    expect(node.next.val).to.be.equal(20);
    expect(node.next.parent).to.be.equal(node);
  });

  it('add right', () => {
    const node = new Node(10);
    const tree = { root: node };
    node.root = tree;
    
    node.add(20);
    expect(node.val).to.be.equal(10);
    expect(node.prev).to.be.null;
    expect(node.next.val).to.be.equal(20);
    expect(node.next.parent).to.be.equal(node);

    node.add(30);
    expect(tree.root).to.be.equal(node);
    expect(node.root).to.be.equal(tree);
    expect(node.val).to.be.equal(10);
    expect(node.next.val).to.be.equal(20);
    expect(node.next.parent).to.be.equal(node);
    expect(node.next.next.val).to.be.equal(30);
    expect(node.next.next.parent).to.be.equal(node.next);

    node.add(15);
    expect(tree.root).to.be.equal(node);
    expect(node.root).to.be.equal(tree);
    expect(node.val).to.be.equal(10);
    expect(node.next.parent).to.be.equal(node);
    expect(node.next.next.parent).to.be.equal(node.next);
  });

  it('add with balance', () => {
    const node = new Node(10);
    const tree = { root: node };
    node.root = tree;

    node.add(9, true);
    node.add(21, true);
    node.add(11, true);
    node.add(121, true);

    expect(tree.root.val).to.be.equal(10);
    expect(tree.root.prev.val).to.be.equal(9);
    expect(tree.root.prev.parent).to.be.equal(tree.root);
    expect(tree.root.next.val).to.be.equal(21);
    expect(tree.root.next.parent).to.be.equal(tree.root);
    expect(tree.root.next.prev.val).to.be.equal(11);
    expect(tree.root.next.prev.parent).to.be.equal(tree.root.next);
    expect(tree.root.next.next.val).to.be.equal(121);
    expect(tree.root.next.next.parent).to.be.equal(tree.root.next);
  });

  it('leftRotation', () => {
    const node = new Node(30);
    const tree = { root: node };
    node.root = tree;

    node.add(20);
    node.add(10);

    node.leftRotation();

    expect(tree.root.val).to.be.equal(20);
    expect(tree.root.prev.val).to.be.equal(10);
    expect(tree.root.next.val).to.be.equal(30);

    expect(tree.root.prev.parent).to.be.equal(tree.root);
    expect(tree.root.next.parent).to.be.equal(tree.root);

    expect(tree.root.prev.prev).to.be.null;
    expect(tree.root.prev.next).to.be.null;
    expect(tree.root.next.prev).to.be.null;
    expect(tree.root.next.next).to.be.null;

    expect(tree.root.root).to.be.equal(tree);
    expect(tree.root.next.root).to.be.equal(null);
  });

  it('rightRotation', () => {
    const node = new Node(10);
    const tree = { root: node };
    node.root = tree;

    node.add(20);
    node.add(30);

    node.rightRotation();

    expect(tree.root.val).to.be.equal(20);
    expect(tree.root.prev.val).to.be.equal(10);
    expect(tree.root.next.val).to.be.equal(30);

    expect(tree.root.prev.parent).to.be.equal(tree.root);
    expect(tree.root.next.parent).to.be.equal(tree.root);

    expect(tree.root.prev.prev).to.be.null;
    expect(tree.root.prev.next).to.be.null;
    expect(tree.root.next.prev).to.be.null;
    expect(tree.root.next.next).to.be.null;

    expect(tree.root.root).to.be.equal(tree);
    expect(tree.root.next.root).to.be.equal(null);
  });

});