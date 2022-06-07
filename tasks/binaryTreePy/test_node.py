from binaryTree.node import Node

class TestNode:
  def test_create_node_1(self):
    node = Node(10)
    assert(node.val == 10)
    assert(node.left == None)
    assert(node.right == None)
  
  def test_create_node_2(self):
    node = Node(10)
    node2 = Node(15)
    node.add(node2)
    assert(node.right == node2)
    assert(node.right.val == 15)
  
  def test_create_node_3(self):
    node = Node(10)
    node2 = Node(5)
    node.add(node2)
    assert(node.left == node2)
    assert(node.left.val == 5)

  def test_get_1(self):
    btree = Node(5)
    btree.add(Node(10))
    node = btree.get(10)
    assert(node.val == 10)
  
  def test_get_2(self):
    btree = Node(5)
    btree.add(Node(10))
    node = btree.get(5)
    assert(node.val == 5)
    node = btree.get(6)
    assert(node == None)

  def test_get_depth(self):
    node = Node(10)
    node2 = Node(12)
    node.add(node2)
    leftDepth, rightDepth = node.getDepth()
    assert((leftDepth, rightDepth) == (0, 1))
    node3 = Node(15)
    node2.add(node3)
    leftDepth, rightDepth = node.getDepth()
    assert((leftDepth, rightDepth) == (0, 2))
    node4 = Node(5)
    node.add(node4)
    leftDepth, rightDepth = node.getDepth()
    assert((leftDepth, rightDepth) == (1, 2))
  
  def test_rotate_left(self):
    node = Node(10)
    node2 = Node(12)
    node.add(node2)
    node3 = Node(15)
    node2.add(node3)
    node.rotateLeft()
    assert(node.val == 12)
    assert(node.left.val == 10)
    assert(node.right.val == 15)
    assert(node.right.right == None)

  def test_is_valid(self):
    btree = Node(10)
    for n in [14, 8, 3, 19, 2, 5, 22, 16, 9, 4]:
      newnode = Node(n)
      btree.add(newnode)
      assert(btree.isBalanced() == True)
  
  def test_remove(self):
    btree = Node(10)
    for n in [14, 8, 3, 19]:
      newnode = Node(n)
      btree.add(newnode)
    for n in [19, 14, 8, 3]:
      btree.remove(19)
      node = btree.get(19)
      assert(node == None)
      assert(btree.isBalanced() == True)
  
  def test_is_equal(self):
    def gen():
      btree = Node(10)
      for n in [14, 8, 3, 19]:
        newnode = Node(n)
        btree.add(newnode)
      return btree
    tree1 = gen()
    tree2 = gen()
    assert(tree1.isEqual(tree2))
    tree2.remove(3)
    assert(not tree1.isEqual(tree2))

  def test_from_json(self):
    jsonTree = {
      "val": 10,
      "left": {
        "val": 5,
        "left": None,
        "right": None,
      },
      "right": {
        "val": 15,
        "left": None,
        "right": None
      },
    }
    btree2 = Node.fromJson(jsonTree)

    btree = Node(10)
    btree.add(Node(5))
    btree.add(Node(15))
    assert(btree.isEqual(btree2))