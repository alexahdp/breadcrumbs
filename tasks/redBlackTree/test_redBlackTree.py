import pytest
from node import Node, RED, BLACK

class TestRedBlackTree:
  def test_is_valid_1(self):
    node = Node(1, None)
    assert(node.val == 1)
    assert(node.color == BLACK)
    assert(node.isValid())


  def test_is_valid_3(self):
    node = Node(4)
    node1 = Node(2, node, RED)
    node2 = Node(8, node, RED)
    node.left = node1
    node.right = node2
    assert(node.isValid())


  def test_is_valid_4(self):
    node = Node(4)
    node1 = Node(2, node, BLACK)
    node2 = Node(8, node, RED)
    node.left = node1
    node.right = node2

    with pytest.raises(Exception):
      node.checkBlackHeight()

  def test_add_1(self):
    node = Node(5)
    node.add(10)
    node.add(2)

    assert(node.val == 5)
    assert(node.color == BLACK)
    assert(node.left.val == 2)
    assert(node.left.color == RED)
    assert(node.right.val == 10)
    assert(node.right.color == RED)
    assert(node.isValid())


  def test_add_2(self):
    node = Node(15, None, BLACK)
    for v in [9, 1, 4, 3, 19, 2, 4, 38, 20, 44, 12, 19, 0]:
      node.add(v)
      assert(node.isValidBinaryTree())
      assert(node.isValid())
      assert(node.checkBlackHeight())


  def test_rotate_right(self):
    node = Node(5, BLACK)
    node2 = Node(3, node, RED)
    node3 = Node(1, node, RED)
    node4 = Node(8, node3, BLACK)

    node.left = node2
    node2.left = node3
    node.right = node4

    node.rotateRight()

    assert(node.val == 3)
    assert(node.color == BLACK)

    assert(node.left.val == 1)
    assert(node.left.color == RED)
    assert(node.left.left == None)
    assert(node.left.right == None)

    assert(node.right.val == 5)
    assert(node.right.color == RED)
    assert(node.right.left == None)

    assert(node.right.right.val == 8)
    assert(node.right.right.color == BLACK)
    assert(node.right.right.left == None)
    assert(node.right.right.right == None)
    assert(node.isValid())


  def test_rotate_left(self):
    node = Node(3, BLACK)
    node2 = Node(1, node, BLACK)
    node3 = Node(5, node, BLACK)
    node4 = Node(8, node3, RED)

    node.left = node2
    node.right = node3
    node3.right = node4

    node.rotateLeft()
    
    assert(node.val == 5)
    assert(node.color == BLACK)

    assert(node.left.val == 3)
    assert(node.left.color == RED)
    assert(node.left.right == None)

    assert(node.left.left.val == 1)
    assert(node.left.left.color == BLACK)
    assert(node.left.left.left == None)
    assert(node.left.left.right == None)

    assert(node.right.val == 8)
    assert(node.right.color == RED)
    assert(node.right.left == None)
    assert(node.right.right == None)
    assert(node.isValid())


  def test_fix_insert_1(self):
    node = Node(15, None, BLACK)
    node1 = Node(10, node, RED)
    node.left = node1
    
    node1.fixInsert()
    
    assert(node.val == 15)
    assert(node.color == BLACK)
    
    assert(node.left.val == 10)
    assert(node.left.color == RED)

    assert(node.isValidBinaryTree())
    assert(node.isValid())
    assert(node.checkBlackHeight())


  # rotation
  def test_fix_insert_2(self):
    node = Node(15, None, BLACK)
    node1 = Node(10, node, RED)
    node2 = Node(5, node1, RED)
    node.left = node1
    node1.left = node2
    node2.fixInsert()
    
    assert(node.val == 10)
    assert(node.color == BLACK)
    
    assert(node.left.val == 5)
    assert(node.left.color == RED)
    
    assert(node.right.val == 15)
    assert(node.right.color == RED)

    assert(node.isValidBinaryTree())
    assert(node.isValid())
    assert(node.checkBlackHeight())


  def test_fix_insert_3(self):
    node = Node(15, None, BLACK)
    node1 = Node(20, node, RED)
    node2 = Node(25, node1, RED)
    node.right = node1
    node1.right = node2
    node2.fixInsert()
    
    assert(node.val == 20)
    assert(node.color == BLACK)
    
    assert(node.left.val == 15)
    assert(node.left.color == RED)
    
    assert(node.right.val == 25)
    assert(node.right.color == RED)

    assert(node.isValidBinaryTree())
    assert(node.isValid())
    assert(node.checkBlackHeight())


  def test_fix_insert_4(self):
    node = Node(10, None, BLACK)
    node1 = Node(5, node, RED)
    node2 = Node(3, node1, RED)
    node3 = Node(25, node, RED)
    node.left = node1
    node1.left = node2
    node.right = node3
    
    node2.fixInsert()
    
    assert(node.val == 10)
    assert(node.color == BLACK)
    
    assert(node.left.val == 5)
    assert(node.left.color == BLACK)

    assert(node.left.left.val == 3)
    assert(node.left.left.color == RED)

    assert(node.right.val == 25)
    assert(node.right.color == BLACK)

    assert(node.isValidBinaryTree())
    assert(node.isValid())
    assert(node.checkBlackHeight())


  # Almost the same as previous
  def test_fix_insert_5(self):
    node = Node(10, None, BLACK)
    node1 = Node(5, node, BLACK)
    node2 = Node(3, node1, RED)
    node3 = Node(15, node, RED)
    node4 = Node(12, node3, BLACK)
    node5 = Node(11, node4, RED)
    node6 = Node(13, node4, RED)
    node7 = Node(1, node2, RED)

    node.left = node1
    node1.left = node2
    node.right = node3
    node3.left = node4
    node4.left = node5
    node4.right = node6
    node2.left = node7
    
    node7.fixInsert()
    
    assert(node.val == 10)
    assert(node.color == BLACK)
    
    assert(node.left.val == 3)
    assert(node.left.color == BLACK)

    assert(node.left.left.val == 1)
    assert(node.left.left.color == RED)

    assert(node.left.right.val == 5)
    assert(node.left.right.color == RED)
    
    assert(node.right.val == 15)
    assert(node.right.color == RED)

    assert(node.right.left.val == 12)
    assert(node.right.left.color == BLACK)

    assert(node.right.left.left.val == 11)
    assert(node.right.left.left.color == RED)

    assert(node.right.left.right.val == 13)
    assert(node.right.left.right.color == RED)

    assert(node.isValidBinaryTree())
    assert(node.isValid())
    assert(node.checkBlackHeight())


  def test_fix_insert_6(self):
    node = Node(10, None, BLACK)
    node1 = Node(5, node, BLACK)
    node2 = Node(8, node1, RED)
    node3 = Node(7, node2, RED)
    node4 = Node(12, node, BLACK)

    node.left = node1
    node1.right = node2
    node2.left = node3
    node.right = node4
    
    node3.fixInsert()
    
    assert(node.val == 10)
    assert(node.color == BLACK)
    
    assert(node.left.val == 7)
    assert(node.left.color == BLACK)

    assert(node.left.left.val == 5)
    assert(node.left.left.color == RED)

    assert(node.left.right.val == 8)
    assert(node.left.right.color == RED)
    
    assert(node.right.val == 12)
    assert(node.right.color == BLACK)

    assert(node.isValidBinaryTree())
    assert(node.isValid())
    assert(node.checkBlackHeight())
