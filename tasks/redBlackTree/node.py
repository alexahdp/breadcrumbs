from typing import Union
import json

RED = 0
BLACK = 1

# root is black
# each red node has two black children
# all paths to all leaves must contain similar number of black nodes
class Node():
  def __init__(self, val: int,
              parent: Union["Node", None] = None,
              color: Union["RED", "BLACK"] = BLACK):
    self.val: int = val
    self.color: Union["RED", "BLACK"] = color
    self.left: Union["Node", None] = None
    self.right: Union["Node", None] = None
    self.parent: Union["Node", None] = parent


  def __str__(self):
    return json.dumps(self.toJson(), indent=2)

  
  def toJson(self):
    return {
      "val": self.val,
      "color": self.color,
      "left": self.left.toJson() if self.left else None,
      "right": self.right.toJson() if self.right else None,
    }


  def find(self, val: int) -> Union["Node", None]:
    currentNode = self
    while currentNode:
      if val < currentNode.val:
        currentNode = currentNode.left
      if val > currentNode.val:
        currentNode = currentNode.right
      return currentNode
    return None


  def findMin(self) -> "Node":
    currentNode = self
    while currentNode.left:
      currentNode = currentNode.left
    return currentNode


  def findMax(self) -> "Node":
    currentNode = self
    while currentNode.right:
      currentNode = currentNode.right
    return currentNode


  def add(self, val):
    prevNode = None
    currentNode = self
    while currentNode:
      prevNode = currentNode
      if val < currentNode.val:
        currentNode = currentNode.left
      elif val > currentNode.val:
        currentNode = currentNode.right
      else:
        return
    parentNode = prevNode
    newNode = Node(val, parentNode, RED)
    
    if val < parentNode.val:
      parentNode.left = newNode
    else:
      parentNode.right = newNode
    
    if parentNode.color == BLACK:
      return
    
    newNode.fixInsert()


  def isValid(self) -> bool:
    isRoot = False
    if self.parent == None:
      isRoot = True

    if isRoot:
      if self.color != BLACK:
        raise Exception("Wrong root color")
    if self.color == RED:
      if self.left:
        if self.left.color != BLACK:
          raise Exception("Wrong left color")
        else:
          self.left.isValid()
      if self.right:
        if self.right.color != BLACK:
          raise Exception("Wrong right color")
        else:
          self.right.isValid()
    elif self.left:
      self.left.isValid()
    elif self.right:
      self.right.isValid()
    return True


  def isValidBinaryTree(self):
    if self.left:
      if self.left.val >= self.val:
        raise Exception("Invalid binary tree")
      self.left.isValidBinaryTree()
    if self.right:
      if self.right.val <= self.val:
        raise Exception("Invalid binary tree")
      self.right.isValidBinaryTree()
    return True
  

  def checkBlackHeight(self):
    self._getBlackHeight()
    return True


  def _getBlackHeight(self):
    currentColor = 1 if self.color == BLACK else 0
    if self.left and self.right:
      leftH = self.left._getBlackHeight()
      rightH = self.right._getBlackHeight()
      if leftH != rightH:
        raise Exception("Tree is skewed")
      return leftH + currentColor
    elif not self.right and self.left:
      return self.left._getBlackHeight() + currentColor
    elif not self.left and self.right:
      return self.right._getBlackHeight() + currentColor
    else:
      return currentColor


  def rotateRight(self):
    val = self.val
    moved = self.left
    right = self.right
    movedRight = moved.right
    movedLeft = moved.left
    
    self.val = moved.val
    moved.val = val
    self.right = moved
    moved.parent = self
    moved.color = RED
    moved.right = right
    if right:
      right.parent = moved
    moved.left = movedRight
    if movedRight:
      movedRight.parent = moved
    self.left = movedLeft
    if movedLeft:
      movedLeft.parent = self
    self.color = BLACK


  def rotateLeft(self):
    val = self.val
    moved = self.right
    left = self.left
    movedLeft = moved.left if moved else None
    movedRight = moved.right
    
    self.val = moved.val
    moved.val = val
    self.left = moved
    moved.parent = self
    moved.color = RED
    moved.left = left
    if left:
      left.parent = moved
    moved.right = movedLeft
    if movedLeft:
      movedLeft.parent = moved
    self.right = movedRight
    if movedRight:
      movedRight.parent = self
    self.color = BLACK


  def fixInsert(self):
    t = self
    if not t.parent:
      t.color = BLACK
      return
    while t.parent and t.parent.color == RED:
      parent = t.parent
      grandParent = parent.parent
      if grandParent.left == parent:
        if grandParent.right and grandParent.right.color == RED:
          parent.color = BLACK
          grandParent.right.color = BLACK
          grandParent.color = RED
          t = grandParent
        else:
          if t == parent.right:
            t = parent
            t.rotateLeft()
          parent.color = BLACK
          grandParent.color = RED
          grandParent.rotateRight()
          t = grandParent
      else:
        if grandParent.left and grandParent.left.color == RED:
          parent.color = BLACK
          grandParent.left.color = BLACK
          grandParent.color = RED
          t = grandParent
        # elif grandParent.left == None:
        else:
          if t == parent.left:
            t = parent
            t.rotateRight()
          parent.color = BLACK
          grandParent.color = RED
          # if t.right:
          grandParent.rotateLeft()
          t = grandParent
    if not t.parent:
      t.color = BLACK


  # def remove(self, val):
  #   stack = []
  #   currentNode = self
  #   while True:
  #     if val < currentNode.val:
  #       if currentNode.left:
  #         stack.append(currentNode)
  #         currentNode = currentNode.left
  #       else:
  #         return
  #     elif val> currentNode.val:
  #       if currentNode.right:
  #         stack.append(currentNode)
  #         currentNode = currentNode.right
  #       else:
  #         return
  #     else:
  #       break
  #   if currentNode.left == None and currentNode.right == None:
  #     if currentNode.color == RED:
  #       # TODO: root
  #       parentNode = stack.pop()
  #       if parentNode.left == currentNode:
  #         parentNode.left = None
  #       else:
  #         parentNode.right = None
  #   elif currentNode.left == None and currentNode.right != None:
  #     currentNode.val = currentNode.right.val
  #     currentNode.left = currentNode.right.left
  #     currentNode.right = currentNode.right.right
  #   elif currentNode.left != None and currentNode.right == None:
  #     currentNode.val = currentNode.left.val
  #     currentNode.right = currentNode.left.right
  #     currentNode.left = currentNode.left.left
  #   else:
  #     maxNode = currentNode.findMax()
  #     maxNode.val, currentNode.val = currentNode.val, maxNode.val
  #     # there are extra steps here
  #     # it is safe, but may be optimized
  #     currentNode.remove(maxNode.val)
