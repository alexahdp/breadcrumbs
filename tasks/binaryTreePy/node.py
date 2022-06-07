from typing import Dict, Tuple, Union
import json

"""
TODO
 - Use loops instead of recursion
 - Rid of getDepth: count depth (skewness and update on add, remove and balance)
Link: http://www.proteus2001.narod.ru/gen/txt/6/avl.html
"""

class Node():
  def __init__(self, val: int, 
              left: Union["Node", None] = None,
              right: Union["Node", None] = None):
    self.val = val
    self.left: Node = left
    self.right: Node = right
  

  def __str__(self):
    return json.dumps(self.toJson(), indent=2)

  
  def toJson(self):
    return {
      "val": self.val,
      "left": self.left.toJson() if self.left else None,
      "right": self.right.toJson() if self.right else None,
    }


  @classmethod
  def fromJson(cls, jsondict: Dict) -> "Node":
    def recur(jsondictNode):
      node = Node(jsondictNode["val"])
      if jsondictNode["left"]:
        node.left = recur(jsondictNode["left"])
      if jsondictNode["right"]:
        node.right = recur(jsondictNode["right"])
      return node
    root = recur(jsondict)
    return root


  def isEqual(self, node: "Node") -> bool:
    if self.val and not node or self.val != node.val:
      return False

    if self.left and self.right:
      return self.left.isEqual(node.left) and self.right.isEqual(node.right)
    elif self.left:
      return self.left.isEqual(node.left) and node.right == None
    elif self.right:
      return self.right.isEqual(node.right) and node.left == None
    else:
      return node.left == None and node.right == None


  def get(self, val) -> Union["Node", None]:
    if self.val > val:
      if self.left:
        return self.left.get(val)
      return None
    if self.val < val:
      if self.right:
        return self.right.get(val)
      return None
    return self

  
  def findMin(self) -> "Node":
    if self.left:
      return self.left.findMin(self)
    return self
  

  def findMax(self) -> "Node":
    if self.right:
      return self.right.findMax(self)
    return self


  def add(self, node: "Node") -> bool:
    if node.val < self.val:
      if self.left == None:
        self.left = node
        return True
      else:
        if self.left.add(node):
          self.balance()
          return True
        else:
          return False
    elif node.val > self.val:
      if self.right == None:
        self.right = node
        return True
      else:
        if self.right.add(node):
          self.balance()
          return True
        else:
          return False
    else:
      return False
  

  def remove(self, val, parent: Union["Node", None] = None) -> bool:
    if val < self.val:
      if self.left:
        if self.left.remove(val, self):
          self.balance()
          return True
        return False
      return False
    if val > self.val:
      if self.right:
        if self.right.remove(val, self):
          self.balance()
          return True
        return False
      return False
    # self.val == val:
    if self.left == None and self.right == None:
      if parent.left == self:
        parent.left = None
        return True
      else:
        parent.right = None
        return True
    else:
      leftDepth, rightDepth = self.getDepth()
      if leftDepth > rightDepth:
        maxNode = self.left.findMax()
        self.val = maxNode.val
        self.left.remove(maxNode.val)
        self.left.balance()
        return True
      else:
        minNode = self.right.findMin()
        self.val = minNode.val
        self.right.remove(minNode.val)
        self.right.balance()
        return True


  def getDepth(self) -> Tuple[int, int]:
    leftDepth, rightDepth = 0, 0
    if self.left:
      leftDepth = max(self.left.getDepth()) + 1
    if self.right:
      rightDepth = max(self.right.getDepth()) + 1
    return leftDepth, rightDepth


  def isBalanced(self) -> bool:
    leftDepth, rightDepth = self.getDepth()
    return abs(leftDepth - rightDepth) < 2


  def rotateLeft(self):
    val = self.val
    right = self.right
    
    self.val = self.right.val
    self.right = right.right

    newNode = Node(val)
    newNode.left = self.left
    newNode.right = right.left
    self.left = newNode
  

  def rotateRight(self):
    val = self.val
    left = self.left

    self.val = self.left.val
    self.left = left.left

    newNode = Node(val)
    newNode.right = self.right
    newNode.left = left.right
    self.right = newNode


  def balance(self):
    leftDepth, rightDepth = self.getDepth()
    if abs(leftDepth - rightDepth) >= 2:
      if leftDepth > rightDepth:
        leftDepth, rightDepth = self.left.getDepth()
        if leftDepth < rightDepth:
          self.left.rotateLeft()
        self.rotateRight()
      else:
        leftDepth, rightDepth = self.right.getDepth()
        if leftDepth > rightDepth:
          self.right.rotateRight()
        self.rotateLeft()
