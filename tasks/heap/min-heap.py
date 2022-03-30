class MinHeap:
  def __init__(self, heapSize):
    self._size = heapSize
    self.heap = [0] * (heapSize + 1)
    self.last = 0;
  
  def add(self, elem):
    self.last += 1
    if self.last > self._size:
      self.last -= 1
      raise Exception('heap size overflow')
    
    self.heap[self.last] = elem
    
    index = self.last
    while index > 1:
      parent = index // 2
      if self.heap[parent] > self.heap[index]:
        self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
        index = parent
      else:
        break

  def peek(self):
    return self.heap[1]
    
  def pop(self):
    if self.last < 1:
      raise Exception("heap is empty")
    
    result = self.heap[1]
    
    self.heap[1] = self.heap[self.last]
    self.last -= 1
    
    index = 1
    while index <= self.last // 2:
      left = index * 2
      right = left + 1
      if self.heap[index] > self.heap[left] or self.heap[index] > self.heap[right]:
        if self.heap[left] > self.heap[right]:
          self.heap[right], self.heap[index] = self.heap[index], self.heap[right]
          index = right
        else:
          self.heap[left], self.heap[index] = self.heap[index], self.heap[left]
          index = left
      else:
        break
    
    return result
  
  def size(self):
    return self.last
  
  def __str__(self):
    return str(self.heap)
