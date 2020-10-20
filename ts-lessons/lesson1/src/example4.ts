// проверка на принадлежность типу

interface Square {
  width: number;
}
interface Rectangle extends Square {
  height: number;
}

type Shape = Square | Rectangle;

function calculateArea(shape: Shape) {
  if (shape instanceof Rectangle) {
    return shape.width * shape.width;
  } else {
    return shape.width * shape.width;
  }
}

// === 2 ===

class Square {
  constructor(public width: number) {}
  
}
class Rectangle extends Square {
  constructor(public width: number, public height: number) {
    super(width);
  }
}

type Shape = Square | Rectangle;

function calculateArea2(shape: Shape) {
  if (shape instanceof Rectangle) {
    return shape.width * shape.width;
  } else {
    return shape.width * shape.width;
  }
}