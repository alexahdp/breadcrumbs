const points = [
  [10, 10],
  [10, 20],
  [20, 10],
  [20, 20],
  [30, 10],
  [30, 20],
];

// what about duplicates in points?
// assume that they are absent

/*
build xMap Map<x, y[]> and yMap Map<y, x[]>
iterate over all points
  exist points with the same y and another x and x2 > x1
    iterate a:
      exist point with the same x and another y = b and y2 > y1
        iterate b:
          exist point with a.x and b.y
            then this is rect
*/
const getRects = (points) => {
  const rects = [];
  // build xMap Map<x, y[]> and yMap Map<y, x[]>
  const xMap = new Map();
  const yMap = new Map();

  for (const point of points) {
    if (!xMap.has(point[0])) xMap.set(point[0], []);
    if (!yMap.has(point[1])) yMap.set(point[1], []);
    xMap.get(point[0]).push(point[1]);
    yMap.get(point[1]).push(point[0]);
  }

  // iterate over all points
  for (const point of points) {
    //   exist points with the same y and another x and x2 > x1
    if (yMap.has(point[1])) {
      for (const x2 of yMap.get(point[1])) {
        if (point[0] >= x2) continue;

        // exist point with the same x and another y = b and y2 > y1
        if (xMap.has(point[0])) {
          for (const y3 of xMap.get(point[0])) {
            if (point[1] >= y3) continue;

            // exist point with a.x and b.y
            // then this is rect
            if (xMap.has(x2)) {
              const y4 = xMap.get(x2).find((y) => y === y3);
              if (y4) {
                rects.push([point, [x2, point[1]], [point[0], y3], [x2, y4]]);
              }
            }
          }
        }
      }
    }
  }
  return rects.length;
};

const test = (a, b) => {
  a === b ? console.log("ok") : console.log("no");
};

test(
  getRects([
    [10, 10],
    [10, 20],
    [20, 10],
    [20, 20],
    [30, 10],
    [30, 20],
  ]),
  3
);

test(
  getRects([
    [10, 10],
    [10, 20],
    [20, 10],
    [20, 20],
    [30, 10],
    [30, 20],
    [40, 10],
    [40, 20],
  ]),
  6
);
