## Rotate image

[Leetcode](https://leetcode.com/problems/rotate-image/)  

#### Solution 1
```javascript
function transpose(matrix) {
  // проход по строкам
  for (let j = 0; j < matrix.length; j++) {
    // проходим только по индексам элементов выше главной диагонали
      // элементы на диагонали не надо менять
      // если будем идти выше и ниже диагонали - будем переставлять туда-обратно
    for (let i = j + 1; i < matrix[j].length; i++) {
      // меняем местами элементы [i, j] и [j, i]
      const box = matrix[j][i];
      matrix[j][i] = matrix[i][j];
      matrix[i][j] = box;
    }
  }
}
function reverse(matrix) {
  for (let j = 0; j < matrix.length; j++) {
    let start = 0;
    let end = matrix[j].length - 1;
    while (start < end) {
      const box = matrix[j][start];
      matrix[j][start] = matrix[j][end];
      matrix[j][end] = box;
      start++;
      end--;
    }
  }
}

transpose(matrix);
reverse(matrix);
```

#### Solution 2
```javascript
function rotate(matrix) {
  // будем вращать элементы по квадратам
  const n = matrix.length;
  for (let i = 0; i < Math.floor(n / 2) + n % 2; i++) {
    for (let j = 0; j < Math.floor(n / 2); j++) {
      const tmp = matrix[n - 1 - j][i];
      matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
      matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
      matrix[j][n - 1 - i] = matrix[i][j]
      matrix[i][j] = tmp
    }
  }
}
rotate(matrix);
```
