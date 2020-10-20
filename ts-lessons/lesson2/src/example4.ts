// Tuple

let v: [number, number] = [1, 5];
let v3: [number, number, number] = [1, 5, 5];
let numbers: number[] = [1,6,7,3,6,9,2];

v = v3; // error (length)
v = numbers; // error (0 and 1 elements could not exist)
v = [5,7]; // ok