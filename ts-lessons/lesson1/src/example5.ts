// структурная типизация

interface Vector {
  x: number;
  y: number;
}

interface NamedVector {
  name: string;
  x: number;
  y: number;
}

const getLength = (v: Vector) => Math.sqrt(v.x^2 + v.y^2);

const normalize = (v: Vector) => {
  const len = getLength(v);
  return { x: v.x / len, y: v.y / len };
}

const beamVector: NamedVector = {
  name: 'sun',
  x: 0,
  y: 0,
};
normalize(beamVector);


// === 6 ===
// ОШИБКА!!!
interface Vector3D {
  x: number;
  y: number;
  z: number;
}
const normalize3D = (v: Vector3D) => {
  const len = getLength(v);
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

// следует заметить, что по причине того, что ts имеет структурную типизацию и не запрещает передачу
// дополнительных свойств, возникает масса проблем с Object.keys(), Object.values()