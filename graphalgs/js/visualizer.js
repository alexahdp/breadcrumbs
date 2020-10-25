const dist = ({x: x1, y: y1}, {x: x2, y: y2}) => Math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2);

export class Visualizer {
  constructor(renderer, rootNode) {
    this.renderer = renderer;
    this.rootNode = rootNode;
  }

  *animator() {
    // создать массив с курсорами
    // в цикле: пока массив курсоров не опустеет:
    //   определить массив для действий
    //   если задан вектор движения
    //      - передвинуться - положить действие в массив
    //   иначе
    //      - создать курсоры для всех выходных элементов
    //      - удалить старый курсор из массива курсоров
    //      - положить новые курсоры в массив курсоров
    //   выдать массив действий в итератор

    let cursors = [{
      node: this.rootNode,
      iterator: null,
      done: true,
    }];

    while (cursors.length > 0) {
      const actions = [];

      for (let i = 0; i < cursors.length; i++) {
        // cursor.done - итератор завершился
        // нулевой курсор инициализируется как завершенный итератор
        const cursor = cursors[i];

        if (cursor.done) {
          cursors.splice(i, 1);
          const newCursors = cursor.node.to.map(line => ({
            node: line.to,
            iterator: this.animatorEach(cursor.node, line.to),
            done: false,
          }));
          cursors.push(...newCursors);
          i--;
          continue;
        }

        const res = cursor.iterator.next();
        if (res.done) {
          cursor.done = true;
        } else {
          actions.push(res.value);
        }
      }

      yield actions;
    }
  }

  *animatorEach(from, to) {
    const vec = { x: to.x - from.x, y: to.y - from.y };
    const len = Math.sqrt(vec.x ** 2 + vec.y ** 2);
    vec.x = vec.x / len;
    vec.y = vec.y / len;

    const dot = {
      x: from.x,
      y: from.y,
      r: 10
    };

    do {
      dot.x += vec.x;
      dot.y += vec.y;
      yield {
        node: dot,
        opts: { color: 'red', stroke:  "#FF0000"},
      };
    } while(dist(dot, to) > 5);
  }
}