import BinaryTree from './BinaryTree.js';

const delay = t => new Promise(res => setTimeout(res, 100));

const data = {
    csv: null,
    selected: null,
    settings: {
        strokeColor: "#29B5FF",
        width: 100
    }
};

(async function() {
  const testData = [];
  const tree = new BinaryTree();

  tree.add(50, true);
  data.csv = tree.exportCSV();

  for (let i = 0; i < 30; i++) {
    const val = Math.round(Math.random() * 100);
    testData.push(val);
    tree.add(val, true);
    tree.validate();
    data.csv = tree.exportCSV();
    await delay();
  }

  const testData2 = _.shuffle(testData);
  for (let b of testData2) {
    tree.remove(b, true);
    tree.validate();
    data.csv = tree.exportCSV();
    await delay();
  }
})()

new Vue({
  el: "#app",
  data: function () {
    return data;
  },
  computed: {
    root: function () {
      if (this.csv) {
        const root = d3.stratify()
          .id(function (d) { return d.node; })
          .parentId(function (d) { return d.parent; })
          (this.csv);
        return this.tree(root);
      }
    },
    tree: function () {
      return d3
        .tree()
        .size([600, this.settings.width - 300]);
    },
    nodes: function () {
      const that = this;
      if (this.root) {
        return this.root.descendants().map(function (d) {
          let x = (200 + d.x) + "px";
          let y = parseInt(-1 * d.y + 30) + "px";
          return {
            id: d.id,
            r: 2.5,
            text: d.id,
            style: {
              transform: "translate(" + x + "," + y + ")"
            },
            textpos: {
              x: d.children ? -8 : 8,
              y: 3
            },
            textStyle: {
              textAnchor: d.children ? "end" : "start"
            }
          };
        });
      }
    },
    links: function () {
      let that = this;
      if (this.root) {
        return this.root.descendants().slice(1).map(function (d) {
          let x = d.x + 200, parent_x = d.parent.x + 200;
          let y = parseInt(-1 * d.y + 30);
          let parent_y = parseInt(-1 * d.parent.y + 30);
          return {
            id: d.id,
            d: "M" + x + "," + y + "L" + parent_x + "," + parent_y,
            style: {
                stroke: that.settings.strokeColor
            }
          };
        });
      }
    }
  },
  methods: {
    select: function (index, node) {
      this.selected = index;
    }
  }
});