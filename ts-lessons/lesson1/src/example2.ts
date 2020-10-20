// интерфейсы

const states = [
  { name: "Alabama", capitol: "Montgomery" },
  { name: "Alaska", capitol: "Juneau" },
  { name: "Arizona", capitol: "Phoenix" },
];

for (const state of states) {
  console.log(state.capital);
}

type State = {
  name: string;
  capital: string;
}

const states2: State[] = [
  { name: "Alabama", capitol: "Montgomery" },
  { name: "Alaska", capitol: "Juneau" },
  { name: "Arizona", capitol: "Phoenix" },
];

for (const state of states2) {
  console.log(state.capital);
}
