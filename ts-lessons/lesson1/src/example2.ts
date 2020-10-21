// === 1 ===
// объединение типов
const successStatus = 'success';
type Status = 'success' | 'fail';

const doOperation = (data: string): Status => {
  console.log(data);
  // ...
  return 'success';
};
const result: Status = doOperation('hello');


// ===  2 ===
// сужение типов

type numstr = string | number;

type Candidate = {
    id: string;
    experience: string;
}

type Vacancy = {
    id: string;
    salary: number;
}

type CandidateOrVacancy = Candidate | Vacancy

function isCandidate(candidate: CandidateOrVacancy): candidate is Candidate {
    return 'experience' in candidate;
}

function toStr(entity: CandidateOrVacancy): string {
  if (isCandidate(entity)) {
      return entity.experience;
  } else {
      entity
  }
  return 'unknown';
}
