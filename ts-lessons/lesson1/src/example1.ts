// union-типы
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
