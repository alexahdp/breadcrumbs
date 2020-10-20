// conjunction-types or intersection types

type CandidateStatus = 'new' | 'inProgress' | 'applied';

interface Profile {
  id: string;
  email: string;
}

interface Candidate {
  vacancyId: string;
  status: CandidateStatus; 
}

const getCandidate = (): Profile & Candidate => ({
  id: '123',
  email: 'alexgmail.com',
  vacancyId: '1234',
  status: 'new',
});

const candidate = getCandidate();
