//   https://github.com/adeo/lmru-bpms-ved--user-task-creator/blob/develop/src/mappings/taskTemplateAndEntityDocTypesMapping.js

enum Role {
  CertBody = 'CertBody'
}

type TaskType = string;

interface Section {
  caption: TaskType;
  fields: {
      type: string;
      docs: {type: string}[];
  }[]
}

interface Task {
  id: string;
  name: string;
  type: string;
  role: Role;
  roleName: string;
  sections: [Section];
}

const exampleTask: Task = {
  id: '11',
  name: 'hello',
  type: 'task',
  role: Role.CertBody,
  roleName: 'df',
  sections: [{
      caption: 'Task',
      fields: [{
          type: 'docList',
          docs: [{type: 'doc'}],
      }]
  }]
};

const getDocTypesFromTask = (task: Task): string[] | void => {
  const section = task.sections.find(x => x.caption === "Task")
  if (section) {
      const field = section.fields.find(x => x.type === "docList")
      if (field) {
          return field.docs.map(doc => doc.type);
      }
  }
};

getDocTypesFromTask(exampleTask);
