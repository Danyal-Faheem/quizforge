export interface Question {
  question: string;
  options: string[];
  answer: string | string[]; // Single answer or multiple correct answers
  explanation: string;
  hint: string;
}
