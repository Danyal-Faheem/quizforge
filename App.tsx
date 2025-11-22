import React, { useState, useCallback } from 'react';
import { QUIZ_DATA } from './constants';
import QuestionCard from './components/QuestionCard';
import QuizSummary from './components/QuizSummary';
import ProgressBar from './components/ProgressBar';
import QuizSidebar from './components/QuizSidebar';

const App: React.FC = () => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [score, setScore] = useState<number>(0);
  const [quizFinished, setQuizFinished] = useState<boolean>(false);
  const [answeredQuestions, setAnsweredQuestions] = useState<Set<number>>(new Set());
  const [userAnswers, setUserAnswers] = useState<Map<number, string[]>>(new Map());

  const handleAnswer = useCallback((selectedOptions: string[]) => {
    const questionAnswer = QUIZ_DATA[currentQuestionIndex].answer;
    const correctAnswers = Array.isArray(questionAnswer) ? questionAnswer : [questionAnswer];
    
    // Store user's answer
    setUserAnswers(prev => new Map(prev).set(currentQuestionIndex, selectedOptions));
    
    // Check if all selected options are correct and all correct answers are selected
    const allSelectedCorrect = selectedOptions.every(opt => correctAnswers.includes(opt));
    const allCorrectSelected = correctAnswers.every(ans => selectedOptions.includes(ans));
    const isCorrect = allSelectedCorrect && allCorrectSelected && selectedOptions.length === correctAnswers.length;
    
    // Only add to score if this is the first time answering correctly
    if (isCorrect && !answeredQuestions.has(currentQuestionIndex)) {
      setScore(prevScore => prevScore + 1);
    }
    
    // Mark question as answered
    setAnsweredQuestions(prev => new Set(prev).add(currentQuestionIndex));
  }, [currentQuestionIndex, answeredQuestions]);

  const handleNext = useCallback(() => {
    const nextIndex = currentQuestionIndex + 1;
    if (nextIndex < QUIZ_DATA.length) {
      setCurrentQuestionIndex(nextIndex);
    } else {
      setQuizFinished(true);
    }
  }, [currentQuestionIndex]);

  const handlePrevious = useCallback(() => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  }, [currentQuestionIndex]);

  const handleRestart = useCallback(() => {
    setCurrentQuestionIndex(0);
    setScore(0);
    setQuizFinished(false);
    setAnsweredQuestions(new Set());
    setUserAnswers(new Map());
  }, []);

  const goToQuestion = useCallback((index: number) => {
    if (index >= 0 && index < QUIZ_DATA.length && !quizFinished) {
      setCurrentQuestionIndex(index);
    }
  }, [quizFinished]);

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-800 dark:text-slate-200 flex flex-col items-center justify-center p-4 font-sans">
      {!quizFinished && (
        <QuizSidebar
          questions={QUIZ_DATA}
          currentQuestionIndex={currentQuestionIndex}
          answeredQuestions={answeredQuestions}
          onQuestionSelect={goToQuestion}
        />
      )}
      
      <div className="w-full max-w-2xl">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white">Quiz Platform</h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 mt-2">Test your knowledge.</p>
        </header>
        
        <main className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl overflow-hidden transition-all duration-300">
          {!quizFinished ? (
            <>
              <ProgressBar current={currentQuestionIndex + 1} total={QUIZ_DATA.length} />
              <QuestionCard
                key={currentQuestionIndex}
                questionData={QUIZ_DATA[currentQuestionIndex]}
                onAnswer={handleAnswer}
                onNext={handleNext}
                onPrevious={handlePrevious}
                isFirstQuestion={currentQuestionIndex === 0}
                isLastQuestion={currentQuestionIndex === QUIZ_DATA.length - 1}
                initialSelectedOptions={userAnswers.get(currentQuestionIndex) || []}
              />
            </>
          ) : (
            <QuizSummary
              score={score}
              total={QUIZ_DATA.length}
              onRestart={handleRestart}
            />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
