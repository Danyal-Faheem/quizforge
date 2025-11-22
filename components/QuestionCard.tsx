import React, { useState } from 'react';
import { Question } from '../types';
import LatexRenderer from './LatexRenderer';

interface QuestionCardProps {
  questionData: Question;
  onAnswer: (selectedOptions: string[]) => void;
  onNext: () => void;
  onPrevious: () => void;
  isFirstQuestion: boolean;
  isLastQuestion: boolean;
  initialSelectedOptions?: string[];
}

const LightbulbIcon: React.FC<{ className?: string }> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2a9 9 0 00-6.36 15.36l-.64.64a1 1 0 001.42 1.42l.64-.64A9 9 0 1012 2zm0 16a7 7 0 110-14 7 7 0 010 14z" />
        <path d="M12 7a1 1 0 00-1 1v.01a1 1 0 002 0V8a1 1 0 00-1-1z" />
        <path d="M12 11a1 1 0 00-1 1v3a1 1 0 002 0v-3a1 1 0 00-1-1z" />
    </svg>
);

const CheckCircleIcon: React.FC<{ className?: string }> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="currentColor">
        <path fillRule="evenodd" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1.47 14.47a.75.75 0 01-1.06-1.06L10.94 14 8.47 11.53a.75.75 0 111.06-1.06L12 11.88l3.47-3.47a.75.75 0 111.06 1.06L13.06 14l2.47 2.47a.75.75 0 11-1.06 1.06L12 15.06l-1.47 1.41z" clipRule="evenodd" />
    </svg>
);


const QuestionCard: React.FC<QuestionCardProps> = ({ 
  questionData, 
  onAnswer, 
  onNext, 
  onPrevious, 
  isFirstQuestion, 
  isLastQuestion,
  initialSelectedOptions = []
}) => {
  const [selectedOptions, setSelectedOptions] = useState<string[]>(initialSelectedOptions);
  const [showHint, setShowHint] = useState<boolean>(false);
  
  // Determine if this is a multiple-answer question
  const isMultipleAnswer = Array.isArray(questionData.answer);
  const correctAnswers = isMultipleAnswer ? questionData.answer : [questionData.answer];
  
  // Determine if correct answers should be shown based on initial state
  const shouldShowCorrectAnswers = () => {
    if (initialSelectedOptions.length === 0) return false;
    if (!isMultipleAnswer) return true; // Single answer always shows after selection
    
    // For multiple answers, check if all correct were selected or all options exhausted
    const allCorrectSelected = correctAnswers.every(ans => initialSelectedOptions.includes(ans));
    const allOptionsSelected = initialSelectedOptions.length === questionData.options.length;
    return allCorrectSelected || allOptionsSelected;
  };
  
  const [showCorrectAnswers, setShowCorrectAnswers] = useState<boolean>(shouldShowCorrectAnswers());

  const handleOptionClick = (option: string) => {
    if (isMultipleAnswer) {
      // Multiple answer mode
      if (selectedOptions.includes(option)) {
        // Deselect if already selected
        const newSelected = selectedOptions.filter(o => o !== option);
        setSelectedOptions(newSelected);
      } else {
        // Add to selected
        const newSelected = [...selectedOptions, option];
        setSelectedOptions(newSelected);
        
        // Check if all correct answers are selected or all options exhausted
        const allCorrectSelected = correctAnswers.every(ans => newSelected.includes(ans));
        const allOptionsSelected = newSelected.length === questionData.options.length;
        
        if (allCorrectSelected || allOptionsSelected) {
          setShowCorrectAnswers(true);
          onAnswer(newSelected);
        }
      }
    } else {
      // Single answer mode - original behavior
      if (selectedOptions.length > 0) return; // Prevent changing answer
      setSelectedOptions([option]);
      setShowCorrectAnswers(true);
      onAnswer([option]);
    }
  };

  const getOptionClass = (option: string) => {
    const isSelected = selectedOptions.includes(option);
    const isCorrect = correctAnswers.includes(option);

    if (isMultipleAnswer) {
      // Multiple answer mode
      if (!showCorrectAnswers) {
        // Before showing correct answers
        if (isSelected) {
          if (isCorrect) {
            // Selected and correct - show green
            return "bg-green-100 dark:bg-green-900/50 border-green-500 dark:border-green-600 text-green-800 dark:text-green-300 ring-2 ring-green-500";
          } else {
            // Selected but incorrect - show red
            return "bg-red-100 dark:bg-red-900/50 border-red-500 dark:border-red-600 text-red-800 dark:text-red-300 ring-2 ring-red-500";
          }
        }
        // Not selected yet - allow selection
        return "bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 border-slate-200 dark:border-slate-600";
      } else {
        // After showing correct answers
        if (isCorrect) {
          return "bg-green-100 dark:bg-green-900/50 border-green-500 dark:border-green-600 text-green-800 dark:text-green-300 ring-2 ring-green-500";
        }
        if (isSelected && !isCorrect) {
          return "bg-red-100 dark:bg-red-900/50 border-red-500 dark:border-red-600 text-red-800 dark:text-red-300 ring-2 ring-red-500";
        }
        return "bg-slate-100 dark:bg-slate-700 border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-400 cursor-not-allowed";
      }
    } else {
      // Single answer mode - original behavior
      if (selectedOptions.length === 0) {
        return "bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 border-slate-200 dark:border-slate-600";
      }

      if (isCorrect) {
        return "bg-green-100 dark:bg-green-900/50 border-green-500 dark:border-green-600 text-green-800 dark:text-green-300 ring-2 ring-green-500";
      }
      if (isSelected && !isCorrect) {
        return "bg-red-100 dark:bg-red-900/50 border-red-500 dark:border-red-600 text-red-800 dark:text-red-300 ring-2 ring-red-500";
      }
      return "bg-slate-100 dark:bg-slate-700 border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-400 cursor-not-allowed";
    }
  };

  return (
    <div className="p-6 sm:p-8">
      <div className="mb-4">
        {isMultipleAnswer && (
          <span className="inline-block bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300 text-sm font-semibold px-3 py-1 rounded-full border border-blue-300 dark:border-blue-700">
            Multiple Correct Answers
          </span>
        )}
      </div>
      <h2 className="text-2xl font-bold leading-tight text-slate-800 dark:text-slate-100 mb-6">
        <LatexRenderer content={questionData.question} />
      </h2>
      
      <div className="space-y-3">
        {questionData.options.map((option, index) => (
          <button
            key={index}
            onClick={() => handleOptionClick(option)}
            disabled={isMultipleAnswer ? showCorrectAnswers : selectedOptions.length > 0}
            className={`w-full text-left p-4 rounded-lg border-2 font-medium transition-all duration-200 flex items-center justify-between ${getOptionClass(option)}`}
          >
            <span><LatexRenderer content={option} /></span>
            {showCorrectAnswers && correctAnswers.includes(option) && <CheckCircleIcon className="w-6 h-6 text-green-600 dark:text-green-400" />}
            {selectedOptions.includes(option) && !correctAnswers.includes(option) && <CheckCircleIcon className="w-6 h-6 text-red-600 dark:text-red-400" />}
          </button>
        ))}
      </div>

      <div className="mt-6 flex justify-between items-start gap-4">
        <div className="flex items-center gap-4">
          {selectedOptions.length === 0 && (
            <button 
              onClick={() => setShowHint(!showHint)}
              className="flex items-center text-sm font-semibold text-blue-600 dark:text-blue-400 hover:underline"
            >
              <LightbulbIcon className="w-5 h-5 mr-1" />
              {showHint ? 'Hide Hint' : 'Show Hint'}
            </button>
          )}
          {!isFirstQuestion && (
            <button
              onClick={onPrevious}
              className="bg-slate-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-slate-700 transition-colors duration-200 shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-500 dark:focus:ring-offset-slate-800"
            >
              Previous
            </button>
          )}
        </div>
        <button
          onClick={onNext}
          className={`font-bold py-2 px-6 rounded-lg transition-colors duration-200 shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 ${
            selectedOptions.length > 0
              ? 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 dark:focus:ring-offset-slate-800'
              : 'bg-gray-500 text-white hover:bg-gray-600 focus:ring-gray-500 dark:focus:ring-offset-slate-800'
          }`}
        >
          {selectedOptions.length > 0 ? (isLastQuestion ? 'Finish' : 'Next') : (isLastQuestion ? 'Finish' : 'Skip')}
        </button>
      </div>
      
      {showHint && selectedOptions.length === 0 && (
        <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg text-yellow-800 dark:text-yellow-200 border border-yellow-200 dark:border-yellow-800">
          <p><span className="font-bold">Hint:</span> <LatexRenderer content={questionData.hint || ''} /></p>
        </div>
      )}
      
      {showCorrectAnswers && (
        <div className="mt-6 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border-l-4 border-blue-500">
          <h3 className="font-bold text-lg text-slate-800 dark:text-slate-100">Explanation</h3>
          <p className="mt-2 text-slate-600 dark:text-slate-300"><LatexRenderer content={questionData.explanation || ''} /></p>
        </div>
      )}
    </div>
  );
};

export default QuestionCard;
