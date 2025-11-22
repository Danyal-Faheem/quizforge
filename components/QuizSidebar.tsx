import React, { useState } from 'react';
import { Question } from '../types';

interface QuizSidebarProps {
  questions: Question[];
  currentQuestionIndex: number;
  answeredQuestions: Set<number>;
  onQuestionSelect: (index: number) => void;
}

const ChevronLeftIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
  </svg>
);

const ChevronRightIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
  </svg>
);

const QuizSidebar: React.FC<QuizSidebarProps> = ({ 
  questions, 
  currentQuestionIndex, 
  answeredQuestions,
  onQuestionSelect 
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const truncateText = (text: string, maxLength: number = 30) => {
    // Remove LaTeX expressions for preview
    const plainText = text.replace(/\$\$[\s\S]+?\$\$|\$[^$]+?\$/g, '...');
    return plainText.length > maxLength 
      ? plainText.substring(0, maxLength) + '...' 
      : plainText;
  };

  return (
    <>
      {/* Toggle Button - Always visible */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="fixed left-4 top-4 z-50 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-lg shadow-lg transition-all duration-300"
        aria-label={isCollapsed ? 'Open sidebar' : 'Close sidebar'}
      >
        {isCollapsed ? (
          <ChevronRightIcon className="w-6 h-6" />
        ) : (
          <ChevronLeftIcon className="w-6 h-6" />
        )}
      </button>

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 h-full bg-white dark:bg-slate-800 shadow-2xl transition-transform duration-300 z-40 ${
          isCollapsed ? '-translate-x-full' : 'translate-x-0'
        }`}
        style={{ width: '300px' }}
      >
        <div className="p-6 pt-16 h-full overflow-y-auto">
          <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
            Questions ({answeredQuestions.size}/{questions.length})
          </h3>
          
          <div className="space-y-2">
            {questions.map((question, index) => {
              const isAnswered = answeredQuestions.has(index);
              const isCurrent = index === currentQuestionIndex;
              
              return (
                <button
                  key={index}
                  onClick={() => onQuestionSelect(index)}
                  className={`w-full text-left p-3 rounded-lg transition-all duration-200 border-2 ${
                    isCurrent
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30'
                      : isAnswered
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/30'
                      : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/50 hover:bg-slate-100 dark:hover:bg-slate-700'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <span
                      className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        isAnswered
                          ? 'bg-green-500 text-white'
                          : isCurrent
                          ? 'bg-blue-500 text-white'
                          : 'bg-slate-300 dark:bg-slate-600 text-slate-700 dark:text-slate-300'
                      }`}
                    >
                      {index + 1}
                    </span>
                    <span className="text-sm text-slate-700 dark:text-slate-300 flex-1">
                      {truncateText(question.question)}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Overlay when sidebar is open */}
      {!isCollapsed && (
        <div
          className="fixed inset-0 bg-black/20 z-30 lg:hidden"
          onClick={() => setIsCollapsed(true)}
        />
      )}
    </>
  );
};

export default QuizSidebar;
