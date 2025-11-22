import React from 'react';

interface QuizSummaryProps {
  score: number;
  total: number;
  onRestart: () => void;
}

const TrophyIcon: React.FC<{ className?: string }> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="currentColor">
        <path fillRule="evenodd" d="M17.5 3.5a2.5 2.5 0 00-2.5 2.5v.5h-6v-.5a2.5 2.5 0 00-5 0V11h16V6a2.5 2.5 0 00-2.5-2.5zM6 13v6.5a1.5 1.5 0 001.5 1.5h9a1.5 1.5 0 001.5-1.5V13H6z" clipRule="evenodd" />
    </svg>
);


const QuizSummary: React.FC<QuizSummaryProps> = ({ score, total, onRestart }) => {
    const percentage = Math.round((score / total) * 100);
    let message = '';
    let messageColor = '';

    if (percentage === 100) {
        message = 'Perfect Score! Outstanding!';
        messageColor = 'text-green-500 dark:text-green-400';
    } else if (percentage >= 80) {
        message = 'Excellent Job! You really know your stuff.';
        messageColor = 'text-blue-500 dark:text-blue-400';
    } else if (percentage >= 50) {
        message = 'Good effort! You passed.';
        messageColor = 'text-yellow-500 dark:text-yellow-400';
    } else {
        message = 'Keep practicing! You can do better.';
        messageColor = 'text-red-500 dark:text-red-400';
    }

    return (
        <div className="p-8 text-center flex flex-col items-center">
            <TrophyIcon className="w-20 h-20 text-yellow-500 dark:text-yellow-400 mb-4" />
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">Quiz Complete!</h2>
            <p className={`text-lg font-semibold ${messageColor}`}>{message}</p>
            
            <div className="my-8">
                <p className="text-5xl font-bold text-slate-800 dark:text-slate-100">{score} / {total}</p>
                <p className="text-slate-600 dark:text-slate-400 font-medium">Correct Answers</p>
            </div>
            
            <div className="w-full max-w-xs mx-auto bg-slate-200 dark:bg-slate-700 rounded-full h-4 mb-8">
                <div 
                    className="bg-gradient-to-r from-blue-500 to-green-500 h-4 rounded-full" 
                    style={{ width: `${percentage}%` }}
                ></div>
            </div>

            <button
                onClick={onRestart}
                className="bg-blue-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-blue-700 transition-colors duration-200 shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:focus:ring-offset-slate-800"
            >
                Restart Quiz
            </button>
        </div>
    );
};

export default QuizSummary;
