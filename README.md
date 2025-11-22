# Generic Quiz Platform

A modern, interactive quiz platform built with React, TypeScript, and Vite. Features LaTeX rendering support, multiple-choice questions with single or multiple correct answers, and a clean, responsive UI.

## Features

### 🎯 Core Functionality
- **Single & Multiple Answer Questions** - Support for both single-choice and multi-select questions
- **LaTeX Rendering** - Beautiful mathematical notation using KaTeX
- **Progress Tracking** - Visual progress bar and question navigation sidebar
- **Persistent State** - Navigate back to previous questions without losing your answers
- **Smart Scoring** - Automatic scoring with detailed explanations

### 🎨 User Experience
- **Collapsible Sidebar** - Quick navigation to any question with visual status indicators
- **Dark Mode Support** - Beautiful UI in both light and dark themes
- **Responsive Design** - Works seamlessly on desktop and mobile devices
- **Interactive Feedback** - Immediate visual feedback for correct/incorrect answers
- **Hints System** - Optional hints for each question

### 📝 Question Types
- **Single Answer** - Traditional multiple-choice questions
- **Multiple Answers** - Select multiple correct options with progressive feedback
- Questions can include LaTeX mathematical expressions in any field

## Demo

The platform supports:
- Questions with mathematical notation (inline: `$x^2$` or block: `$$\int f(x)dx$$`)
- Multiple correct answer questions with smart reveal logic
- Detailed explanations with LaTeX support
- Navigation between questions while preserving state

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/generic-quiz-platform.git
   cd generic-quiz-platform
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
generic-quiz-platform/
├── components/
│   ├── LatexRenderer.tsx      # LaTeX rendering component
│   ├── ProgressBar.tsx         # Quiz progress indicator
│   ├── QuestionCard.tsx        # Individual question display
│   ├── QuizSidebar.tsx         # Question navigation sidebar
│   └── QuizSummary.tsx         # Final score summary
├── App.tsx                     # Main application component
├── constants.ts                # Quiz questions data
├── types.ts                    # TypeScript type definitions
├── index.tsx                   # Application entry point
├── index.html                  # HTML template
└── vite.config.ts             # Vite configuration
```

## Creating Quiz Questions

Questions are defined in `constants.ts`. Here's the format:

### Using AI to Generate Questions

The project includes a `prompt.txt` file that you can use with AI models (like ChatGPT, Claude, or Gemini) to automatically generate quiz questions from your study materials:

1. **Open `prompt.txt`** - This contains a pre-configured prompt template
2. **Attach your study materials** - Upload lecture slides, notes, or textbooks to the AI
3. **Submit the prompt** - The AI will generate questions in the correct JSON format
4. **Copy the output** - Paste the generated JSON array into `constants.ts`

The prompt is specifically designed to generate:
- Multiple choice questions with 4 options
- Detailed explanations with mathematical reasoning
- Helpful hints for each question
- Questions that test both conceptual understanding and calculations
- Properly formatted JSON that works directly with the platform

**Example workflow:**
```bash
# 1. Copy the prompt from prompt.txt
# 2. Go to your preferred AI (ChatGPT, Claude, etc.)
# 3. Paste the prompt and attach your study materials
# 4. Copy the generated JSON array
# 5. Replace QUIZ_DATA in constants.ts with the new questions
```

### Manual Question Creation

You can also create questions manually in `constants.ts`:

### Single Answer Question
```typescript
{
  question: "What is 2 + 2?",
  options: ["3", "4", "5", "6"],
  answer: "4",  // Single correct answer as string
  explanation: "2 + 2 equals 4",
  hint: "Think about basic arithmetic"
}
```

### Multiple Answer Question
```typescript
{
  question: "Which are prime numbers?",
  options: ["2", "4", "7", "9"],
  answer: ["2", "7"],  // Multiple correct answers as array
  explanation: "2 and 7 are the only prime numbers in this list",
  hint: "Prime numbers are only divisible by 1 and themselves"
}
```

### Using LaTeX
```typescript
{
  question: "What is the derivative of $x^2$?",
  options: ["$x$", "$2x$", "$x^3$", "$2$"],
  answer: "$2x$",
  explanation: "Using the power rule: $\\frac{d}{dx}x^2 = 2x$",
  hint: "Remember the power rule"
}
```

## Technologies Used

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **KaTeX** - LaTeX rendering
- **Tailwind CSS** - Styling (via CDN)
- **react-katex** - React wrapper for KaTeX

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
