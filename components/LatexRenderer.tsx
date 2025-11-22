import React from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

interface LatexRendererProps {
  content: string;
}

const LatexRenderer: React.FC<LatexRendererProps> = ({ content }) => {
  // Split content by LaTeX delimiters while preserving them
  // Match $$...$$ (block) first, then $...$ (inline)
  const parts = content.split(/(\$\$[\s\S]+?\$\$|\$[^$]+?\$)/g);

  return (
    <>
      {parts.map((part, index) => {
        // Block math: $$...$$
        if (part.startsWith('$$') && part.endsWith('$$')) {
          const math = part.slice(2, -2);
          return <BlockMath key={index} math={math} />;
        }
        // Inline math: $...$
        else if (part.startsWith('$') && part.endsWith('$')) {
          const math = part.slice(1, -1);
          return <InlineMath key={index} math={math} />;
        }
        // Regular text
        return <span key={index}>{part}</span>;
      })}
    </>
  );
};

export default LatexRenderer;
