"""
Reflection Agent for Self-Correction
Evaluates and refines answers through reflection
"""

import ollama
from typing import Dict


class ReflectionAgent:
    """Agent that reflects on and improves answers."""
    
    def __init__(self, model: str = "llama3.1:8b"):
        """Initialize reflection agent."""
        self.model = model
    
    def reflect(self, query: str, initial_answer: str, context: str = "") -> Dict[str, str]:
        """
        Reflect on an initial answer and provide improvements.
        
        Args:
            query: The original question
            initial_answer: The initial answer to reflect on
            context: Optional context used to generate the answer
        
        Returns:
            Dictionary with reflection and improved answer
        """
        reflection_prompt = f"""You are a critical evaluator. Review the following answer and identify any issues:

Question: {query}

Answer to evaluate: {initial_answer}

Analyze the answer for:
1. Accuracy - Is the information correct?
2. Completeness - Does it fully answer the question?
3. Clarity - Is it clear and well-structured?
4. Relevance - Does it stay on topic?

Provide a brief critique (2-3 sentences)."""

        try:
            # Get reflection
            reflection_response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": reflection_prompt}]
            )
            
            reflection = reflection_response["message"]["content"]
            
            # Generate improved answer
            improvement_prompt = f"""Based on this critique, provide an improved answer:

Original Question: {query}

Initial Answer: {initial_answer}

Critique: {reflection}

{"Context: " + context if context else ""}

Provide an improved, corrected answer. Only provide the final answer without mentioning improvements or critiques:"""
            
            improvement_response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": improvement_prompt}]
            )
            
            improved_answer = improvement_response["message"]["content"]
            
            return {
                "reflection": reflection,
                "improved_answer": improved_answer
            }
            
        except Exception as e:
            return {
                "reflection": f"Error during reflection: {str(e)}",
                "improved_answer": initial_answer
            }
    
    def should_reflect(self, initial_answer: str) -> bool:
        """
        Determine if reflection is needed based on answer quality.
        Simple heuristic: reflect if answer is very short or contains uncertainty markers.
        """
        uncertainty_markers = [
            "i don't know", "not sure", "unclear", "maybe", 
            "possibly", "cannot answer", "no information"
        ]
        
        answer_lower = initial_answer.lower()
        
        # Reflect if answer is too short or contains uncertainty
        if len(initial_answer) < 50:
            return True
        
        if any(marker in answer_lower for marker in uncertainty_markers):
            return True
        
        return False
    
    def chain_reflect(self, query: str, initial_answer: str, context: str = "", 
                     max_iterations: int = 2) -> Dict[str, any]:
        """
        Chain multiple reflection iterations for progressive improvement.
        
        Args:
            query: The original question
            initial_answer: The initial answer
            context: Optional context
            max_iterations: Maximum number of reflection iterations
        
        Returns:
            Dictionary with all reflections and final answer
        """
        current_answer = initial_answer
        reflections = []
        
        for i in range(max_iterations):
            if not self.should_reflect(current_answer) and i > 0:
                break
            
            result = self.reflect(query, current_answer, context)
            reflections.append({
                "iteration": i + 1,
                "reflection": result["reflection"],
                "answer": result["improved_answer"]
            })
            
            current_answer = result["improved_answer"]
        
        return {
            "initial_answer": initial_answer,
            "reflections": reflections,
            "final_answer": current_answer,
            "iterations": len(reflections)
        }
