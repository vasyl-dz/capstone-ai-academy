"""
Evaluation Step with LangSmith
Measures chatbot performance on test questions using OpenEvals
"""

import os
from typing import List, Dict, Optional, Callable
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from openevals.llm import create_llm_as_judge
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RAGEvaluator:
    """Evaluator for RAG chatbot using LangSmith and OpenEvals."""
    
    # Custom evaluation prompts for LLM-as-a-Judge
    ACCURACY_PROMPT = """Evaluate answer accuracy.

Criterion: ACCURACY
The predicted answer must be factually correct and supported by the retrieved context.

Question: {inputs}
Expected: {reference_outputs}
Predicted: {outputs}

Score 0.0-1.0 (0.0=incorrect, 0.5=partially correct, 1.0=fully correct).
Respond with only the score:"""

    RELEVANCE_PROMPT = """Evaluate answer relevance.

Criterion: RELEVANCE
The predicted answer must directly and completely address the user's question.

Question: {inputs}
Answer: {outputs}

Score 0.0-1.0 (0.0=irrelevant, 0.5=somewhat relevant, 1.0=perfectly relevant).
Respond with only the score:"""

    CLARITY_PROMPT = """Evaluate answer clarity.

Criterion: CLARITY
The predicted answer must be well-written, easy to understand, and follow good grammatical and professional standards.

Question: {inputs}
Answer: {outputs}

Score 0.0-1.0 (0.0=unclear, 0.5=moderately clear, 1.0=exceptionally clear).
Respond with only the score:"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 project_name: str = "rag-chatbot-eval",
                 llm_model: str = "llama3.1:8b"):
        """Initialize evaluator with LangSmith client and Ollama model."""
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError("LANGSMITH_API_KEY not found in environment")
        
        os.environ["LANGSMITH_API_KEY"] = self.api_key
        self.client = Client()
        self.project_name = project_name
        
        # Store model string for openevals (expects "ollama:model_name" format)
        self.llm_model = f"ollama:{llm_model}"
    
    def create_test_dataset(self, 
                           test_questions: List[Dict[str, str]], 
                           dataset_name: str = "rag_test_set") -> str:
        """
        Create a test dataset in LangSmith.
        
        Args:
            test_questions: List of dicts with 'question' and 'expected_answer'
            dataset_name: Name for the dataset
        
        Returns:
            Dataset name
        """
        try:
            # Delete existing dataset if it exists
            try:
                self.client.delete_dataset(dataset_name=dataset_name)
            except:
                pass
            
            # Create new dataset
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Test questions for RAG chatbot evaluation"
            )
            
            # Add examples
            for item in test_questions:
                self.client.create_example(
                    dataset_id=dataset.id,
                    inputs={"question": item["question"]},
                    outputs={"expected_answer": item.get("expected_answer", "")}
                )
            
            print(f"✓ Created dataset '{dataset_name}' with {len(test_questions)} questions")
            return dataset_name
            
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            return None
    
    def get_evaluator(self, prompt: str, key: str) -> Callable:
        """Generate LangSmith-compatible evaluators."""
        def langsmith_evaluator(inputs: Dict, outputs: Dict, reference_outputs: Dict):
            evaluator = create_llm_as_judge(
                prompt=prompt,
                model=self.llm_model,
                feedback_key=key,
            )
            eval_result = evaluator(
                inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
            )
            return eval_result

        return langsmith_evaluator
    
    def run_evaluation(self, 
                       chatbot_function,
                       dataset_name: str = "rag_test_set") -> Dict[str, any]:
        """
        Run evaluation on a dataset using the chatbot function.
        
        Args:
            chatbot_function: Function that takes a question and returns an answer
            dataset_name: Name of the test dataset
        
        Returns:
            Evaluation results
        """
        def target_function(inputs: Dict) -> Dict:
            question = inputs["question"]
            answer = chatbot_function(question)
            return {"answer": answer}
        
        try:
            results = evaluate(
                target_function,
                data=dataset_name,
                evaluators=[
                    self.get_evaluator(self.ACCURACY_PROMPT, "accuracy"),
                    self.get_evaluator(self.RELEVANCE_PROMPT, "relevance"),
                    self.get_evaluator(self.CLARITY_PROMPT, "clarity"),
                    ],
                experiment_prefix=self.project_name
            )
            
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
    
    def print_results(self, results):
        """Print evaluation results in a readable format."""
        if not results:
            print("No results to display")
            return
        
        if hasattr(results, 'aggregate_scores'):
            print("\nEvaluation Results:")
            for metric, score in results.aggregate_scores.items():
                print(f"  {metric}: {score:.3f}")


def run_default_evaluation(chatbot_function):
    """Run evaluation with default test questions."""
    
    # Default test questions
    test_questions = [
        {
            "question": "What are the production 'Do's' for RAG?",
            "expected_answer": "1. Use Hybrid Search (vector + BM25) as baseline; 2. Implement metadata filtering; 3. Monitor retrieval quality (recall@k, NDCG);  4. Keep embeddings fresh;  5. Evaluate systematically."
        },
        {
            "question": "What is the difference between standard retrieval and the ColPali approach?",
            "expected_answer": "The ColPali approach method leverages a combination of vision language models and late interaction matching using ColBERT. In contrast, standard retrieval typically relies on text-based search methods."
        },
        {
            "question": "Which embedding model is used in the github repository https://github.com/vasyl-dz/rag-ai-academy",
            "expected_answer": "BAAI/bge-base-en-v1.5"
        }
    ]
    
    try:
        evaluator = RAGEvaluator()
        dataset_name = evaluator.create_test_dataset(test_questions)
        
        if dataset_name:
            print("\nRunning evaluation...")
            results = evaluator.run_evaluation(chatbot_function, dataset_name)
            evaluator.print_results(results)
            return results
        
    except ValueError as e:
        print(f"\n⚠ Evaluation skipped: {str(e)}")
        print("To enable evaluation, set LANGSMITH_API_KEY environment variable")
        return None
    except Exception as e:
        print(f"\n⚠ Evaluation error: {str(e)}")
        return None
