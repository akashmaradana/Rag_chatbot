import generation
import sys

def test_gen():
    print("Loading pipeline...")
    try:
        pipe = generation.get_llm_pipeline()
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return

    # Use a sufficiently long context to test truncation logic availability
    context = (
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. "
        "Learning can be supervised, semi-supervised or unsupervised. "
        "Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, "
        "and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, "
        "machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, "
        "where they have produced results comparable to and in some cases surpassing human expert performance. "
        "Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. "
        "ANs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog. "
        "The adjective 'deep' in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can."
    )
    question = "What fields have deep learning been applied to?"
    
    print(f"\nContext Length: {len(context)} chars")
    print(f"Question: {question}")
    
    print("\nGenerating answer...")
    try:
        answer = generation.generate_answer(pipe, context, question)
        print(f"\n[ANSWER]: {answer}")
        
        if not answer or len(answer.strip()) == 0:
            print("FAILURE: Generated answer is empty.")
            sys.exit(1)
        else:
            print("SUCCESS: Answer generated.")
    except Exception as e:
        print(f"FAILURE: Generation raised an exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gen()
