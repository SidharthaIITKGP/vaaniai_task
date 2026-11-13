from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    question_id: str
    category: str
    source_papers: str
    difficulty: str
    question: str
    expected_behavior: str = "Answer from the papers with validated citations."


QUESTIONS = (
    EvaluationQuestion("T1", "transformer", "1706.03762v7", "Easy", "Explain scaled dot-product attention and why the dot product is divided by √dₖ."),
    EvaluationQuestion("T2", "transformer", "1706.03762v7", "Medium", "What are the three ways multi-head attention is used in the Transformer architecture?"),
    EvaluationQuestion("T3", "transformer", "1706.03762v7", "Medium", "How does the Transformer prevent the decoder from attending to future tokens?"),
    EvaluationQuestion("T4", "transformer", "1706.03762v7", "Medium", "Why did the authors choose sinusoidal positional encodings instead of learned positional embeddings?"),
    EvaluationQuestion("T5", "transformer", "1706.03762v7", "Easy", "What BLEU score did the Transformer big model achieve on WMT 2014 English-to-German translation, and how long was it trained?", "Should report 28.4 BLEU and 3.5 days with citations."),
    EvaluationQuestion("T6", "transformer", "1706.03762v7", "Hard", "According to Table 3, how does changing the number of attention heads affect translation performance?", "Use the ablation results; do not claim that more heads are always better."),
    EvaluationQuestion("M1", "moshi", "2410.00037v1", "Easy", "What are Helium and Mimi, and what role does each play in Moshi?"),
    EvaluationQuestion("M2", "moshi", "2410.00037v1", "Medium", "How does Moshi enable full-duplex conversations, including interruptions and overlapping speech?"),
    EvaluationQuestion("M3", "moshi", "2410.00037v1", "Medium", "What is Inner Monologue, and how does it improve speech generation?"),
    EvaluationQuestion("M4", "moshi", "2410.00037v1", "Medium", "What are Moshi's theoretical and practical latency figures, and how does its architecture support low-latency speech?", "Distinguish 160 ms theoretical latency from 200 ms practical latency."),
    EvaluationQuestion("M5", "moshi", "2410.00037v1", "Hard", "Why does Mimi use split residual vector quantization instead of a single RVQ?", "Explain the semantic-versus-acoustic reconstruction trade-off."),
    EvaluationQuestion("M6", "moshi", "2410.00037v1", "Hard", "How can Moshi perform streaming ASR and TTS by changing the relative delay between text and audio tokens?", "Explain how relative delay changes which modality determines the output."),
    EvaluationQuestion("H1", "h_net", "2507.07955v2", "Easy", "Why do the authors argue that traditional BPE tokenization limits fully end-to-end language modeling?"),
    EvaluationQuestion("H2", "h_net", "2507.07955v2", "Medium", "How does H-Net's routing module identify chunk boundaries using cosine similarity?", "Explain boundaries based on similarity between adjacent representations."),
    EvaluationQuestion("H3", "h_net", "2507.07955v2", "Hard", "What problem does the smoothing module solve, and how does its exponential moving average work?", "Explain smoothing and gradient flow."),
    EvaluationQuestion("H4", "h_net", "2507.07955v2", "Medium", "Explain the roles of the encoder, main network, and decoder in H-Net. Why are Mamba-2 layers used in the encoder and decoder?"),
    EvaluationQuestion("H5", "h_net", "2507.07955v2", "Hard", "What is the purpose of H-Net's ratio loss, and how does it prevent trivial chunking strategies?"),
    EvaluationQuestion("H6", "h_net", "2507.07955v2", "Hard", "How does the two-stage H-Net compare against the one-stage version, and after how many training bytes does it overtake the tokenized Transformer at XL scale?", "Identify the approximately 30 billion training-byte crossover for two-stage H-Net XL."),
    EvaluationQuestion("X1", "cross_paper", "1706.03762v7;2507.07955v2", "Hard", "Compare how the original Transformer and H-Net handle input sequences. How does H-Net's dynamic chunking differ from the Transformer's tokenization approach?", "Cite both the Transformer and H-Net papers."),
    EvaluationQuestion("X2", "cross_paper", "1706.03762v7;2410.00037v1", "Hard", "How does Moshi build on Transformer architectures for real-time speech, and which limitations of traditional text-based dialogue pipelines does it address?", "Cite both the Transformer and Moshi papers."),
    EvaluationQuestion("X3", "cross_paper", "2410.00037v1;2507.07955v2", "Hard", "Both Moshi and H-Net employ hierarchical processing. What does hierarchy mean in each architecture, and how do their objectives differ?", "Cite both papers and do not conflate audio-token hierarchy with dynamic input chunking."),
    EvaluationQuestion("U1", "unanswerable", "1706.03762v7", "Unanswerable", "What was the exact monetary cost in USD of training the original Transformer big model?", "Refuse the exact cost; training time and hardware do not establish the bill."),
    EvaluationQuestion("U2", "unanswerable", "2410.00037v1", "Unanswerable", "How many simultaneous users can Moshi serve on one NVIDIA RTX 4090 GPU?", "State that the paper does not establish that capacity."),
    EvaluationQuestion("U3", "unanswerable", "2507.07955v2", "Unanswerable", "What is H-Net's measured inference latency when handling 100 simultaneous production requests?", "State that this measurement is not provided."),
    EvaluationQuestion("U4", "unanswerable", "2410.00037v1", "Unanswerable", "What is the exact carbon footprint in kilograms of CO₂ from training Moshi?", "Refuse the unsupported numeric claim."),
    EvaluationQuestion("U5", "unanswerable", "2410.00037v1", "Unanswerable", "What was Moshi's measured response latency when deployed on Vaani.ai's production infrastructure?", "State that this deployment measurement is not in the paper."),
    EvaluationQuestion("R1", "intent_router", "", "Routing", "Hey", "Natural greeting with no retrieval and zero model calls in hybrid mode."),
    EvaluationQuestion("R2", "intent_router", "", "Routing", "What can you do?", "Explain capabilities with no retrieval and zero model calls in hybrid mode."),
    EvaluationQuestion("R3", "intent_router", "1706.03762v7", "Routing", "Hey, explain multi-head attention.", "Route to grounded RAG despite the greeting."),
    EvaluationQuestion("R4", "intent_router", "", "Routing", "Who won the 2022 FIFA World Cup?", "Return a scope-aware refusal, not an answer from model memory."),
)

