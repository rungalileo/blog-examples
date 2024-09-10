from typing import Optional

import promptquality as pq
from promptquality import Scorers

all_metrics = [
    Scorers.latency,
    Scorers.pii,
    Scorers.toxicity,
    Scorers.tone,
    #rag metrics below
    Scorers.context_adherence,
    Scorers.completeness_gpt,
    Scorers.chunk_attribution_utilization_gpt,
    # Uncertainty, BLEU and ROUGE are automatically included
]

#Custom scorer for response length
def executor(row) -> Optional[float]:
    if row.response:
        return len(row.response)
    else:
        return 0

def aggregator(scores, indices) -> dict:
    return {'Response Length': sum(scores)/len(scores)}

length_scorer = pq.CustomScorer(name='Response Length', executor=executor, aggregator=aggregator)
all_metrics.append(length_scorer)