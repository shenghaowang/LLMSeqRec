from enum import Enum


class ModelType(Enum):
    PopRec = "poprec"
    MatrixFactorization = "mf"
    SASRec = "sasrec"
    LLMSeqRec = "llmseqrec"
