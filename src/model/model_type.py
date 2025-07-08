from enum import Enum


class ModelType(Enum):
    PopRec = "poprec"
    MatrixFactorization = "matrix_factorization"
    SASRec = "sasrec"
    LLMSeqRec = "llm_seqrec"
