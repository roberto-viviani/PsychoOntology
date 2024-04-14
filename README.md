# PsychoOntology
Testing creation of ontologies from psychometric scales

Using a dataset of responses to scale items, different AI-driven strategies are evaluated to estimate an ontology of the items comprised in the scales. The ontology is defined by representing the semantic similarity of the items of the scales, and is evaluated by comparing the semantic similarity with the correlation of answers from the same items in participants. The semantics is obtained from a large language model. Note that the affinity between scale items is estimated only from their semantics; answers from participants are not used to define the ontology (in contrast to standard approaches, such as item-response models), but only to validate it.

If the creation of the domain ontology is successful, it may be used in two ways. The first is to identify overlaps between scales based on their semantics (independently from the correlation of answers, as the answers are not used to create the ontology). The second is to identify participants that respond inconsistently.

The most direct strategy to form the ontology consists of using semantic embeddings from a large language model. The refinement of this strategy that we epxlore here is the use of a diffusion kernel to create an ontology reflecting the semantic of the specific set of items used in the scales, rather than that of the corpus used to train the language model. Because diffusion kernels create de-noised diffusion operators, we expect the diffusion-based ontology to perform better than the one based directly on the embeddings of the language model.
