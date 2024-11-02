# Multi-Objective Community Detectio

## Summary

Community detection in social networks is often framed as a single-objective optimization problem, where various heuristics or approximate algorithms are employed to optimize a target function that encapsulates the concept of community. This work aims to investigate multi-objective optimization formulations for community detection, utilizing multi-objective evolutionary algorithms to find efficient solutions. Initially, we will analyze and compare functions that can be used for community detection. Experiments on both artificial and real networks will be conducted to evaluate the performance of the multi-objective formulation against popular algorithms.

## Introduction

Community detection in social networks has garnered significant attention due to the crucial roles communities play in the structure-function relationship, as seen in web and biological networks [1, 2]. Communities are, broadly speaking, groups of densely interconnected nodes that are only loosely connected to the rest of the network [3]. To extract such groups, a target function is typically chosen to capture the intuition of a community as a group of nodes with stronger internal connectivity than external.

Traditionally, the problem of community detection is treated as a single-objective optimization issue, limiting the solution to a specific property of community structure and potentially leading to failures when the optimization objectives are inadequate [4]. To address these limitations, this work proposes to investigate a multi-objective approach to the problem of community detection in social networks, providing decision-makers with a set of solutions that offer greater flexibility and accuracy compared to single-objective algorithms [5].

## Objectives

This research proposes to explore a multi-objective approach to efficiently solve the community detection problem. Initially, we will delve into the concept of modularity [6], among other objectives, to enhance performance in community detection tasks.

## Methods

In the context of this study, we will analyze how different researchers interact with one another, considering, for instance, the network of faculty members in the Graduate Program in Computer Science in Brazil.

The CSILab laboratory will provide the necessary physical space and processing resources for the development of this research project. The planned activities for this scientific endeavor include:

- Review of the state of the art
- Study of complex network concepts
- Data collection and processing
- Model construction
- Model validation

## References

[1] E. Ravasz, A.L. Somera, A. Mongru, Z.N. Oltvai, A.L. Barabasi, Hierarchical organization of modularity in metabolic networks, Science 297 (5586) (2002) 1551–1555.
[2] G.W. Flake, S. Lawrence, C.L. Giles, F.M. Coetzee, Self-organization and identification of web communities, IEEE Computer 35 (3) (2002) 66–71.
[3] S. Boccaletti, V. Latora, Y. Moreno, M. Chavez, D.U. Hwang, Complex networks: Structure and dynamics, Physics Report 424 (4-5) (2006) 175–308.
[4] S. Fortunato, M. Barthelemy, Resolution limit in community detection, Proceedings of the National Academy of Sciences 104 (1) (2007) 36–41.
[5] K. Deb, A. Pratab, S. Agarwal, T. MeyArivan, A fast and elitist multiobjective genetic algorithm: NSGA-II, IEEE Transactions on Evolutionary Computation 6 (2) (2002) 182–197.
[6] M.E.J. Newman, M. Girvan, Finding and evaluating community structure in networks, Physical Review E 69 (026113) (2004).

---
