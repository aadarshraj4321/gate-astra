
"""
==============================================================================
GATE SYLLABUS - SINGLE SOURCE OF TRUTH
==============================================================================
This file contains the complete and structured syllabus for both the
Computer Science (CS) and Data Science & AI (DA) papers.

- 'subject': The primary subject name, prefixed with paper code (CS/DA).
- 'topic': The main topic within the subject.
- 'sub_topic': The specific sub-topic. Can be None if the topic is atomic.
==============================================================================
"""

# --------------------------------------------------------------------------
# PAPER 1: COMPUTER SCIENCE AND INFORMATION TECHNOLOGY (CS)
# --------------------------------------------------------------------------
CS_SYLLABUS = [
    # Section 1: Engineering Mathematics
    {"subject": "CS - Engineering Mathematics", "topic": "Discrete Mathematics", "sub_topic": "Propositional and first order logic."},
    {"subject": "CS - Engineering Mathematics", "topic": "Discrete Mathematics", "sub_topic": "Sets, relations, functions, partial orders and lattices, monoids, groups."},
    {"subject": "CS - Engineering Mathematics", "topic": "Discrete Mathematics", "sub_topic": "Graphs: connectivity, matching, coloring."},
    {"subject": "CS - Engineering Mathematics", "topic": "Discrete Mathematics", "sub_topic": "Combinatorics: counting, recurrence relations, generating functions."},
    {"subject": "CS - Engineering Mathematics", "topic": "Linear Algebra", "sub_topic": "Matrices, determinants, system of linear equations, eigenvalues and eigenvectors, LU decomposition."},
    {"subject": "CS - Engineering Mathematics", "topic": "Calculus", "sub_topic": "Limits, continuity and differentiability. Maxima and minima. Mean value theorem. Integration."},
    {"subject": "CS - Engineering Mathematics", "topic": "Probability and Statistics", "sub_topic": "Random variables. Uniform, normal, exponential, poisson and binomial distributions. Mean, median, mode and standard deviation. Conditional probability and Bayes theorem."},

    # Section 2: Digital Logic
    {"subject": "CS - Digital Logic", "topic": "Number Representations", "sub_topic": "Fixed and floating point"},
    {"subject": "CS - Digital Logic", "topic": "Computer Arithmetic", "sub_topic": "Integer and floating point"},
    {"subject": "CS - Digital Logic", "topic": "Boolean algebra", "sub_topic": None},
    {"subject": "CS - Digital Logic", "topic": "Minimization of Boolean functions", "sub_topic": None},
    {"subject": "CS - Digital Logic", "topic": "Logic gates", "sub_topic": "Static CMOS implementations"},
    {"subject": "CS - Digital Logic", "topic": "Combinational circuits", "sub_topic": "arithmetic circuits, code converters, multiplexers, decoders"},
    {"subject": "CS - Digital Logic", "topic": "Sequential circuits", "sub_topic": "latches and flip-flops, counters, shift-registers and finite state machines"},

    # Section 3: Computer Organization and Architecture
    {"subject": "CS - Computer Organization and Architecture", "topic": "Machine instructions", "sub_topic": None},
    {"subject": "CS - Computer Organization and Architecture", "topic": "Addressing modes", "sub_topic": None},
    {"subject": "CS - Computer Organization and Architecture", "topic": "ALU, data-path and control unit", "sub_topic": None},
    {"subject": "CS - Computer Organization and Architecture", "topic": "Instruction pipelining", "sub_topic": None},
    {"subject": "CS - Computer Organization and Architecture", "topic": "Pipeline hazards", "sub_topic": None},
    {"subject": "CS - Computer Organization and Architecture", "topic": "Memory hierarchy", "sub_topic": "cache"},
    {"subject": "CS - Computer Organization and Architecture", "topic": "Memory hierarchy", "sub_topic": "main memory"},
    {"subject": "CS - Computer Organization and Architecture", "topic": "Memory hierarchy", "sub_topic": "secondary storage"},
    {"subject": "CS - Computer Organization and Architecture", "topic": "I/O interface", "sub_topic": "interrupt handling"},
    {"subject": "CS - Computer Organization and Architecture", "topic": "I/O interface", "sub_topic": "Direct Memory Access (DMA)"},

    # Section 4: Programming and Data Structures
    {"subject": "CS - Programming and Data Structures", "topic": "Programming in C", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Recursion", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Arrays", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Stacks", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Queues", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Linked lists", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Trees", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Binary search trees", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Binary heaps", "sub_topic": None},
    {"subject": "CS - Programming and Data Structures", "topic": "Graphs", "sub_topic": None},

    # Section 5: Algorithms
    {"subject": "CS - Algorithms", "topic": "Searching", "sub_topic": "Linear Search, Binary Search"},
    {"subject": "CS - Algorithms", "topic": "Sorting", "sub_topic": "Bubble, Selection, Insertion, Merge, Quick Sort"},
    {"subject": "CS - Algorithms", "topic": "Hashing", "sub_topic": None},
    {"subject": "CS - Algorithms", "topic": "Asymptotic complexity analysis", "sub_topic": "Worst, average and best cases"},
    {"subject": "CS - Algorithms", "topic": "Asymptotic notations", "sub_topic": "Big-O, Theta, Omega"},
    {"subject": "CS - Algorithms", "topic": "Algorithm design techniques", "sub_topic": "Greedy approach"},
    {"subject": "CS - Algorithms", "topic": "Algorithm design techniques", "sub_topic": "Dynamic programming"},
    {"subject": "CS - Algorithms", "topic": "Algorithm design techniques", "sub_topic": "Divide-and-conquer"},
    {"subject": "CS - Algorithms", "topic": "Graph traversals", "sub_topic": "DFS, BFS"},
    {"subject": "CS - Algorithms", "topic": "Minimum spanning trees", "sub_topic": "Prim's, Kruskal's algorithms"},
    {"subject": "CS - Algorithms", "topic": "Shortest paths", "sub_topic": "Dijkstra's algorithm"},

    # Section 6: Theory of Computation
    {"subject": "CS - Theory of Computation", "topic": "Regular expressions and finite automata", "sub_topic": None},
    {"subject": "CS - Theory of Computation", "topic": "Context-free grammars and push-down automata", "sub_topic": None},
    {"subject": "CS - Theory of Computation", "topic": "Regular and context-free languages", "sub_topic": "Pumping lemma"},
    {"subject": "CS - Theory of Computation", "topic": "Turing machines and undecidability", "sub_topic": None},

    # Section 7: Compiler Design
    {"subject": "CS - Compiler Design", "topic": "Lexical analysis", "sub_topic": None},
    {"subject": "CS - Compiler Design", "topic": "Parsing", "sub_topic": "LL(1), LR(1)"},
    {"subject": "CS - Compiler Design", "topic": "Syntax-directed translation", "sub_topic": None},
    {"subject": "CS - Compiler Design", "topic": "Runtime environments", "sub_topic": None},
    {"subject": "CS - Compiler Design", "topic": "Intermediate code generation", "sub_topic": "Three-address code"},
    {"subject": "CS - Compiler Design", "topic": "Local optimisation", "sub_topic": None},
    {"subject": "CS - Compiler Design", "topic": "Data flow analyses", "sub_topic": "Constant propagation, liveness analysis, common subexpression elimination"},

    # Section 8: Operating System
    {"subject": "CS - Operating System", "topic": "System calls", "sub_topic": None},
    {"subject": "CS - Operating System", "topic": "Processes and Threads", "sub_topic": None},
    {"subject": "CS - Operating System", "topic": "Inter-process communication", "sub_topic": None},
    {"subject": "CS - Operating System", "topic": "Concurrency and Synchronization", "sub_topic": "Locks, semaphores, monitors"},
    {"subject": "CS - Operating System", "topic": "Deadlock", "sub_topic": "Prevention, avoidance, detection, recovery"},
    {"subject": "CS - Operating System", "topic": "CPU Scheduling", "sub_topic": "FCFS, SJF, Priority, Round Robin"},
    {"subject": "CS - Operating System", "topic": "I/O Scheduling", "sub_topic": None},
    {"subject": "CS - Operating System", "topic": "Memory Management", "sub_topic": "Paging, segmentation"},
    {"subject": "CS - Operating System", "topic": "Virtual Memory", "sub_topic": "Demand paging, page replacement algorithms"},
    {"subject": "CS - Operating System", "topic": "File systems", "sub_topic": "File allocation, directory structure"},

    # Section 9: Databases
    {"subject": "CS - Databases", "topic": "ER-model", "sub_topic": None},
    {"subject": "CS - Databases", "topic": "Relational model", "sub_topic": "Relational algebra, tuple calculus, SQL"},
    {"subject": "CS - Databases", "topic": "Integrity constraints", "sub_topic": None},
    {"subject": "CS - Databases", "topic": "Normal forms", "sub_topic": "1NF, 2NF, 3NF, BCNF"},
    {"subject": "CS - Databases", "topic": "File organization", "sub_topic": None},
    {"subject": "CS - Databases", "topic": "Indexing", "sub_topic": "B and B+ trees"},
    {"subject": "CS - Databases", "topic": "Transactions", "sub_topic": "ACID properties"},
    {"subject": "CS - Databases", "topic": "Concurrency control", "sub_topic": "Locks, timestamps, 2PL"},

    # Section 10: Computer Networks
    {"subject": "CS - Computer Networks", "topic": "Layering Concepts", "sub_topic": "OSI and TCP/IP Protocol Stacks"},
    {"subject": "CS - Computer Networks", "topic": "Switching", "sub_topic": "Packet, circuit and virtual circuit switching"},
    {"subject": "CS - Computer Networks", "topic": "Data Link Layer", "sub_topic": "Framing, error detection, Medium Access Control (MAC), Ethernet, bridging"},
    {"subject": "CS - Computer Networks", "topic": "Network Layer", "sub_topic": "IP addressing (IPv4/IPv6), CIDR notation, routing protocols (shortest path, flooding, distance vector, link state)"},
    {"subject": "CS - Computer Networks", "topic": "Network Layer Support Protocols", "sub_topic": "ARP, DHCP, ICMP"},
    {"subject": "CS - Computer Networks", "topic": "Transport Layer", "sub_topic": "Flow control, congestion control, UDP, TCP, sockets"},
    {"subject": "CS - Computer Networks", "topic": "Application Layer Protocols", "sub_topic": "DNS, SMTP, HTTP, FTP, Email"},
]


# --------------------------------------------------------------------------
# PAPER 2: DATA SCIENCE AND ARTIFICIAL INTELLIGENCE (DA)
# --------------------------------------------------------------------------
DA_SYLLABUS = [
    # Section 1: Probability and Statistics
    {"subject": "DA - Probability and Statistics", "topic": "Counting", "sub_topic": "Permutation and combination"},
    {"subject": "DA - Probability and Statistics", "topic": "Probability", "sub_topic": "Axioms, Conditional Probability, Bayes' Theorem"},
    {"subject": "DA - Probability and Statistics", "topic": "Random Variables", "sub_topic": "Distribution functions (uniform, normal, exponential, poisson, binomial)"},
    {"subject": "DA - Probability and Statistics", "topic": "Descriptive Statistics", "sub_topic": "Mean, Median, Mode, Standard Deviation, Variance"},
    {"subject": "DA - Probability and Statistics", "topic": "Correlation and Covariance", "sub_topic": None},
    {"subject": "DA - Probability and Statistics", "topic": "Conditional Expectation and Variance", "sub_topic": None},
    {"subject": "DA - Probability and Statistics", "topic": "Central Limit Theorem", "sub_topic": None},
    {"subject": "DA - Probability and Statistics", "topic": "Hypothesis Testing", "sub_topic": "Chi-square test, t-test"},

    # Section 2: Linear Algebra
    {"subject": "DA - Linear Algebra", "topic": "Systems of linear equations", "sub_topic": None},
    {"subject": "DA - Linear Algebra", "topic": "Vector spaces, basis, dimension", "sub_topic": None},
    {"subject": "DA - Linear Algebra", "topic": "Eigenvalues and eigenvectors", "sub_topic": None},
    {"subject": "DA - Linear Algebra", "topic": "Matrix decomposition", "sub_topic": "SVD, LU, QR"},
    {"subject": "DA - Linear Algebra", "topic": "Matrix properties", "sub_topic": "Determinant, rank, nullity, projections"},

    # Section 3: Calculus and Optimization
    {"subject": "DA - Calculus and Optimization", "topic": "Functions of single variable", "sub_topic": "limit, continuity, differentiability"},
    {"subject": "DA - Calculus and Optimization", "topic": "Taylor series", "sub_topic": None},
    {"subject": "DA - Calculus and Optimization", "topic": "Maxima and minima", "sub_topic": None},
    {"subject": "DA - Calculus and Optimization", "topic": "Vector Calculus", "sub_topic": "Gradient, divergence, curl"},
    {"subject": "DA - Calculus and Optimization", "topic": "Unconstrained Optimization", "sub_topic": "Gradient Descent, Newton's Method"},
    {"subject": "DA - Calculus and Optimization", "topic": "Constrained Optimization", "sub_topic": "Lagrange Multipliers"},

    # Section 4: Programming, Data Structures and Algorithms
    {"subject": "DA - Programming, Data Structures and Algorithms", "topic": "Programming in Python", "sub_topic": None},
    {"subject": "DA - Programming, Data Structures and Algorithms", "topic": "Data Structures", "sub_topic": "Stacks, queues, linked lists, trees, graphs"},
    {"subject": "DA - Programming, Data Structures and Algorithms", "topic": "Algorithms", "sub_topic": "Searching, sorting, hashing"},
    {"subject": "DA - Programming, Data Structures and Algorithms", "topic": "Complexity Analysis", "sub_topic": "Worst-case and average-case"},
    
    # Section 5: Database Management and Warehousing
    {"subject": "DA - Database Management and Warehousing", "topic": "Relational model and SQL", "sub_topic": None},
    {"subject": "DA - Database Management and Warehousing", "topic": "Indexing", "sub_topic": "B+ trees"},
    {"subject": "DA - Database Management and Warehousing", "topic": "Transactions and Concurrency Control", "sub_topic": None},
    {"subject": "DA - Database Management and Warehousing", "topic": "Data Warehousing", "sub_topic": "Schema (star, snowflake), OLAP"},

    # Section 6: Machine Learning
    {"subject": "DA - Machine Learning", "topic": "Supervised Learning", "sub_topic": "Regression and classification problems"},
    {"subject": "DA - Machine Learning", "topic": "Linear Models", "sub_topic": "Linear Regression, Logistic Regression"},
    {"subject": "DA - Machine Learning", "topic": "Loss functions", "sub_topic": None},
    {"subject": "DA - Machine Learning", "topic": "Ensemble Methods", "sub_topic": "Bagging, boosting, Random Forests"},
    {"subject": "DA - Machine Learning", "topic": "Model evaluation", "sub_topic": "Cross-validation, metrics (accuracy, precision, recall, F1-score)"},
    {"subject": "DA - Machine Learning", "topic": "Unsupervised Learning", "sub_topic": "Clustering (k-means, hierarchical)"},
    {"subject": "DA - Machine Learning", "topic": "Dimensionality Reduction", "sub_topic": "PCA (Principal Component Analysis)"},
    {"subject": "DA - Machine Learning", "topic": "Neural Networks", "sub_topic": "Multi-layer perceptrons, activation functions, backpropagation"},

    # Section 7: AI
    {"subject": "DA - AI", "topic": "Search Algorithms", "sub_topic": "Informed, uninformed, heuristic search"},
    {"subject": "DA - AI", "topic": "Logical Reasoning", "sub_topic": "Propositional and first-order logic"},
    {"subject": "DA - AI", "topic": "Reasoning under Uncertainty", "sub_topic": "Conditional independence, Bayesian networks, probabilistic inference"},
]




# --------------------------------------------------------------------------
# SECTION 3: GENERAL APTITUDE (GA) - Common for all papers
# --------------------------------------------------------------------------
GA_SYLLABUS = [
    # Part A: Verbal Aptitude
    {"subject": "GA - General Aptitude", "topic": "Verbal Aptitude", "sub_topic": "Basic English grammar: tenses, articles, adjectives, prepositions, conjunctions, verb-noun agreement"},
    {"subject": "GA - General Aptitude", "topic": "Verbal Aptitude", "sub_topic": "Vocabulary: words, idioms, and phrases in context"},
    {"subject": "GA - General Aptitude", "topic": "Verbal Aptitude", "sub_topic": "Reading and comprehension"},
    {"subject": "GA - General Aptitude", "topic": "Verbal Aptitude", "sub_topic": "Narrative sequencing"},
    
    # Part B: Quantitative Aptitude
    {"subject": "GA - General Aptitude", "topic": "Quantitative Aptitude", "sub_topic": "Data interpretation: data graphs (bar graphs, pie charts, etc.), 2- and 3-dimensional plots, maps, and tables"},
    {"subject": "GA - General Aptitude", "topic": "Quantitative Aptitude", "sub_topic": "Numerical computation and estimation: ratios, percentages, powers, exponents and logarithms, permutations and combinations, and series"},
    {"subject": "GA - General Aptitude", "topic": "Quantitative Aptitude", "sub_topic": "Mensuration and geometry"},
    {"subject": "GA - General Aptitude", "topic": "Quantitative Aptitude", "sub_topic": "Elementary statistics and probability"},
    
    # Part C: Analytical Aptitude
    {"subject": "GA - General Aptitude", "topic": "Analytical Aptitude", "sub_topic": "Logic: deduction and induction"},
    {"subject": "GA - General Aptitude", "topic": "Analytical Aptitude", "sub_topic": "Analogy"},
    {"subject": "GA - General Aptitude", "topic": "Analytical Aptitude", "sub_topic": "Numerical relations and reasoning"},
    
    # Part D: Spatial Aptitude
    {"subject": "GA - General Aptitude", "topic": "Spatial Aptitude", "sub_topic": "Transformation of shapes: translation, rotation, scaling, mirroring, assembling, and grouping"},
    {"subject": "GA - General Aptitude", "topic": "Spatial Aptitude", "sub_topic": "Paper folding, cutting, and patterns in 2 and 3 dimensions"}
]

# We combine ALL lists for easy import
ALL_SYLLABUS_DATA = CS_SYLLABUS + DA_SYLLABUS + GA_SYLLABUS