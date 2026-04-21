# Answers to llm training session

## Step 1

What is an LLM, mathematically?
Model weights, which are numbers impliying probabilities to select the next token, built in layers (deep learning)

What does "training" actually change?
Training makes the probabilities for the next token to conform better to some training set

Difference between pretraining and RLHF?
In pretraining we only take the data as training data and train the model using it
In RLHF humans check the results as flag them as good or bad, for additional specific training

Why GPUs and not CPUs?
GPUs can do parallel computation efficiently

## Step 2

How many GPUs does Ai2 use?
Jupiter: 128-node gpu cluster with 1024 GPUs
Augusta: 160-node X 8 GPUs per node = 1280 GPUs
TOTAL: 2304

Tokens per training step for the 32B model? (batch size × seq length)
2048 * 4096 = 8,388,608

What goes wrong at this scale that wouldn't on a laptop? (§6.3)
Hardware failures

## Step 3

Total tokens? Web vs curated ratio?
Over 3.71T/3.9B internet

How did 21T raw bytes become 3.7T tokens? (§2.4.1)
??

Why include non-web sources separately?
Higher quality sources; also including specified topics and knowledge

What are the repeated n-gram strings in §3.1, and why do they matter?
??