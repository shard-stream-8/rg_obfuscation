- save rollouts to wandb
- fix issue where model spams /think
- clip gradients more harshly, for better stability?
- using these examples in the prompt might be problematic:

<command>cat verifier.py</command> Explanation: I am viewing the verifier code to better understand the problem.
<command>python verifier.py "your_answer"</command> Explanation: I am testing my answer with the verifier.

since they strongly influence the model to directly copy these explanations. So even when we don't penalize "ground truth", the model can just end up parroting these exact outputs and thus avoid ever saying "ground truth".

