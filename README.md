# GreedyAL

main_AL_cifar.py demonstrates IIR on cifar10 <br>
main_AL_2Dpoints.py demonstrates AL in a toy example<br><br> <br>
Greedy Active Learning Method
![Alg1_2](https://github.com/user-attachments/assets/df5da906-3378-406a-a288-b110a9526bb8)
Main flow of the AL cycle. The top-K candidate set at cycle t determined by the classifier $C_t(\theta)$,
can be selected as the pool from the unlabeled/search corpus. The AL module extracts a batch set $X_b$ which
is sent for annotation by a user (oracle) that generates the label set $Y_b$. Based on the extended training set,
a new classifier $C_{t+1}(\theta)$ is trained for the next cycle.
<br><br><br>

![Alg2_1](https://github.com/user-attachments/assets/35c2b5bb-b426-47c4-892c-0cf4287ee836)
To calculate the score for a point $x_i$ in the candidate set, we train a classifier $C(\theta_i^+)$ by assuming
the sample is positive. Similarly, we train another classifier $C(\theta_i^-)$
with a negative label. The impact value $S_i$
 is then determined as the minimum value obtained by applying an acquisition function $\mathcal{F}$ to both options.
<br><br><br>


![tree_dfs](https://github.com/user-attachments/assets/12e08fb0-ad0b-4439-8608-64d1a222a1ca)

In the SVM scenario, the GAL algorithm employs a binary tree structure. The initial point $x_{i0}$
is chosen through the NEXT procedure. The red circles represent the results obtained from
NEXT, which are based on the corresponding pseudo-labels.
<br><br><br>

![tin_can](https://github.com/user-attachments/assets/61cad3fd-8ced-45e3-8356-4893003327ee)
Image retrieval results for Tin Can in FSOD-IR dataset with B = 3 at iteration 4. Green
boxes stand for relevant results while red boxes account for false positives. The second query image has two
objects: Can and Display monitor. The RBMAL method mistakenly retrieves images with monitor, where
GAL succeeds to find the common pattern in the queries. This example illustrates how the initial ambiguity
regarding the object is gradually resolved through the active learning cycles, allowing the algorithm to
effectively capture the query concept.
