# CS4248-Group-30-Project
This repository hosts the code used for our CS4248 Natural Language Processing Project

Authors: Adrian, Billy, Kenji, Nick, Norbert, Russell 

Mentor: Yisong

# Supplementary

## The confounder problem
Confounding memes where benign text + benign image = hateful are a common problem in multimodal hate speech detection. This is because instead of pattern matching, it requires true reasoning for detection.

## Why did we split the classification task into two steps?
Research has also shown that Chain-of-Thought may be an illusion, where if a model has a slight bias towards classifying an image or text as harmful or benign, it will generate tokens that mimic the statistical distribution of human explanations that lead to that specific outcome instead of exploring logical paths to determine the outcome  (Turpin et al., 2023). It is reverse-engineering a justification to fit the initial probabilistic lean.

Example:

![alt text](readme-images/image.png)

This is a benign (ground truth) meme where the LMM rationalise instead of actually reasoning with both hateful and benign interpretations.

### Output comparison

| Model output | Label | Explanation |
|---|---|---|
| LMM output | Hateful | The OCR text "still better than mexican" is interpreted as a derogatory comparison targeting Mexican cuisine, and the neutral pizza image is judged not to offset the harmful text. |

| CARA-generated contrastive hypothesis | Metaphor | Meaning |
|---|---|---|
| Hateful | Close-up of pizza slice with charred crust | The pizza's "perfect" appearance is contrasted with the implied "imperfection" of Mexican food, reinforcing a negative judgment. |
| Benign | Close-up of pizza slice with charred crust | The image can be interpreted as purely descriptive of food texture and appearance, without inherent judgment about other cuisines. |

This forces CARA to explicitly consider both hateful and benign interpretations before deciding, which led to a **Benign** output for this example.

## So why did we use RoBERTa as a judge instead of LLM?
The decision to use RoBERTa over LLM-as-a-judge is due to how this is more of a classification task and how encoder-only models and large generative models handle classification tasks.

Encoder-only is designed for bidirectional language understanding and sequence classification and excels at entailment tasks, whereas Generative LLMs are decode-based, autoregressive models optimised for next-token prediction.

Slight differences in prompts and text can also severely alter the final outputs from LMMs, making them unsuitable for a judge. Research has also shown that models prefer convincingly written sycophantic responses over correct ones (Sharma et al., 2025).

## Why does RAG with MemeCap not work on our problem?

We prepared embeddings using the facebook dataset. When we then ran the same pizza meme through a RAG model. This was what we got.

![alt text](readme-images/image.png)

```
"score": 0.3227677345275879,
"metaphor": "a bald man",
"meaning": "Americans",
"title": "Also Guatemala works"

"score": 0.32171958684921265,
"metaphor": "same black cat",
"meaning": "mexican beans",
"title": "I didnt make this. Just a Facebook find."

"score": 0.3023276925086975,
"metaphor": "the right image",
"meaning": "Europeans",
"title": "I have been told that McDonalds in Europe tastes better too."
```

This showed that the RAG model was looking for words that aligned with the word “mexican” and therefore provided an output that talked about “a bald man”, “mexican beans”, etc. This made us realise that “context” in memes differs greatly from the context normally RAG models are used for (Answering questions related to a certain topic, etc.). Memes require larger inferential leaps to arrive at the intended intention. As such, adding the RAG model only served as more noise for our model, reducing the accuracy.

## Model Performance

| Model | Accuracy | AUROC |
|---|---:|---:|
| CARA | 0.738 | 0.807 |

Comparing our results to the leaderboard of Facebook Hateful Memes Challenge in 2021, our model would performed well, possibly placing in the 5th or 6th place.

![alt text](readme-images/fb-leaderboard.png)

As the competition was held quite a while ago, we also did a little more research on recent research papers to see how SOTA models performed on the same dataset. Only in the last few years have we seen significant improvements in the performance of models on this dataset, with the introduction of LMMs and better training techniques.

Only recently have we seen models that are able to achieve an accuracy of above 80% and an AUROC of above 90%. This shows that there is still a lot of room for improvement in this task, and that the task is still quite challenging.

| Model | Acc | AUROC |
| :---- | :---- | :---- |
| [ExPO-HM](https://arxiv.org/abs/2510.08630) (Mei et al., 2026\) | 76.5 | \- |
| [RA-HMD (Qwen2.5-VL-7B)](https://aclanthology.org/2025.emnlp-main.1215/) (Mei et al., 2025\)  | 82.1 | 91.1 |
| [RA-HMD (Qwen2.5-VL-2B)](https://aclanthology.org/2025.emnlp-main.1215/) (Mei et al., 2025\)  | 79.1 | 88.4 |
| [PromptHate](https://aclanthology.org/2022.emnlp-main.22/) (Cao et al., 2022\)  | 72.98 | 81.45 |

*Note: ExPO-HM scores are different as they are based on a different test set.

# AI Declaration
This project uses the following AI tools:
- Copilot Auto for quick scaffolding and boilerplate code for our own experiments and test and debugging.
- Gemini Pro Nano Banana was used to generate images in early stages of the poster for STePS
- Copilot Auto was used to generated the bash scripts used for remote cluster usage, but these were heavily modified by us to fit our needs and the cluster environment.



## Remote cluster workflow (send code, setup, run)

This project uses three scripts for remote usage:

- `deploy_and_submit.sh`: sync local code to the cluster (deploy only).
- `remote_setup.sh`: create `.venv` and install Python dependencies on the cluster.
- `remote_run.sh`: run inference on GPU (uses `srun` automatically if needed).

### 1) Set up passwordless SSH from WSL (one time)

To avoid entering your password every time, configure SSH key-based auth first.

1. Generate an SSH key (run in WSL):

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "wsl@$(hostname)" -f ~/.ssh/id_ed25519
```

2. Copy your public key to the remote host (replace user/host):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <name>@xlogin.comp.nus.edu.sg
```

If `ssh-copy-id` is not available, run:

```bash
cat ~/.ssh/id_ed25519.pub | ssh <name>@xlogin.comp.nus.edu.sg 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

3. (Optional) Use `ssh-agent` to cache your key passphrase:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

4. Configure a convenient SSH host entry in `~/.ssh/config`:

```
Host xlogin
	HostName xlogin.comp.nus.edu.sg
	User <name>
	IdentityFile ~/.ssh/id_ed25519
	AddKeysToAgent yes
```

### 2) Send code to cluster

From your local project root:

```bash
./deploy_and_submit.sh
```

This syncs code to `~/CS4248/<project-folder>` on the remote and excludes `.venv`, `venv`, `.env`, and `models`.

### 3) Setup environment on cluster (first time, or when deps change)

```bash
ssh <name>@xlogin.comp.nus.edu.sg
cd ~/CS4248/<project-folder>
chmod +x remote_setup.sh remote_run.sh
./remote_setup.sh
```

### 4) Run inference on GPU

Interactive run (recommended for quick testing):

```bash
ssh <name>@xlogin.comp.nus.edu.sg
cd ~/CS4248/<project-folder>
./remote_run.sh --prompt "Hello"
```

Batch run with Slurm:

```bash
ssh <name>@xlogin.comp.nus.edu.sg
cd ~/CS4248/<project-folder>
sbatch remote_job.sbatch
```

Check jobs/logs:

```bash
squeue -u $(whoami)
tail -f slurm-<jobid>.out
```

Notes:
- Ensure remote `~/.ssh` permissions are `700` and `authorized_keys` is `600`.
- If your institution requires additional authentication (2FA, LDAP), follow local instructions or contact admins.
- If `remote_job.sbatch` is used, run `./remote_setup.sh` first so `.venv` exists before the batch job starts.
