
# Dropout Disco : Week 1 Report

--- 
# Part 1: Word2vec Training (SGNS / CBOW)


---

| **Query Word** | **SGNS Top Matches**                                                  | **CBOW Top Matches**                                                   |
| -------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **king**       | valdemar (0.7430), abijam (0.7295), reigned (0.7247), haakon (0.7197) | throne (0.7543), kings (0.7472), prince (0.7277), reigned (0.7222)     |
| **queen**      | hih (0.7058), sibylla (0.7045), battenberg (0.6965), valois (0.6932)  | princess (0.8449), lady (0.7860), elizabeth (0.7517), consort (0.7467) |


---

## King - Man + Woman = ??
| **Model** | **Top Analogy Matches** |
| --------- | -------------------------------------------------------------------------------------- |
| **SGNS**  | valois (0.6689), wedded (0.6506), yolande (0.6381), dowager (0.6357), heiress (0.6255) |
| **CBOW**  | throne (0.6971), son (0.6833), prince (0.6804), queen (0.6535), consort (0.6327)       |


---

| Anchor           | SGNS #1              | SGNS #2             | SGNS #3              | SGNS #4             | SGNS #5               | CBOW #1              | CBOW #2             | CBOW #3             | CBOW #4            | CBOW #5               |
| ---------------- | -------------------- | ------------------- | -------------------- | ------------------- | --------------------- | -------------------- | ------------------- | ------------------- | ------------------ | --------------------- |
| **animal**       | animals (0.766)      | carrion (0.713)     | carnivores (0.696)   | annelida (0.692)    | integumentary (0.690) | animals (0.852)      | eating (0.817)      | insect (0.784)      | bees (0.776)       | habits (0.775)        |
| **chihuahua**    | sonora (0.805)       | ciudad (0.780)      | puebla (0.776)       | tamaulipas (0.765)  | jalisco (0.749)       | ciudad (0.889)       | prado (0.845)       | hernando (0.829)    | puebla (0.827)     | universitaria (0.823) |
| **city**         | town (0.787)         | downtown (0.749)    | metropolitan (0.747) | suburbs (0.746)     | outskirts (0.735)     | cities (0.799)       | town (0.762)        | towns (0.715)       | suburbs (0.702)    | downtown (0.695)      |
| **london**       | stansted (0.722)     | guildford (0.688)   | glasgow (0.683)      | birmingham (0.681)  | paddington (0.678)    | manchester (0.695)   | edinburgh (0.683)   | rotherhithe (0.653) | shoreditch (0.651) | oxfordshire (0.649)   |

---

| Anchor           | SGNS #1              | SGNS #2             | SGNS #3              | SGNS #4             | SGNS #5               | CBOW #1              | CBOW #2             | CBOW #3             | CBOW #4            | CBOW #5               |
| ---------------- | -------------------- | ------------------- | -------------------- | ------------------- | --------------------- | -------------------- | ------------------- | ------------------- | ------------------ | --------------------- |
| **clothing**     | garments (0.797)     | jewelry (0.769)     | footwear (0.759)     | textiles (0.748)    | apparel (0.745)       | footwear (0.899)     | sewing (0.838)      | clothes (0.813)     | carpets (0.810)    | jewelry (0.810)       |
| **color**        | colour (0.813)       | colors (0.756)      | srgb (0.715)         | palettes (0.702)    | etre (0.699)          | colour (0.873)       | colors (0.859)      | colours (0.790)     | hues (0.762)       | matte (0.761)         |
| **red**          | white (0.765)        | yellow (0.740)      | blue (0.710)         | green (0.699)       | colored (0.678)       | white (0.869)        | blue (0.816)        | green (0.772)       | yellow (0.768)     | black (0.699)         |
| **emotion**      | emotions (0.839)     | pessimistic (0.820) | absurd (0.818)       | anomie (0.809)      | perceptive (0.805)    | emotions (0.879)     | intellect (0.862)   | conscious (0.861)   | loneliness (0.858) | wishful (0.844)       |

---

| Anchor           | SGNS #1              | SGNS #2             | SGNS #3              | SGNS #4             | SGNS #5               | CBOW #1              | CBOW #2             | CBOW #3             | CBOW #4            | CBOW #5               |
| ---------------- | -------------------- | ------------------- | -------------------- | ------------------- | --------------------- | -------------------- | ------------------- | ------------------- | ------------------ | --------------------- |
| **country**      | faroes (0.690)       | mozambique (0.687)  | manama (0.684)       | djibouti (0.680)    | bicontinental (0.680) | nation (0.754)       | economy (0.702)     | mozambique (0.700)  | countries (0.694)  | angola (0.693)        |
| **india**        | punjab (0.777)       | hyderabad (0.772)   | pakistan (0.767)     | ahmedabad (0.755)   | gujarat (0.745)       | pakistan (0.795)     | gujarat (0.787)     | kashmir (0.778)     | punjab (0.767)     | bangladesh (0.760)    |
| **fruit**        | zibethinus (0.852)   | fruits (0.840)      | durio (0.804)        | vines (0.803)       | citrus (0.803)        | fruits (0.927)       | seeds (0.913)       | dried (0.901)       | meat (0.900)       | vegetables (0.900)    |
| **apple**        | iigs (0.764)         | macintosh (0.752)   | iie (0.750)          | iic (0.747)         | iicx (0.727)          | macintosh (0.805)    | hypercard (0.764)   | iic (0.756)         | iie (0.743)        | amiga (0.739)         |



---

| Anchor           | SGNS #1              | SGNS #2             | SGNS #3              | SGNS #4             | SGNS #5               | CBOW #1              | CBOW #2             | CBOW #3             | CBOW #4            | CBOW #5               |
| ---------------- | -------------------- | ------------------- | -------------------- | ------------------- | --------------------- | -------------------- | ------------------- | ------------------- | ------------------ | --------------------- |
| **professional** | sumo (0.715)         | trombonists (0.693) | dominatrixes (0.681) | skilled (0.657)     | cheerleading (0.653)  | amateur (0.849)      | athletes (0.788)    | skilled (0.775)     | sumo (0.767)       | professionals (0.756) |
| **lawyer**       | bayard (0.769)       | cowen (0.751)       | pilz (0.746)         | schine (0.740)      | lecturer (0.738)      | pundit (0.848)       | keating (0.841)     | journalist (0.834)  | spokesman (0.833)  | activist (0.830)      |
| **technology**   | technologies (0.749) | automation (0.727)  | mems (0.717)         | engineering (0.715) | robotics (0.705)      | technologies (0.836) | engineering (0.808) | robotics (0.754)    | automation (0.752) | advances (0.743)      |
| **python**       | sketch (0.728)       | monty (0.728)       | pythons (0.677)      | sketches (0.661)    | perl (0.651)          | monty (0.862)        | sketch (0.669)      | grail (0.654)       | gms (0.636)        | palin (0.635)         |

---

## **Feature Fusion** /*SGNS*, *CBOW*

| Feature         | Size |
|-----------------|------|
| Title (Word2Vec)| 300  |
| Type            | 8    |
| Day of Week     | 3    |
| Domain          | 8    |
| Hour            | 1    |
| Karma           | 1    |
| Descendants     | 1    |
| **Total**       | 322  |

---
## Model MAE Comparison 

![image](https://hackmd.io/_uploads/S1OhuaFQle.png)
---

## Model R2 Comparison
<img width="918" alt="image" src="https://github.com/user-attachments/assets/91096eef-8a3a-443f-804b-df575ee3d3e4" />


---

## Observations: 
1. CBOW >> SGNS
2. With both Karma + Descendants: R2=0.7, MAE~=7 Upvotes
3. Without Karma: R2=0.3, MAE~=20 Upvotes
4. Without Descendants: R2<0, MAE~=30 Upvotes

TO-DO: given user "created date", last snapshot "karma", we can predict karma given the post date.
---

### ETHAN'S TOP TIP!
Memory overflow

---

### SKIPGRAM EVALUATION
| **Target Word**   | Nearest 1             | Nearest 2             | Nearest 3             |
|-------------------|-----------------------|------------------------|------------------------|
| **ferocious**     | smokes (0.4627)       | seizes (0.4580)        | independents (0.4565)  |
| **jourdan**       | shane (0.4720)        | subsist (0.4708)       | refuse (0.4682)        |
| **ctesiphon**     | coexist (0.4539)      | ubuweb (0.4529)        | hearse (0.4368)        |
| **duchenne**      | canterbury (0.4881)   | discus (0.4270)        | vegetation (0.4231)    |
| **computer**      | disc (0.4982)         | net (0.4898)           | liberalization (0.4889)|
| **tanned**        | hutongs (0.4814)      | breakout (0.4775)      | reid (0.4644)          |
| **hyland**        | puget (0.4779)        | beccus (0.4693)        | nimzowitsch (0.4468)   |
| **protanopic**    | tyramine (0.4627)     | chileans (0.4480)      | desventuradas (0.4430) |
| **python**        | misspellings (0.5156) | competently (0.4987)   | cretaceous (0.4959)    |
| **oses**          | vlacq (0.4761)        | suitable (0.4638)      | vanity (0.4590)        |

---

## Dan got excited about devops

![image](https://hackmd.io/_uploads/SyIy56tmxl.png)

---

![image](https://hackmd.io/_uploads/Hk3_uaKXex.png)

```
#!/usr/bin/env bash
# run like `source setup.sh` to ensure active shell is set up with venv
apt update
# ensure we have all the utils we need
apt install -y vim rsync git git-lfs nvtop htop tmux curl
apt upgrade -y
# install uv and sync
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
# activate virtual environment for running python scripts
source .venv/bin/activate
echo "Setup complete - virtual environment activated. You can now run Python scripts directly."
echo "Run 'git lfs pull' to download large files."
```

---

and got stuck on the CBOW model... making it worse with every commit!

![Screenshot from 2025-06-11 15-16-31](https://hackmd.io/_uploads/BJHrU6Fmxg.png)

![Screenshot from 2025-06-12 14-57-47](https://hackmd.io/_uploads/rydSUTtmee.png)

---

then spent [redacted] hours today trying to make our custom CBOW dataset nicer

![image](https://hackmd.io/_uploads/SyOxDaKXlg.png)

---

e.g. before...

![image](https://hackmd.io/_uploads/H1MtwpYQxg.png)

---

and after...

![image](https://hackmd.io/_uploads/SJ30vatQle.png)

---

:melting_face: :fire: 

![](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNTNndTRzOTR0d2V3bmxyaXZqNm9oZnh2YmgxZnRwMGFkczBiYjgyMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Q81NcsY6YxK7jxnr4v/giphy.gif)

---

# <div style="font-size: 54px;">Things I Have Seen</div>

<div style="font-size: 28px;">

| **Command line**        | **Embeddings**          | **Feature engineering** |
|-------------------------|-------------------------|-------------------------|
| **SSH**                 | **Word2Vec (CBOW, SG)** | **Tokenizer**           |
| **Remote servers**      | **Python**              | **Text8**               |
| **Tmux**                | **Copilot**             | **Hacker News**         |
| **PyTorch**             | **Hyperparameters**     | **Data leaking**        |
| **Virtual environments**| **PostgreSQL**          | **TQDM**                |
| **Data handling**       | **EDA**                 | **UV**                  |
| **Splitting data**      | **Training an MLP**     | **APT**                 |
| **.sh scripts**         | **.parquet file**       | **Brew**                |
| **wandb**               | **hackmd**              | **Git LFS**             |

</div>
