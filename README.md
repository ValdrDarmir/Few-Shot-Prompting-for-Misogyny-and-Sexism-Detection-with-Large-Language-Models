# Few-Shot-Prompting-for-Misogyny-and-Sexism-Detection-with-Large-Language-Models

## Abstract

With the rise of social media and traffic on these sites, where access can be fast and anonymous, hate speech, especially misogyny, has become a problem and detecting such comments are a growing challenge, especially in low-resource languages.
Although there are many approaches in high-resource languages which perform well, there is a lack of solutions in languages with little training data, such as German.
Therefore, the GermEval2024 GerMS-Detect competition [[1]](#source) was initiated to address this problem.
This work attempts to solve the problem of identifying misogyny and sexism using open source large language models and few-shot prompting.
Here we show that our model achieves a micro F1-score of 0.585 and a score of 0.637 in subtask 1 and 0.302 in subtask 2 in the open track and therefore takes first place within the competition.
The results show that our model delivers good results even without fine-tuning or a lot of training data.

**Please note**
- *This work contains texts that are sexist and misogynistic. Such texts are unavoidable due to the nature of this work.*
- Corpus available at [GermEval Corpus](https://ofai.github.io/GermEval2024-GerMS/download.html)
- GermEval2024 GerMS-Detect: [Competition](https://ofai.github.io/GermEval2024-GerMS/)

## Task Description
*This shared task is about the detection of sexism/misogyny in comments posted in (mostly) German language to the comment section of an Austrian online newspaper.* [[2]](#source)

### Subtask 1:

*In subtask 1 the goal is to predict labels for each text in a dataset where the labels are derived from the original labels assigned by several human annotators.* [[2]](#source)

- Taskdescription for subtask 1 at [OFAI Github](https://ofai.github.io/GermEval2024-GerMS/subtask1.html)

### Subtask 2:

*In subtask 2 the goal is to predict the distribution for each text in a dataset where the distribution is derived from the original distribution of labels assigned by several human annotators.* [[2]](#source)

- Taskdescription for subtask 2 at [OFAI Github](https://ofai.github.io/GermEval2024-GerMS/subtask2.html)

## 01 Data

All file records should be inserted here. The structure of such a jsonl file is shown here (```example_file.jsonl```) as an example.
The data records for the GermEval2024 GerMS-Detect competition can be found under:
- [HuggingFace](https://huggingface.co/datasets/ofai/GerMS-AT)
- [GitHub](https://ofai.github.io/GermEval2024-GerMS/download.html)

Please **note** that the required Gold Label data records must be generated for evaluation. This can be done using the Python program ```(04_code/helper/merge_labels_for_testset.py)```. The files required for this are also linked above (targets and traindev).

## 02 Few-Shot Examples
This folder contains the Few-Shot Examples for the various runs.
These examples were randomly selected from the Competition Phase Trainset (only a maximum text length of the examples was considered in the selection).


## 03 Input 

This folder contains the prompts and output files used for all tests.
For copyright reasons, the ```result.jsonl``` files are empty, as they contain the texts to be predicted in addition to the ID
(The files with the forecasts can be forwarded to me on request)
An example output in such a ```result.jsonl``` file would be:

```
{"id": "BeispielID1", "text": "Text zum bestimmen", "annotations": [{"user": "A001", "label": "1-Gering"}, {"user": "A002", "label": "1-Gering"}, {"user": "A005", "label": "1-Gering"}, {"user": "A008", "label": "1-Gering"}, {"user": "A007", "label": "1-Gering"}, {"user": "A004", "label": "1-Gering"}, {"user": "A009", "label": "2-Vorhanden"}, {"user": "A003", "label": "2-Vorhanden"}, {"user": "A010", "label": "2-Vorhanden"}, {"user": "A012", "label": "2-Vorhanden"}]}
```

Experiments carried out with [fireworks](https://fireworks.ai/) and [openai](https://platform.openai.com) API:
- Zero-shot Mixtral 8x7B
- 5-shot Mixtral 8x7B
- 5-shot + summary guidelines Mixtral 8x7B
- 10-shot Mixtral 8x7B
- 10-shot Mixtral 8x22B


- 5-shot GPT-4o mini
- 5-shot +summary guidelines GPT-4o mini
- 5-shot GPT-3.5 Turbo
- 10-shot GPT-4o mini

## 04 Code
The following folder contains all python files used. The ```data_analysis``` folder contains all the scripts used to analyze the data and generate graphs and word clouds. This folder also contains the script ```statistical_tests``` for the significance test.  The ```helper``` folder contains all scripts that have undertaken smaller tasks in the workflow, such as merging data or deleting empty elements. The ```prediction``` folder contains all scripts that were used to predict the texts.

**Users who want to use these scripts must enter their own API key for fireworks or openai in ```config.py.```**

## 05 Results

All results from the various test runs are saved in this folder. In the subfolder ```presentation```, there are presentations that were held during the course and describe the competition in more detail. The ```results_runs``` subfolder contains the results from the tests. The ```.tsv files``` for Subtask 1 and Subtask 2 are always specified here (as well as the zip file required for [codabench](https://www.codabench.org/competitions/2745/)).
Also the results for subtask 1 and 2 (```scores_{Subtask}.json```). The metrics Accuracy, Precision, Recall and Micro-F1 score can be found in the ```{method}_metrics.txt```. The word clouds and the images of the work (distribution labels and prompt pipeline) can be found in the ```visuals``` subfolder.

### Results for Subtask 1:
| Method                         | Score | Multi-Maj F1 | Bin-Maj F1 | BinOne F1 | BinAll F1 | Dis. Bin F1 |
|---------------------------------|-------|--------------|------------|-----------|-----------|-------------|
| Zero Shot Mixtral 8x22B         | 0.538 | 0.226        | 0.646      | 0.635     | 0.631     | 0.554       |
| 10 Shot Mixtral 8x7B            | 0.620 | **0.390**    | 0.750      | 0.707     | 0.694     | 0.557       |
| **10 Shot Mixtral 8x22B**       | **0.637** | 0.316    | **0.783**  | **0.771** | **0.729** | **0.590**   |
| 5 Shot + GS Mixtral 8x7B*       | 0.608 | 0.415        | 0.722      | 0.703     | 0.667     | 0.534       |
| 5 Shot Mixtral 8x7B*            | 0.623 | 0.387        | 0.748      | 0.718     | 0.704     | 0.558       |
| 5 Shot GPT-4o mini*             | 0.511 | 0.345        | 0.609      | 0.657     | 0.492     | 0.453       |
| 10 Shot GPT-4o mini*            | 0.539 | 0.485        | 0.643      | 0.684     | 0.536     | 0.580       |
| 5 Shot + GS GPT-4o mini*        | 0.481 | 0.323        | 0.576      | 0.618     | 0.461     | 0.566       |
| 5 Shot GPT-3.5 turbo*           | 0.514 | 0.340        | 0.608      | 0.607     | 0.541     | 0.472       |

*Results of Subtask 1 (own approaches). The best result is bold. Methods with an asterisk * were not submitted in the official competition. GS = Guidelines Summary.*


### Results for Subtask 2:

| Method                         | Score | JS Dist Multi | JS Dist Bin |
|---------------------------------|-------|---------------|-------------|
| Zero Shot Mixtral 8x22B         | 0.412 | 0.491         | 0.334       |
| 10 Shot Mixtral 8x7B            | 0.349 | 0.423         | 0.274       |
| **10 Shot Mixtral 8x22B**       | **0.302** | **0.385**   | **0.218**   |
| 5 Shot + GS Mixtral 8x7B*       | 0.362 | 0.429         | 0.296       |
| 5 Shot Mixtral 8x7B*            | 0.325 | 0.394         | 0.256       |
| 5 Shot GPT-4o mini*             | 0.507 | 0.591         | 0.422       |
| 10 Shot GPT-4o mini*            | 0.467 | 0.547         | 0.388       |
| 5 Shot + GS GPT-4o mini*        | 0.543 | 0.630         | 0.456       |
| 5 Shot GPT-3.5 turbo*           | 0.505 | 0.585         | 0.425       |

*Results of Subtask 2 (own approaches). The best result is bold. Methods with an asterisk * were not submitted in the official competition. GS = Guidelines Summary.*

## Quickstart

To use the various scripts, the following steps must be carried out:
- Installation of all requirements (see ```requirements.txt```)
- Download the required corpora ([GermEval Korpus](https://ofai.github.io/GermEval2024-GerMS/download.html)) and install in ```01_data```
- Customize the path in the various Python files. These paths can be found in the ```main()``` function at the end of each file.
- Insert a valid API key from openai or fireworks in ```config.py``` (```04_code/prediction/config.py```)

## Source

[1] Krenn, B., Petrak, J., Kubina, M., & Burger, C. (2024). GERMS-AT: A Sexism/Misogyny Dataset of Forum Comments from an Austrian Online Newspaper. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (pp. 7728–7739).

[2] Website: Krenn, B., Petrak, J., & Gross, S. (2024). GermEval 2024: Shared Task on German Misogyny Detection (GerMS). Austrian Research Institute for Artificial Intelligence (OFAI). Retrieved September 28, 2024, from https://ofai.github.io/GermEval2024-GerMS/


## Citation

If you use our work, please cite our paper and link our repo.

```bibtex
@misc
    {donhauser_2024_few_shot_prompting_for_sexism_detection_llm, 
    title = "Few-Shot Prompting for Misogyny and Sexism Detection with
    Large Language Models", 
    author = "Donhauser, Niklas",  
    month = september, year = "2024", 
    address = "Regensburg, Germany", 
} 
```



