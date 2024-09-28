# Few-Shot-Prompting-for-Misogyny-and-Sexism-Detection-with-Large-Language-Models

## Abstract

With the rise of social media and traffic on these sites, where access can be fast and anonymous, hate speech, especially misogyny, has become a problem and detecting such comments are a growing challenge, especially in low-resource languages.
Although there are many approaches in high-resource languages which perform well, there is a lack of solutions in languages with little training data, such as German.
Therefore, the GermEval2024 GerMS-Detect competition was initiated to address this problem.
This work attempts to solve the problem of identifying misogyny and sexism using open source large language models and few-shot prompting.
Here we show that our model achieves a micro F1-score of 0.585 and a score of 0.637 in subtask 1 and 0.302 in subtask 2 in the open track and therefore takes first place within the competition.
The results show that our model delivers good results even without fine-tuning or a lot of training data.

*This work contains texts that are sexist and misogynistic. Such texts are unavoidable due to the nature of this work.*


## 01 Data

All file records should be inserted here. The structure of such a jsonl file is shown here (```example_file.jsonl```) as an example.
The data records for the GermEval2024 GerMS-Detect competition can be found under:
- [HuggingFace](https://huggingface.co/datasets/ofai/GerMS-AT)
- [GitHub](https://ofai.github.io/GermEval2024-GerMS/download.html)

Please **note** that the required Gold LAbel data records must be generated for evaluation. This can be done using the Python programme ```(04_code/helper/merge_labels_for_testset.py)```. The files required for this are also linked above (targets and traindev).

## 02 Few-Shot Examples
This folder contains the Few-Shot Examples for the various runs.
These examples were randomly selected from the Competition Phase Trainset (only a maximum text length of the examples was considered in the selection).


## 03 Input 

This folder contains the prompts and output files used for all tests.
For copyright reasons, the ```result.jsonl``` files are empty, as they contain the texts to be predicted in addition to the ID.
An example output in such a ```result.jsonl``` file would be:

```
{"id": "BeispielID1", "text": "Text zum bestimmen", "annotations": [{"user": "A001", "label": "1-Gering"}, {"user": "A002", "label": "1-Gering"}, {"user": "A005", "label": "1-Gering"}, {"user": "A008", "label": "1-Gering"}, {"user": "A007", "label": "1-Gering"}, {"user": "A004", "label": "1-Gering"}, {"user": "A009", "label": "2-Vorhanden"}, {"user": "A003", "label": "2-Vorhanden"}, {"user": "A010", "label": "2-Vorhanden"}, {"user": "A012", "label": "2-Vorhanden"}]}
```

Experiments carried out with fireworks and openai API:
- Zero-shot Mixtral 8x7B
- 5-shot Mixtral 8x7B
- 5-shot + summary guidelines Mixtral 8x7B
- 10-shot Mixtral 8x7B
- 10-shot Mixtral 8x22B


- 5-shot GPT-4o mini
- 10-shot GPT-4o mini
- 5-shot +summary guidelines GPT-4o mini
- 5-shot GPT-3.5 Turbo

## 04 Code
The following folder contains all python files used. The ```data_analysis``` folder contains all the scripts used to analyse and generate graphs and word clouds. The ```helper``` folder contains all scripts that have undertaken smaller tasks in the workflow, such as merging data or deleting empty elements. The ```prediction``` folder contains all scripts that were used to predict the texts.

**Users who want to use these scripts must enter their own API key for fireworks or openai in ```config.py.```**

## 05 results

All results from the various test runs are saved in this folder. In the subfolder ```presentation```, there are presentations that were held during the course and describe the competition in more detail. The ```results_runs``` subfolder contains the results from the tests. The ```.tsv files``` for Subtask 1 and Subtask 2 are always specified here (as well as the zip file required for [codabench](https://www.codabench.org/competitions/2745/)).
Also the results for subtask 1 and 2 (```scores_{Subtask}.json```). The metrics Accuracy, Precision, Recall and Micro-F1 score can be found in the ```{method}_metrics.txt```. The word clouds and the images of the work (distribution labels and prompt pipeline) can be found in the ```visuals``` subfolder.
## Source

[1] Krenn, B., Petrak, J., Kubina, M., & Burger, C. (2024). GERMS-AT: A Sexism/Misogyny Dataset of Forum Comments from an Austrian Online Newspaper. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (pp. 7728–7739).


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



