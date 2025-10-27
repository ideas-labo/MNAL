# MNAL
<!-- ABOUT THE PROJECT -->
## About The Project
This online repository is supplementary to the work on Human-Machine Bug Report Identification. It contains the raw data, code for the proposed approach, and Python script to replicate our experiments.

This README file describes the structure of the provided files (Raw data, source code and results). as well as information on the content of this repository.

## Table of Content
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#Table of Content">Table of Content</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#Data">Data</a></li>
    <li><a href="#Research Questions">Research Questions</a></li>
    <li><a href="#Result">Result</a></li>
    <li><a href="#Result Analysis">Result Analysis</a></li>
    <li><a href="#Discussion">Discussion</a></li>
  </ol>
</details>

## Getting Started
### Prerequisites
Run in Python3.10, PyTorch 1.12.1 and CUDA 11.7.0

### Installation

```
git clone https://github.com/anonymous38f7s/MNAL
```

## Data
### Experiment dataset
You can find the raw data of training set used in our experiments at this [link](https://tickettagger.blob.core.windows.net/datasets/nlbse23-issue-classification-train.csv.tar.gz). For test set, it can be downloaded at this [link](https://tickettagger.blob.core.windows.net/datasets/nlbse23-issue-classification-test.csv.tar.gz). These files are meant to be loaded in data_gen.py script to preprocess the data. To replicate our work, you may also need to run model_gen.py script to generate the initial model for the purpose of warm-up training.

## Research Questions
You can use the python code below to run experiments for each respective research question as shown below:
| File name | Corresponding RQ |
| --- | --- |
| RQ1.py | How effective is the effort-aware uncertainty sampling? |
| RQ2.py | How useful is the pseudo-labeling? |
| RQ3.py | To what extent can MNAL improve an arbitrary neural language model? |
| RQ4.py | How effective is MNAL against the state-of-the-art approaches for identifying bug reports? |

Please execute the command lines below to perform the experiment for corresponding RQ.

### RQ1

```
python rq1.py --initial_size INITIAL_SIZE  --query_size QUERY_SIZE --method_setting METHOD_NAME --start_from_run RUN --start_from_step STEP
```

### RQ2

```
python rq2.py --initial_size INITIAL_SIZE  --query_size QUERY_SIZE --method_setting METHOD_NAME --start_from_run RUN --start_from_step STEP
```

### RQ3


```
python rq3.py --initial_size INITIAL_SIZE  --query_size QUERY_SIZE --method_setting METHOD_NAME --start_from_run RUN --start_from_step STEP --model_name MODEL_NAME
```

### RQ4

```
python rq4.py --initial_size INITIAL_SIZE  --query_size QUERY_SIZE --start_from_run RUN --sota SOTA
```

The explanation of each parameter input is listed in the table below:

| Input name | Corresponding RQ |
| --- | --- |
| `INITIAL_SIZE` | The size of warm-up training. In our setting, it can be 300, 500 or 700. |
| `QUERY_SIZE` | The queried report amount in each timestep. In our setting, it can be 300, 500 or 700. |
| `METHOD_NAME` | The approach you would like to perform. In our setting, it can be 'MNAL', 'MNAL_ran' and 'MNAL_un' |
| `RUN` | The repeated run that you would like to start from. In our setting, it can be from 1 to 10. |
| `STEP` | The timestep that you would like to start from. In our setting, it can be from 1 to 10. |
| `METHOD_NAME` | The SOTA approach you would like to perform. In our setting, it can be 'Thung et al.' , 'hbrPredictor', 'Ge et al.' or 'EMBLEM' |


**Note:** The script above does not encompass testing for all SOTA methods. Regarding the HINT [1] method, we have reproduced it in the `rq4_HINT` folder. Execute the following command to run this method:
````
cd rq4_HINT
shell ./sun_ssl.sh
````
- [1] S. Gao, W. Mao, C. Gao, L. Li, X. Hu, X. Xia, and M. R. Lyu, “Learning in the wild:
Towards leveraging unlabeled data for effectively tuning pre-trained code models,” in
Proceedings of the 46th IEEE/ACM International Conference on Software Engineering, ICSE
2024, Lisbon, Portugal, April 14-20, 2024. ACM, 2024, pp. 80:1–80:13. [Online]. Available:
https://doi.org/10.1145/3597503.3639216  

## Result 

The experimental result data we obtained is too large, so we have shared the data anonymously on Dropbox. You can download it from this [link](https://www.dropbox.com/scl/fo/o45rrmaolsvnfp8zldqox/h?rlkey=zkqrpev4qqpxyftr9jukvnk45&dl=0).

## Result Analysis

After obtaining experimental result data, you can use the `result_analysis.py` script to obtain the analysis data as used in our paper. The result analysis in each section are listed below in terms of figure number and table number. 

| File name | Corresponding Section |
| --- | --- |
| Table 1, Fig 7 | RQ1 |
| Table 2, Fig 8 | RQ2 |
| Table 3, Fig 9, Fig 10, Fig 11 | RQ3 |
| Table 5, Fig 12 | RQ4 |
| Table 6, Fig 13 | RQ5 |
| Figure 14, Figure 15 | Discussion |

To retrieve corresponding table data, please run:

```
python result_analysis.py --table NUMBER_TO_ENTER
```
where `NUMBER_TO_ENTER` is the number of table.

To retrieve corresponding figure data, please run:

```
python result_analysis.py --fig NUMBER_TO_ENTER
```
where `NUMBER_TO_ENTER` is the number of figure.


## Discussion

### Discussion Questions
- **Discussion Question 1**: Replace `diversity` with `uncertainty` as the sampling metric for active learning  
- **Discussion Question 2**: Test the performance of MANL under the condition of imbalanced training sample classes  
- **Discussion Question 3**: Test the performance upper bound of MANL  

### Discussion Experiment Source Code
The source code related to the above discussion experiments is encapsulated within the `discussion` folder, which contains the following contents:
````
discussion/
├── result/
├── shell_scripts/
├── balance_test.py
├── data_gen.py
├── diversity_test.py
├── model_gen.py
├── res_summary.py
├── tool_funcs.py
└── upper_bound.py
````
- The `shell_scripts` directory contains shell scripts for one-click execution of experiments.  
- The `result` directory contains the evaluation results of each experiment at each active learning sampling step, stored in JSON format.  

### Running the discussion experiments
First, navigate to the `shell_scripts` directory:
````
cd ./shell_scripts
````

#### Install Dependencies
````
./install.sh
````

#### Generate Imbalanced Class Data
````
./data_gen.sh
````

#### Initialize the Model
````
./model_gen.sh
````

#### Discussion Q1
````
./diversity_test.sh
````

#### Discussion Q2
````
./balance_test.sh
````

#### Discussion Q3
````
./upper_bound_test.sh
````