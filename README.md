
# HiMATE
Implementation of the paper "**HIMATE: A Hierarchical Multi-Agent Framework for Machine Translation Evaluation**".
## Background
#### Abstract
The advancement of Large Language Models (LLMs) enables flexible and interpretable automatic evaluations. In the field of machine translation evaluation, utilizing LLMs with translation error annotations based on Multidimensional Quality Metrics (MQM) yield more human-aligned judgments. However, we contend that existing methods inadequately exploit the fine-grained structural and semantic information within the MQM hierarchy. Furthermore, there is a deficiency in active mechanisms to mitigate inherent model hallucinations in comparative LLM-based approaches. In this paper, we propose **HiMATE**, a **Hi**erarchical **M**ulti-**A**gent Framework for Machine **T**ranslation **E**valuation. To enable detailed error subtype evaluation, we develop hierarchical multi-agent systems grounded in the MQM error typology. Two additional strategies are incorporated to enhance the reliability of error detection and severity assessment within the framework, leveraging self-reflection and collaborative agents discussions. Empirically, HiMATE surpasses competitive baselines across different datasets, demonstrating its effectiveness in conducting human-aligned machine translation evaluations. Further analysis underscores its strengths in error assessment and span detection.
#### Framework Introduction
We propose HIMATE, a novel multi-agent machine translation evaluation framework that employs MQM hierarchy-derived structural-semantic information to configure agent topology, generating human-aligned judgments through a three-phase process: subtype error evaluation initiates the workflow, followed by self-reflection mechanisms to verify detected errors, and culminates in collaborative discussions to achieve consensus.
![overall_framework](https://github.com/nlp2ct-shijie/HiMATE/blob/main/framework.png)
#### Practical Advantages
- HiMATE enhances alignment with human assessment, achieving the best or second-best correlation and similarity on the ZH-EN and EN-DE datasets of MQM22 in segment-level evaluation, and the best meta score on EN-DE dataset of MQM24 in system-level meta-evaluation.
- HiMATE demonstrates superior accuracy in error span identification compared to existing LLM-based methods, particularly excelling in long-sentence robustness and fine-grained error localization.
- HiMATE provides consistent performance improvements across diverse text domains without necessitating domain-specific adaptations.
#### Main Results
![segment_level_results](https://github.com/nlp2ct-shijie/HiMATE/blob/main/segment_level_results.png)
![system_level_results](https://github.com/nlp2ct-shijie/HiMATE/blob/main/system_level_results.png)

The results of Tables 1 and Table 2 above clearly demonstrate the excellent performance of HiMATE in both segment-level and system-level evaluations. The high correlation with human evaluation and the best meta score emphasize the effectiveness of HiMATE in conducting high-quality evaluations. The performance of HiMATE in different backbone models further confirms its robustness and efficiency.

## Requirement & Installation
- Python >= 3.10
- openai >= 1.28.0


## Evaluation
#### Data Preparation
The input data used in evaluation is put at `path/to/HiMATE/data`, which is from the WMT22 Metrics Shared Task. If you want to analyze other data, place the source file `source.txt` and the translation file `hyp.txt` in the folder `path/to/HiMATE/data/year/language/system/`.

#### Agent Evaluation
- The code for each stage is located at `path/to/HiMATE/code`.
- To generate the evaluation results, please run the shell file `path/to/HiMATE/code/pipeline_ende.sh`
```shell
python pipeline.py -m gpt-4o-mini \ # the used model
				   -l ende \ # the evaluated language, zhen or ende
				   -d wmt22 \ # the total dataset
				   --system comet_bestmbr \ # the evaluated system
				   --agent-type Addition \ # the evaluated error type
				   --dataset-prefix ../data \ # the prefix of dataset
				   --agent-type-prefix ../prompts \ # the prefix of prompts
				   -t 0 \ # the temperature of model
				   --next-id 0 \ # the next id of dataset to be evaluated
				   --logpath ../logs \ # the path of log files
				   --key YOUR_API_KEY \ # the API key
				   --url YOUR_API_URL # the API url
```

#### Score Computation
After completing the evaluation, run `/code/score_computation.py` to compute the scores. When calculating the scores, you can adjust parameters to obtain results for different stages.

```shell
python score_computation.py -l path/to/logs
							-s SE   # stage to be calculated, chosen from SE, SRv, CD
							-o score.txt
```

## Framework Directory Structure
```
HiMATE
├── data                                   # the input data for evaluation
├── code                                   # all of the code used for evaluation
│   ├── s1_subtype_evaluation              # the code used in subtype evaluation
│   ├── s2_self_reflection_correction      # the code used in self-reflection correction
│   ├── s3_self_reflection_verification    # the code used in self-reflection verification
│   ├── s4_collaborative_discussion        # the code used in collaborative discussion
├── prompts
│   ├── description_SE.json                # the prompts used in subtype evaluation
│   ├── description_SRc.json               # the prompts used in self-reflection correction
│   ├── description_SRv.json               # the prompts used in self-reflection verification
│   ├── description_CD.json                # the prompts used in collaborative discussion
├── messages                               # the messages of each stage
├── responses                              # the responses of each stage
├── logs                                   # the final output log files
```

## Citation
```
@article{zhang2025himate,
  title={HiMATE: A Hierarchical Multi-Agent Framework for Machine Translation Evaluation}, 
  author={Shijie Zhang and Renhao Li and Songsheng Wang and Philipp Koehn and Min Yang and Derek F. Wong},
  journal={arXiv preprint arXiv:2505.16281},
  year={2025},
  url={https://arxiv.org/abs/2505.16281}
}
```
