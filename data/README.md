# Dataset Setup

Download and place datasets in this directory with the following structure:

```
data/
├── Slake1.0/              # SLAKE dataset
│   ├── train.json
│   ├── test.json
│   ├── validate.json
│   └── imgs/
├── vqa_med_2019/
│   └── VQAMed2019Test/
│       ├── VQAMed2019_Test_Questions_w_Ref_Answers.txt
│       └── VQAMed2019_Test_Images/
├── vqa_med_2020/
│   └── VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1/
│       ├── VQAMed2020-Task1-VQAnswering-Test-Questions.txt
│       ├── VQAMed2020-Task1-VQAnswering-Test-ReferenceAnswers.txt
│       └── Task1-2020-VQAnswering-Test-Images/
└── vqa_med_2021/
    └── Task1-VQA-2021-TestSet-w-GroundTruth/
        ├── Task1-VQA-2021-TestSet-Questions.txt
        ├── Task1-VQA-2021-TestSet-ReferenceAnswers.txt
        └── images/VQA-500-Images/
```

## Download Links

- **RAD-VQA**: Loaded automatically from HuggingFace (`flaviagiammarino/vqa-rad`)
- **SLAKE**: https://www.med-vqa.com/slake/
- **VQA-Med-2019**: https://github.com/abachaa/VQA-Med-2019
- **VQA-Med-2020**: https://github.com/abachaa/VQA-Med-2020
- **VQA-Med-2021**: https://github.com/abachaa/VQA-Med-2021
