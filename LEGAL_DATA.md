# Application to Legal Data

The LexGLUE benchmark includes two single-label multi-class classification tasks. These are referred to as SCOTUS and LEDGAR

```
@article{Chalkidis2022LexGLUEAB,
  title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English},
  author={Ilias Chalkidis and Abhik Jana and Dirk Hartung and Michael James Bommarito and Ion Androutsopoulos and Daniel Martin Katz and Nikolaos Aletras},
  journal={ArXiv},
  year={2022},
  volume={abs/2110.00976}
}
```

SCOTUS has 12 output classes and LEDGAR has 100, so the latter is a better fit for comparison, 
This task is for classification of contract provisions, based on SEC filings.

There are two law-specific models that should be tried on this task, LEGAL-BERT 

```
@article{Chalkidis2020LEGALBERTTM,
  title={LEGAL-BERT: The Muppets straight out of Law School},
  author={Ilias Chalkidis and Manos Fergadiotis and Prodromos Malakasiotis and Nikolaos Aletras and Ion Androutsopoulos},
  journal={ArXiv},
  year={2020},
  volume={abs/2010.02559}
}
```

and CaseLaw BERT

```
@article{Zheng2021WhenDP,
  title={When does pretraining help?: assessing self-supervised learning for law and the CaseHOLD dataset of 53,000+ legal holdings},
  author={Lucia Zheng and Neel Guha and Brandon R. Anderson and Peter Henderson and Daniel E. Ho},
  journal={Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law},
  year={2021}
}
```

These are reported as top performers on LEDGAR in the LexGLUE paper.

Doing this is a simple matter of running the code. Currently, this repository does not use a contrastive learning loss, 
but it should.