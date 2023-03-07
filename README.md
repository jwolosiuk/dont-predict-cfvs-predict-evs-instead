# Don't Predict Counterfactual Values, Predict Expected Values Instead

This repository contains Suplementary Material to the article:
**Don't predict counterfactual values, predict expected values instead**
that was presented at The 37th AAAI Conference on Artificial Intelligence, Feb. 7-14, Washington, DC, USA, 2023.

Alongside code, you will find the [Supplementary Material PDF file](SupplementaryMaterial.pdf).

The code presents the entire approach discussed in the paper except for:
1. the data and bucketing generation process (random data is used instead); 
2. certain low-level performance optimizations, which do not affect the quality of the method but only its speed. The full code version is used commercially.

File [main.py](main.py) runs experiments which were done in paper. You can run it using python 3.8 after running `pip install -r requirements.txt`.

## How to cite our article:




BibTex:

@inproceedings{Wolosiuk2023,
  title={{Don’t Predict Counterfactual Values, Predict Expected Values Instead}},
  author={Wo{\'l}osiuk, Jeremiasz and {\'S}wiechowski, Maciej and Ma{\'n}dziuk, Jacek},
  booktitle={Proceedings of the AAAI 2023 Conference on Artificial Intelligence},
  volume={TO-BE-DECIDED},
  number={TO-BE-DECIDED},
  pages={TO-BE-DECIDED},
  publisher={AAAI},
  year={2023}
}


Plain text MLA:

Wo{\'l}osiuk, J., {\'S}wiechowski, M. and Ma{\'n}dziuk J. "Don’t Predict Counterfactual Values, Predict Expected Values Instead." Proceedings of the AAAI 2023 Conference on Artificial Intelligence. Vol. [TO-BE-DECIDED]. No. [TO-BE-DECIDED]. Pages.[TO-BE-DECIDED] 2023.
