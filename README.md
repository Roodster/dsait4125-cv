# Installation instructions


1. Clone the repository
  ```git clone https://github.com/Roodster/dsait4125-cv.git```

2. Setup virtual environment  
   ``` python -m venv .venv ```  
   ``` source .venv/bin/activate```  
   ``` pip install -r requirements.txt ```

   **NOTE**: If we need a poetry due to dependency issues feel free to set it up.

3. Development
   Please make changes on dedicated branches, i.e.
   ```git checkout -b <branch>```

   Depending on the impact of the change, please request review before accepting the pull request :).

## Accreditation

2d dataset:
```latex
@misc{dsprites17,
author = {Loic Matthey and Irina Higgins and Demis Hassabis and Alexander Lerchner},
title = {dSprites: Disentanglement testing Sprites dataset},
howpublished= {https://github.com/deepmind/dsprites-dataset/},
year = "2017",
}
```
