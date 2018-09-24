#Introduction
This repository is my model for NIPS 2018 Competition Competition Convai2 (http://convai.io/) 
# How to install
- Use Python3 to run this code
- Clone the reppo: "git clone https://github.com/khaimaitien/Mai_convai2.git"
- Install Parlai: "cd Mai_convai2", "python setup.py develop"
- Install additional dependencies: pytorch, torchtext, whoosh ("pip install Whoosh")
- Uncompress file: "cd Mai_convai2/parlai/agents/mai_model"; "unzip embedding.out.zip" 
- Uncompress file: "cd Mai_convai2/parlai/agents/mai_model"; "unzip model.zip"
# Evaluation
Evaluating is a little bit slow because the model is relatively heavy !
- File config for evaluation: "eval_f1.py", "eval_hits.py" in folder: "Mai_convai2/projects/convai2/baselines/mai_model"
- F1: "cd Mai_convai2/projects/convai2/baselines/mai_model"; "python eval_f1.py"
- Hit@1: "cd Mai_convai2/projects/convai2/baselines/mai_model"; "python eval_hits.py"
# Evaluation result
- By running evaluation, the result should be like this:
    - F1: ``` ```
    - Hit@1: ``````