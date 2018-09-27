# Introduction

This repository is my model for Convai2 2018 NIPS Competition (http://convai.io/). 
- My team name is: "Khai Mai Alt"
- My model name is: "attention-based similarity"

# How to install
- Use Python3 to run this code
- Clone the reppo: "git clone https://github.com/khaimaitien/Mai_convai2.git"
- Install Parlai: "cd Mai_convai2", "python setup.py develop"
- Install additional dependencies: pytorch, torchtext, whoosh ("pip install Whoosh")
- Uncompress file: "cd Mai_convai2/parlai/agents/mai_model"; "unzip embedding.out.zip" 
- Uncompress file: "cd Mai_convai2/parlai/agents/mai_model"; "unzip model.zip"

# Evaluation
- File config for evaluation: "eval_f1.py", "eval_hits.py" in folder: "Mai_convai2/projects/convai2/baselines/mai_model"
- F1: "cd Mai_convai2/projects/convai2/baselines/mai_model"; "python eval_f1.py"
- Hit@1: "cd Mai_convai2/projects/convai2/baselines/mai_model"; "python eval_hits.py"

# Evaluation result
Time for evaluation might be a little bit long as the model is relatively heavy ! (about 31959s for Hit@1 and 17939s for F1). 
- By running evaluation, the result should be like this:
    - Hit@1: 
    ``` 
        ...
        31946s elapsed: {'exs': 7798, 'time_left': '12s', 'hits@1': 0.853, '%done': '99.96%'}
        31959s elapsed: {'exs': 7801, 'time_left': '0s', 'hits@1': 0.853, '%done': '100.00%'}
        EPOCH DONE
        finished evaluating task convai2:self using datatype valid
        {'exs': 7801, 'hits@1': 0.853} 
    ```
    - F1: 
    ```
        ...
        17939s elapsed: {'time_left': '0s', '%done': '100.00%', 'f1': 0.1764, 'exs': 7801}
        EPOCH DONE
        finished evaluating task convai2:self:no_cands using datatype valid
        {'f1': 0.1764, 'exs': 7801}
        ============================
        FINAL F1: 0.1764
    ```