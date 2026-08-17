# ReBoot Digest

ReBoot is the first framework to enable fully encrypted and non-interactive training of Multi-Layer Perceptrons (MLPs) using CKKS bootstrapping.
ReBoot has been introduced in the paper: ["ReBoot: Encrypted Training of Deep Neural Networks with CKKS Bootstrapping"](https://arxiv.org/abs/2506.19693).

* this is a digested version of the original [ReBoot GitHub repository](https://github.com/albertopirillo/ReBoot)
*  _Pirillo & Colombo (2025)._ **ReBoot** _: Encrypted Training of Deep Neural Networks with CKKS bootstrapping._ 
  [[arxiv link](https://arxiv.org/abs/2506.19693)], [[local pdf](doc/pdf/Pirillo.Colombo--2025--ReBoot%3DEncrypted.training.of.DNN.withCKKS.bootstrapping.pdf)]

![article-figure-01.png](doc/figures/article-figure-01.png)

* changes:
  * refactored build mechanism, removed containerization
  * added Python packaging 
  * added explanations


## Build 
* pull in openFHE as submodule
    ``` 
    git submodule update --init --recursive
    ```

* build the Python package for ReBuild (see [build doc](doc/build.md) for details)
  ```
  uv venv
  uv sync
  uv pip install -e . -v
  ```


* build _py-openfhe_
  <div style="background-color:#3f1414; border-left: 6px solid #ff4d4d; color:#ffcccc; padding: 10px 16px; margin: 8px 0;">
  <strong style="color:#ff6666;">⚠ WARNING:</strong> Do NOT call <code>uv sync</code> after this step. 
  It removes package <code>openfhe</code> as it is not declared (on purpuse) in <code>pyproject.toml</code>. 
  See <a href="doc/build.md">doc/build.md</a> for details.
  Re-run <code>make py-openfhe</code> after every <code>uv sync</code>.
  </div>
  
    ``` 
    uv pip install --python .venv/bin/python "pybind11>=2.12
    make py-openfhe
    ```


* test if `reboot` import works (must be in the virtual environment)
  ```
  source .venv/bin/activate
  
  python -c "import reboot"
  ```


* run sample MNIST experiment
  ```
  experiments/1_training_plain/run-sample-exp.sh
  ```

* run experiment training 
  ```
  experiments/3_training_encrypted/training_encrypted.sh
  ```


## Project structure 

``` 
ReBoot
├── build        # build artifacts - not checked into git
├── cpp          # all CPP code
├── doc          # documentation
├── experiments  # experiments for the publication
├── extern       # external dependencies/submodules 
└── py           # Python source code
```


## Authors of the original repository: 
- [Alberto Pirillo](https://github.com/albertopirillo): alberto.pirillo@mail.polimi.it
- [Luca Colombo](https://github.com/lucacolombo97): luca2.colombo@polimi.it

