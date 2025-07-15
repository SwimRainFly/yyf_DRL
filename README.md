# RACCOON


This is a PyTorch implementation of Graph Neural Networks combined with Deep Reinforcement Learning (GNN-DRL) for multi-agent environments. The solution integrates graph neural networks to model agent interactions and relationships, enhancing cooperative behaviors and decision-making in complex multi-agent systems.


# How to use my code?

 
You can dircetly run 'MAPPO_MPE_main.py' in your own IDE


# Requirements

```python
absl-py==2.0.0
annotated-types==0.7.0
anyio==4.9.0
cachetools==5.3.2
certifi==2023.11.17
charset-normalizer==3.3.2
cloudpickle==1.2.2
colorama==0.4.6
cycler==0.12.1
dgl==1.1.2
distro==1.9.0
et_xmlfile==2.0.0
exceptiongroup==1.3.0
fonttools==4.44.3
future==0.18.3
gitignore_parser==0.1.12
google-auth==2.25.2
google-auth-oauthlib==1.2.0
gpt-readme==0.1.2
grpcio==1.60.0
gym==0.15.4
gym-notices==0.0.8
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.4
importlib-metadata==6.8.0
Jinja2==3.1.2
jiter==0.10.0
joblib==1.3.2
kiwisolver==1.4.5
Markdown==3.5.1
markdown-it-py==3.0.0
MarkupSafe==2.1.3
matplotlib==3.5.1
mdurl==0.1.2
# Editable install with no version control (multiagent==0.0.1)
-e d:\pycharm\py_projects\yyf_gnn-drl\marl\mpe
networkx==3.2.1
numpy==1.26.2
numpy-stl==3.1.1
oauthlib==3.2.2
openai==1.91.0
opencv-python==4.8.1.78
openpyxl==3.1.5
packaging==23.2
pandas==1.2.4
Pillow==10.1.0
protobuf==4.23.4
psutil==5.9.6
pyasn1==0.5.1
pyasn1-modules==0.3.0
pydantic==2.11.7
pydantic_core==2.33.2
pyglet==1.3.2
Pygments==2.19.2
pyparsing==3.1.1
python-dateutil==2.8.2
python-utils==3.8.1
pytz==2023.3.post1
requests==2.31.0
requests-oauthlib==1.3.1
rich==14.0.0
rsa==4.9
scikit-learn==1.3.2
scipy==1.11.4
seaborn==0.13.1
six==1.16.0
sniffio==1.3.1
tensorboard==2.15.1
tensorboard-data-server==0.7.2
threadpoolctl==3.2.0
torch @ file:///D:/whl/torch-1.12.0%2Bcu113-cp39-cp39-win_amd64.whl#sha256=e2e96f5c85b344f274cf3c934c36661db5871223f4ffe9c45d5f86a233103314
torch-cluster @ file:///C:/Users/82770/whl/torch_cluster-1.6.0%2Bpt112cu113-cp39-cp39-win_amd64.whl#sha256=acc7aceb8e407124d54c34186df49de5f9ea1a031e31eec09bcab0f19b4e9fba
torch-scatter @ file:///C:/Users/82770/whl/torch_scatter-2.1.0%2Bpt112cu113-cp39-cp39-win_amd64.whl#sha256=8113f8a18642f428f29975c3ab6f9312e5b1f56ae848ec5b7db1ea902a927560
torch-sparse @ file:///C:/Users/82770/whl/torch_sparse-0.6.16%2Bpt112cu113-cp39-cp39-win_amd64.whl#sha256=a8daf0f429b81701f026a3070f9e6e76e0c116fad696ffd87dd9975ea027a995
torch-spline-conv @ file:///C:/Users/82770/whl/torch_spline_conv-1.2.1%2Bpt112cu113-cp39-cp39-win_amd64.whl#sha256=526a967b25d629aae8e24f17f26a771210042468d73abefa6d4837325547dc1a
torch_geometric==2.4.0
torchaudio @ file:///D:/whl/torchaudio-0.12.0%2Bcu113-cp39-cp39-win_amd64.whl#sha256=7c0a36b7454f2630bb85fa9ce14bf662686aaaa67879e7211fb4b61d8acf95a8
torchvision @ file:///D:/whl/torchvision-0.13.0%2Bcu113-cp39-cp39-win_amd64.whl#sha256=c7f8588281444070f9cdfe0a074882987eec21e5ccb290bcbac336a2a960e6d1
tqdm==4.66.1
typing-inspection==0.4.1
typing_extensions==4.14.0
urllib3==2.1.0
Werkzeug==3.0.1
zipp==3.17.0

```

# Hyperparameters setting

```python
num_mec==4
num_vehicles==20
n_agent==24
(You should keep num_mec + num_vehicles = n_agent)
num_contents==100
```

# Some details

We also provide the implementation of not using GNN. You can set 'use_gnn'=False in the hyperparameters setting, if you don't want to use RNN.

 
