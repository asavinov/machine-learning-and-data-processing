# A collection of resources on machine learning, data processing and related areas

* [Analysis of different types of data](#analysis-of-different-types-of-data)
* [Data and knowledge engineering](#data-and-knowledge-engineering)
* [Asynchronous data processing](#asynchronous-data-processing)
* [Libraries, utilities tools](#libraries-utilities-tools)
* [Other resources](#other-resources)


## Analysis of different types of data

### Time series and forecasting

Books and articles:
* ARIMA Model – Complete Guide to Time Series Forecasting in Python: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python

Implementations:
* ![stars](https://img.shields.io/github/stars/facebook/prophet) 
https://github.com/facebook/prophet Prophet https://facebook.github.io/prophet/ 
* ![stars](https://img.shields.io/github/stars/unit8co/darts) 
https://github.com/unit8co/darts A python library for easy manipulation and forecasting of time series
* ![stars](https://img.shields.io/github/stars/alan-turing-institute/sktime) 
https://github.com/alan-turing-institute/sktime A unified toolbox for machine learning with time series
* ![stars](https://img.shields.io/github/stars/thuml/Time-Series-Library) 
https://github.com/thuml/Time-Series-Library A Library for Advanced Deep Time Series Models
* ![stars](https://img.shields.io/github/stars/facebookresearch/Kats) 
https://github.com/facebookresearch/Kats Kats, a kit to analyze time series data, a lightweight, easy-to-use, generalizable, and extendable framework to perform time series analysis, from understanding the key statistics and characteristics, detecting change points and anomalies, to forecasting future trends.
* ![stars](https://img.shields.io/github/stars/Nixtla/statsforecast) 
https://github.com/Nixtla/neuralforecast Lightning fast forecasting with statistical and econometric models
* ![stars](https://img.shields.io/github/stars/Nixtla/statsforecast) 
https://github.com/Nixtla/neuralforecast Scalable and user friendly neural forecasting algorithms
* ![stars](https://img.shields.io/github/stars/ourownstory/neural_prophet) 
https://github.com/ourownstory/neural_prophet Neural Network based Time-Series model
* ![stars](https://img.shields.io/github/stars/alkaline-ml/pmdarima) 
https://github.com/alkaline-ml/pmdarima A statistical library designed to fill the void in Python's time series analysis capabilities, including the equivalent of R's auto.arima function

Stock price forecasting:
* ![stars](https://img.shields.io/github/stars/huseinzol05/Stock-Prediction-Models) 
https://github.com/huseinzol05/Stock-Prediction-Models Gathers machine learning and deep learning models for Stock forecasting including trading bots and simulations
* ![stars](https://img.shields.io/github/stars/borisbanushev/stockpredictionai) 
https://github.com/borisbanushev/stockpredictionai In this noteboook I will create a complete process for predicting stock price movements. Follow along and we will achieve some pretty good results. For that purpose we will use a Generative Adversarial Network (GAN) with LSTM, a type of Recurrent Neural Network, as generator, and a Convolutional Neural Network, CNN, as a discriminator

Lists:
* ![stars](https://img.shields.io/github/stars/MaxBenChrist/awesome_time_series_in_python) 
https://github.com/MaxBenChrist/awesome_time_series_in_python This curated list contains python packages for time series analysis

### Spatial data

* ![stars](https://img.shields.io/github/stars/geopandas/geopandas) 
https://github.com/geopandas/geopandas Python tools for geographic data
* ![stars](https://img.shields.io/github/stars/giswqs/leafmap) 
https://github.com/giswqs/leafmap A Python package for geospatial analysis and interactive mapping with minimal coding in a Jupyter environment

### Audio and sound

General audio processing libraries:
* ![stars](https://img.shields.io/github/stars/librosa/librosa) 
https://github.com/librosa/librosa Python library for audio and music analysis https://librosa.org/
* ![stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis) 
https://github.com/tyiannak/pyAudioAnalysis Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications
* ![stars](https://img.shields.io/github/stars/MTG/essentia) 
https://github.com/MTG/essentia Essentia http://essentia.upf.edu/
* ![stars](https://img.shields.io/github/stars/libAudioFlux/audioFlux) 
https://github.com/libAudioFlux/audioFlux A library for audio and music analysis, feature extraction
* ![stars](https://img.shields.io/github/stars/keunwoochoi/kapre) 
https://github.com/keunwoochoi/kapre keras audio preprocessor that let you calculate STFT/melspectrograms directly in your keras model on the fly without saving them in your storage
* ![stars](https://img.shields.io/github/stars/bmcfee/resampy) 
https://github.com/bmcfee/resampy Efficient sample rate conversion in python

Text to speech (TTS)
* ![stars](https://img.shields.io/github/stars/openai/whisper) 
https://github.com/openai/whisper Robust Speech Recognition via Large-Scale Weak Supervision
* ![stars](https://img.shields.io/github/stars/fishaudio/fish-speech) 
https://github.com/fishaudio/fish-speech SOTA Open Source TTS. Fish Speech V1.5 is a leading text-to-speech (TTS) model trained on more than 1 million hours of audio data in multiple languages. Model: https://huggingface.co/fishaudio/fish-speech-1.5 Demo: https://fish.audio/ Documentation: https://speech.fish.audio/
* ![stars](https://img.shields.io/github/stars/microsoft/VibeVoice)
https://github.com/microsoft/VibeVoice Frontier Open-Source Text-to-Speech https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f
* ![stars](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice)
https://github.com/FunAudioLLM/CosyVoice Multi-lingual large voice generation model, providing inference, training and deployment full-stack ability. Models: https://huggingface.co/FunAudioLLM/models Demo: https://funaudiollm.github.io/cosyvoice3/
* ![stars](https://img.shields.io/github/stars/SWivid/F5-TTS) 
https://github.com/SWivid/F5-TTS Official code for "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" https://swivid.github.io/F5-TTS/
* ![stars](https://img.shields.io/github/stars/espnet/espnet) 
https://github.com/espnet/espnet End-to-End Speech Processing Toolkit 
* ![stars](https://img.shields.io/github/stars/facebookresearch/wav2letter) 
https://github.com/facebookresearch/wav2letter Automatic Speech Recognition Toolkit
* ![stars](https://img.shields.io/github/stars/yl4579/StyleTTS2) 
https://github.com/yl4579/StyleTTS2 StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models 
* ![stars](https://img.shields.io/github/stars/huggingface/parler-tts) 
https://github.com/huggingface/parler-tts Inference and training library for high-quality TTS models
* ![stars](https://img.shields.io/github/stars/hexgrad/kokoro) 
https://github.com/hexgrad/kokoro An inference library for Kokoro-82M. Kokoro is an open-weight TTS model with 82 million parameters https://huggingface.co/hexgrad/Kokoro-82M Demo (works in browser): https://huggingface.co/spaces/webml-community/kokoro-webgpu
* ![stars](https://img.shields.io/github/stars/QwenLM/Qwen3-TTS)
https://github.com/QwenLM/Qwen3-TTS Qwen3-TTS is an open-source series of TTS models developed by the Qwen team at Alibaba Cloud, supporting stable, expressive, and streaming speech generation, free-form voice design, and vivid voice cloning.
Hugging Face: https://huggingface.co/collections/Qwen/qwen3-tts
Demo (HF): https://huggingface.co/spaces/Qwen/Qwen3-TTS
Blog: https://qwen.ai/blog?id=qwen3tts-0115
Paper: https://github.com/QwenLM/Qwen3-TTS/blob/main/assets/Qwen3_TTS.pdf
* ![stars](https://img.shields.io/github/stars/Camb-ai/MARS5-TTS) 
https://github.com/Camb-ai/MARS5-TTS MARS5 speech model (TTS) from CAMB.AI 
* ![stars](https://img.shields.io/github/stars/DigitalPhonetics/IMS-Toucan#Toucan) 
https://github.com/DigitalPhonetics/IMS-Toucan#Toucan Multilingual and Controllable Text-to-Speech Toolkit of the Speech and Language Technologies Group at the University of Stuttgart
* ![stars](https://img.shields.io/github/stars/kyutai-labs/pocket-tts) 
https://github.com/kyutai-labs/pocket-tts A TTS that fits in your CPU (and pocket) https://huggingface.co/kyutai/pocket-tts Only English
* ![stars](https://img.shields.io/github/stars/jishengpeng/WavTokenizer) 
https://github.com/jishengpeng/WavTokenizer SOTA discrete acoustic codec models with 40 tokens per second for audio language modeling. https://huggingface.co/novateur/WavTokenizer Article: https://arxiv.org/abs/2408.16532 Demo: https://wavtokenizer.github.io
* https://huggingface.co/Supertone/supertonic-2 Supertonic 2 — Lightning Fast, On-Device TTS, Multilingual TTS

Speech recognition (ASR):

* ![stars](https://img.shields.io/github/stars/wenet-e2e/wenet)
https://github.com/wenet-e2e/wenet Production First and Production Ready End-to-End Speech Recognition Toolkit
* ![stars](https://img.shields.io/github/stars/pyannote/pyannote-audio)
https://github.com/pyannote/pyannote-audio Neural building blocks for speaker diarization: speech activity detection, speaker change detection, overlapped speech detection, speaker embedding 

Pitch trackers:
* ![stars](https://img.shields.io/github/stars/marl/crepe) 
https://github.com/marl/crepe REPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is state-of-the-art (as of 2018), outperforming popular pitch trackers such as pYIN and SWIPE https://arxiv.org/abs/1802.06182
* pYIN: https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf
* SWIPE: https://pdfs.semanticscholar.org/0fd2/6e267cfa9b6d519967ea00db4ffeac272777.pdf

Other
* ![stars](https://img.shields.io/github/stars/AddictedCS/soundfingerprinting) 
https://github.com/AddictedCS/soundfingerprinting audio acoustic fingerprinting fingerprinting in C# (advertising etc. - not speech)
* ![stars](https://img.shields.io/github/stars/OverLordGoldDragon/ssqueezepy) 
https://github.com/OverLordGoldDragon/ssqueezepy Synchrosqueezing, wavelet transforms, and time-frequency analysis in Python

Books and other resources:
* Think DSP: Digital Signal Processing in Python: http://greenteapress.com/thinkdsp/thinkdsp.pdf
  * Include a Python lib and examples in Python: https://github.com/AllenDowney/ThinkDSP 
* Digital Signal Processing: https://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/index.html
* Sound Processing with Short Time Fourier Transform: https://www.numerical-tours.com/matlab/audio_1_processing/
* Theory of digital signal processing (DSP): signals, filtration (IIR, FIR, CIC, MAF), transforms (FFT, DFT, Hilbert, Z-transform) etc. https://github.com/capitanov/dsp-theory

### Text NLP

* ![stars](https://img.shields.io/github/stars/explosion/spaCy) 
https://github.com/explosion/spaCy (in Cython) -  Industrial-strength Natural Language Processing (NLP) with Python and Cython https://spacy.io
* ![stars](https://img.shields.io/github/stars/facebookresearch/fastText) 
https://github.com/facebookresearch/fastText - Library for fast text representation and classification
* ![stars](https://img.shields.io/github/stars/RaRe-Technologies/gensim) 
https://github.com/RaRe-Technologies/gensim Topic Modelling for Humans, robust semantic analysis, topic modeling and vector-space modeling
* ![stars](https://img.shields.io/github/stars/nltk/nltk) 
https://github.com/nltk/nltk - NLTK
* ![stars](https://img.shields.io/github/stars/zalandoresearch/flair) 
https://github.com/zalandoresearch/flair A very simple framework for state-of-the-art Natural Language Processing (NLP)
* ![stars](https://img.shields.io/github/stars/wireservice/csvkit) 
https://github.com/wireservice/csvkit

Lists:
* ![stars](https://img.shields.io/github/stars/keon/awesome-nlp) 
https://github.com/keon/awesome-nlp
* ![stars](https://img.shields.io/github/stars/astorfi/Deep-Learning-NLP) 
https://github.com/astorfi/Deep-Learning-NLP
* ![stars](https://img.shields.io/github/stars/brianspiering/awesome-dl4nlp) 
https://github.com/brianspiering/awesome-dl4nlp

### LLMs

* ![stars](https://img.shields.io/github/stars/huggingface/pytorch-transformers) 
https://github.com/huggingface/pytorch-transformers A library of state-of-the-art pretrained models for Natural Language Processing (NLP)
* ![stars](https://img.shields.io/github/stars/VectifyAI/PageIndex) 
https://github.com/VectifyAI/PageIndex PageIndex: Document Index for Vectorless, Reasoning-based RAG
* ![stars](https://img.shields.io/github/stars/p-e-w/heretic) 
https://github.com/p-e-w/heretic Fully automatic censorship removal for language models

### Video

* ![stars](https://img.shields.io/github/stars/iperov/DeepFaceLab) 
https://github.com/iperov/DeepFaceLab DeepFaceLab is a tool that utilizes machine learning to replace faces in videos.
* ![stars](https://img.shields.io/github/stars/Zulko/moviepy) 
https://github.com/Zulko/moviepy - Video editing with Python

### Images

* ![stars](https://img.shields.io/github/stars/opencv/opencv) 
https://github.com/opencv/opencv Open Source Computer Vision Library
* ![stars](https://img.shields.io/github/stars/facebookresearch/sam2) 
https://github.com/facebookresearch/sam2 The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model
* ![stars](https://img.shields.io/github/stars/AlexeyAB/darknet) 
YOLO v4: https://github.com/AlexeyAB/darknet
  * https://pjreddie.com/darknet/yolo/
  * https://medium.com/what-is-artificial-intelligence/the-yolov4-algorithm-introduction-to-you-only-look-once-version-4-real-time-object-detection-5fd8a608b0fa
  * https://medium.com/@whats_ai/what-is-the-yolo-algorithm-introduction-to-you-only-look-once-real-time-object-detection-f26aa81475f2
  * How to train YOLO to detect your own objects: https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
* ![stars](https://img.shields.io/github/stars/jantic/DeOldify) 
https://github.com/jantic/DeOldify A Deep Learning based project for colorizing and restoring old images (and video!)
* ![stars](https://img.shields.io/github/stars/microsoft/Bringing-Old-Photos-Back-to-Life) 
https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life Bringing Old Photo Back to Life (CVPR 2020 oral)
* ![stars](https://img.shields.io/github/stars/python-pillow/Pillow) 
https://github.com/python-pillow/Pillow PIL is the Python Imaging Library
* ![stars](https://img.shields.io/github/stars/scikit-image/scikit-image) 
https://github.com/scikit-image/scikit-image Image processing in Python
* ![stars](https://img.shields.io/github/stars/symisc/sod) 
https://github.com/symisc/sod An Embedded, Modern Computer Vision & Machine Learning Library
* DeepDetect https://github.com/beniz/deepdetect/tree/master/demo/objsearch http://www.deepdetect.com/
  * Object similarity search: https://github.com/beniz/deepdetect/tree/master/demo/objsearch
    * for suggesting labels (bounding box)

### Graphs, RDFs etc.

Graph stores:
* ![stars](https://img.shields.io/github/stars/eBay/beam) 
https://github.com/eBay/beam A distributed knowledge graph store 
* ![stars](https://img.shields.io/github/stars/gchq/Gaffer) 
https://github.com/gchq/Gaffer A large-scale entity and relation database supporting aggregation of properties 

Databases:
* ![stars](https://img.shields.io/github/stars/dgraph-io/dgraph) 
https://github.com/dgraph-io/dgraph Dgraph Fast, Distributed Graph DB https://dgraph.io/
* Neo4j
* ![stars](https://img.shields.io/github/stars/janusgraph/janusgraph) 
https://github.com/janusgraph/janusgraph https://janusgraph.org/ 
  JanusGraph [2] has support for a lot of different backends, built by the old team behind TitanDB
* ![stars](https://img.shields.io/github/stars/arangodb/arangodb) 
https://github.com/arangodb/arangodb ArangoDB is a native multi-model database with flexible data models for documents, graphs, and key-values. Best open source graph database is ArrangoDB they have master to master cluster

Visualizations and dashboards:
* graphviz

### Statistics
* ![stars](https://img.shields.io/github/stars/lebigot/uncertainties) 
https://github.com/lebigot/uncertainties Transparent calculations with uncertainties on the quantities involved (aka "error propagation"); calculation of derivatives
* ![stars](https://img.shields.io/github/stars/facebookarchive/bootstrapped) 
https://github.com/facebookarchive/bootstrapped Generate bootstrapped confidence intervals for A/B testing in Python

### Reinforcement learning

* ![stars](https://img.shields.io/github/stars/tensorflow/agents) 
https://github.com/tensorflow/agents TF-Agents: A reliable, scalable and easy to use TensorFlow library for Contextual Bandits and Reinforcement Learning
  * https://github.com/tensorflow/agents/blob/master/docs/tutorials/0_intro_rl.ipynb

### AI, data mining, machine learning algorithms

Resources:
* ![stars](https://img.shields.io/github/stars/astorfi/TensorFlow-World-Resources) 
https://github.com/astorfi/TensorFlow-World-Resources
* ![stars](https://img.shields.io/github/stars/astorfi/Deep-Learning-World) 
https://github.com/astorfi/Deep-Learning-World

Algorithms:
* XGBoost, CatBoost, LightGBM
* ![stars](https://img.shields.io/github/stars/spotify/annoy) 
https://github.com/spotify/annoy Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk
  * USP: ability to use static files as indexes, share index across process, that is, in-memory and efficient and multi-process
  * e.g, music recommendation in Spotify, similar images (for labeling etc.)  

## Data and knowledge engineering

### Feature engineering

Time series:
* ![stars](https://img.shields.io/github/stars/blue-yonder/tsfresh) 
https://github.com/blue-yonder/tsfresh
* ![stars](https://img.shields.io/github/stars/bukosabino/ta) 
https://github.com/bukosabino/ta Technical Analysis Library using Pandas (Python)
* ![stars](https://img.shields.io/github/stars/benfulcher/hctsa) 
https://github.com/benfulcher/hctsa Highly comparative time-series analysis code repository
* ![stars](https://img.shields.io/github/stars/cerlymarco/tsmoothie) 
https://github.com/cerlymarco/tsmoothie A python library for time-series smoothing and outlier detection in a vectorized way
* ![stars](https://img.shields.io/github/stars/chlubba/catch22) 
https://github.com/chlubba/catch22 catch-22: CAnonical Time-series CHaracteristics

Feature extraction:
* ![stars](https://img.shields.io/github/stars/Featuretools/featuretools) 
https://github.com/Featuretools/featuretools
* ![stars](https://img.shields.io/github/stars/mapbox/robosat) 
https://github.com/mapbox/robosat - feature extraction from aerial and satellite imagery. Semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds
* ![stars](https://img.shields.io/github/stars/tyarkoni/pliers) 
https://github.com/tyarkoni/pliers - Automated feature extraction in Python (audio/video)

Feature selection:
* ![stars](https://img.shields.io/github/stars/WillKoehrsen/feature-selector) 
https://github.com/WillKoehrsen/feature-selector Feature selector is a tool for dimensionality reduction of machine learning datasets.
  Find unnecessary (redundant) features using simple methods: missing values, high correlation etc.
  See article: https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
* ![stars](https://img.shields.io/github/stars/scikit-learn-contrib/boruta_py) 
https://github.com/scikit-learn-contrib/boruta_py - Python implementations of the Boruta all-relevant feature selection method
* ![stars](https://img.shields.io/github/stars/EpistasisLab/scikit-rebate) 
https://github.com/EpistasisLab/scikit-rebate - A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning
* ![stars](https://img.shields.io/github/stars/abhayspawar/featexp) 
https://github.com/abhayspawar/featexp eature exploration for supervised learning

Hyper-parameter optimization:
* ![stars](https://img.shields.io/github/stars/pfnet/optuna) 
https://github.com/pfnet/optuna - A hyperparameter optimization framework https://optuna.org
* ![stars](https://img.shields.io/github/stars/EpistasisLab/tpot) 
https://github.com/EpistasisLab/tpot Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
* ![stars](https://img.shields.io/github/stars/instacart/lore) 
https://github.com/instacart/lore Lore makes machine learning approachable for Software Engineers and maintainable for Machine Learning Researchers
* ![stars](https://img.shields.io/github/stars/ClimbsRocks/auto_ml) 
https://github.com/ClimbsRocks/auto_ml [UNMAINTAINED] Automated machine learning for analytics & production
* ![stars](https://img.shields.io/github/stars/machinalis/featureforge) 
https://github.com/machinalis/featureforge - creating and testing machine learning features, with a scikit-learn compatible API

Lists:
* ![stars](https://img.shields.io/github/stars/MaxHalford/xam) 
https://github.com/MaxHalford/xam - Personal data science and machine learning toolbox
* ![stars](https://img.shields.io/github/stars/xiaoganghan/awesome-feature-engineering) 
https://github.com/xiaoganghan/awesome-feature-engineering

### Feature stores

* ![stars](https://img.shields.io/github/stars/feast-dev/feast) 
https://github.com/feast-dev/feast Feature Store for Machine Learning https://feast.dev
* ![stars](https://img.shields.io/github/stars/feathr-ai/feathr) 
https://github.com/feathr-ai/feathr An Enterprise-Grade, High Performance Feature Store 
* ![stars](https://img.shields.io/github/stars/logicalclocks/hopsworks) 
https://github.com/logicalclocks/hopsworks Hopsworks - Data-Intensive AI platform with a Feature Store

Resources:
* Rethinking Feature Stores: https://medium.com/@changshe/rethinking-feature-stores-74963c2596f0
* ML Feature Stores: A Casual Tour:
  * https://medium.com/@farmi/ml-feature-stores-a-casual-tour-fc45a25b446a
  * https://medium.com/@farmi/ml-feature-stores-a-casual-tour-30a93e16d213
  * https://medium.com/@farmi/ml-feature-stores-a-casual-tour-3-3-877557792c43
* How to Build your own Feature Store: https://www.logicalclocks.com/blog/how-to-build-your-own-feature-store
* Michelangelo Palette: A Feature Engineering Platform at Uber: https://www.infoq.com/presentations/michelangelo-palette-uber/
* Zipline: Airbnb’s Machine Learning Data Management Platform https://databricks.com/session/zipline-airbnbs-machine-learning-data-management-platform
* Fact Store at Scale for Netflix Recommendations: https://databricks.com/session/fact-store-scale-for-netflix-recommendations

### Model management

TBD

### AutoML

AutoML libs:
* ![stars](https://img.shields.io/github/stars/microsoft/nni) 
https://github.com/microsoft/nni An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning
* ![stars](https://img.shields.io/github/stars/EpistasisLab/tpot) 
https://github.com/EpistasisLab/tpot A Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
* ![stars](https://img.shields.io/github/stars/keras-team/autokeras) 
https://github.com/keras-team/autokeras AutoML library for deep learning
* ![stars](https://img.shields.io/github/stars/automl/auto-sklearn) 
https://github.com/automl/auto-sklearn Automated Machine Learning with scikit-learn
* ![stars](https://img.shields.io/github/stars/google/automl) 
https://github.com/google/automl Google Brain AutoML
* ![stars](https://img.shields.io/github/stars/tensorflow/adanet) 
https://github.com/tensorflow/adanet Fast and flexible AutoML with learning guarantees
* ![stars](https://img.shields.io/github/stars/mindsdb/mindsdb) 
https://github.com/mindsdb/mindsdb Machine Learning in one line of code
* ![stars](https://img.shields.io/github/stars/pycaret/pycaret) 
https://github.com/pycaret/pycaret An open source, low-code machine learning library in Python
* ![stars](https://img.shields.io/github/stars/AxeldeRomblay/MLBox) 
https://github.com/AxeldeRomblay/MLBox MLBox is a powerful Automated Machine Learning python library
* ![stars](https://img.shields.io/github/stars/automl/HpBandSter) 
https://github.com/automl/HpBandSter a distributed Hyperband implementation on Steroids
* ![stars](https://img.shields.io/github/stars/microsoft/FLAML) 
https://github.com/microsoft/FLAML A fast and lightweight AutoML library
* ![stars](https://img.shields.io/github/stars/automl/autoweka) 
https://github.com/automl/autoweka
* ![stars](https://img.shields.io/github/stars/shankarpandala/lazypredict) 
https://github.com/shankarpandala/lazypredict Lazy Predict help build a lot of basic models without much code and helps understand which models works better without any parameter tuning

Systems and companies:
* DataRobot
* DarwinAI
* H2O.ai Dreverless AI
* OneClick.ai

### Workflow/job management, Data integration, ETL, web scrapping, orchestration

For workflows, it is important how the following concepts are implemented:
* Workflow task: what can be executed and how we specify the tasks to be executed (programmatically, declaratively etc.)
* Dependencies: when and how tasks are started. It is not only about simple conditions like whether a (previous) task has been finished and with which status. Conditions might be much more complex and be essentially separate tasks.
  * How to define conditional execution where execution of a task depends on dynamic conditions
  * Choosing a task to be executed dynamically, that is, task to be executed is not known at the time of graph definition
* Triggering workflows: How a whole workflow execution can be started (from inside or outside). These functions make workflow managers similar to asynchronous systems:
  * From external systems, for example, by listening to some protocol
  * Synchronous execution, for example, using schedulers like once per day

A conventional workflow management system deals with task arguments and task return values. The goal here is to make the whole workflow with all its tasks data-aware, for example, by sharing some data. For the point of view of data processing, there is an additional aspect:
* If it is a graph of data processing, then the question is where and how the system takes the data properties into account:
  * Data state and data ranges like only data for the last month
  * Data structure like columns and tables, for example, only these tables
* Such data-aware workflows are most important for data-driven business, and the question is what they know about the data and how this knowledge is used to manage and execute workflow.

General purpose workflow management systems:
* ![stars](https://img.shields.io/github/stars/apache/airflow) 
https://github.com/apache/airflow https://airflow.apache.org A platform to programmatically author, schedule, and monitor workflows. "Airflow is based on DAG representation and doesn’t have a concept of input or output, just of flow."
  * Tasks (Operators). Pluggable Python classes with many conventional execution types provided like `PythonOperator` or `BashOperator`. Note that these are task types which require custom task code for parameterization and instantiation. For `PythonOperator`, the custom code is provided as a Python function: `python_callable=my_task_fn`. Function arguments are passed in another parameter `op_kwargs`.
    * https://marclamberti.com/blog/airflow-pythonoperator/
  * Data. Small data sharing between tasks is performed via XCOM (cross-communication messages).
    * https://marclamberti.com/blog/airflow-xcom/
  * Dependencies. It is done programmatically and statically, for example: `my_task_1 >> my_task_2 >> [my_task_3, my_task_4]`
  * Scheduling. It is possible to specify `start_date` and `schedule_interval` (using CRON expression or `timedelta` object).
* ![stars](https://img.shields.io/github/stars/celery/celery) 
https://github.com/celery/celery Distributed Task Queue http://celeryproject.org/
* ![stars](https://img.shields.io/github/stars/spotify/luigi) 
https://github.com/spotify/luigi build complex pipelines of (long-running) batch jobs like Hadoop jobs, Spark jobs, dumping data to/from databases, running machine learning algorithms, Python snippet etc. "Luigi is based on pipelines of tasks that share input and output information and is target-based".
  * Dependencies. "Luigi doesn’t use DAGs. Instead, Luigi refers to "tasks" and "targets." Targets are both the results of a task and the input for the next task." "Luigi has 3 steps to construct a pipeline: `requires()` defines the dependencies between the tasks, `output()` defines the the target of the task, `run()` defines the computation performed by each task"
  * Scheduling. "Luigi ... has a central scheduler *and* custom calendar schedule capabilities, providing users with lots of flexibility" (in contrast to Airflow).
* ![stars](https://img.shields.io/github/stars/prefecthq/prefect) 
https://github.com/prefecthq/prefect The easiest way to automate your data
* ![stars](https://img.shields.io/github/stars/azkaban/azkaban) 
https://github.com/azkaban/azkaban Azkaban workflow manager
* ![stars](https://img.shields.io/github/stars/dagster-io/dagster) 
https://github.com/dagster-io/dagster A data orchestrator for machine learning, analytics, and ETL
* Oozie 
* ![stars](https://img.shields.io/github/stars/ploomber/ploomber) 
https://github.com/ploomber/ploomber The fastest way to build data pipelines. Develop iteratively, deploy anywhere
* ![stars](https://img.shields.io/github/stars/d6t/d6tflow) 
https://github.com/d6t/d6tflow Python library for building highly effective data science workflows (on top of luigi)
* ![stars](https://img.shields.io/github/stars/grailbio/reflow) 
https://github.com/grailbio/reflow A language and runtime for distributed, incremental data processing in the cloud

ML workflow, pipelines, training, deployment etc.
* ![stars](https://img.shields.io/github/stars/kubeflow/kubeflow) 
https://github.com/kubeflow/kubeflow Machine Learning Toolkit for Kubernetes
* ![stars](https://img.shields.io/github/stars/mlflow/mlflow) 
https://github.com/mlflow/mlflow Open source platform for the machine learning lifecycle
* ![stars](https://img.shields.io/github/stars/cortexlabs/cortex) 
https://github.com/cortexlabs/cortex Cloud native model serving infrastructure. Cortex is an open source platform for deploying machine learning models as production web services
* ![stars](https://img.shields.io/github/stars/Netflix/metaflow) 
https://github.com/Netflix/metaflow Build and manage real-life data science projects with ease

Data science support and tooling
* ![stars](https://img.shields.io/github/stars/drivendata/cookiecutter-data-science) 
https://github.com/drivendata/cookiecutter-data-science A logical, reasonably standardized, but flexible project structure for doing and sharing data science work
* ![stars](https://img.shields.io/github/stars/orchest/orchest) 
https://github.com/orchest/orchest A new kind of IDE for Data Science

ETL and data integration:
* ![stars](https://img.shields.io/github/stars/airbytehq/airbyte) 
https://github.com/airbytehq/airbyte Airbyte is an open-source EL(T) platform that helps you replicate your data in your warehouses, lakes and databases
* ![stars](https://img.shields.io/github/stars/mara/data-integration) 
https://github.com/mara/data-integration A lightweight opinionated ETL framework, halfway between plain scripts and Apache Airflow
* ![stars](https://img.shields.io/github/stars/python-bonobo/bonobo) 
https://github.com/python-bonobo/bonobo Extract Transform Load for Python 3.5+ https://www.bonobo-project.org/
* https://gitlab.com/meltano/meltano Meltano: ELT for the DataOps era — Open source, self-hosted, CLI-first, debuggable, and extensible. Embraces Singer and its library of connectors, and leverages dbt for transformation https://meltano.com/

Stream processing:
* ![stars](https://img.shields.io/github/stars/robinhood/faust) 
https://github.com/robinhood/faust Stream processing library, porting the ideas from Kafka Streams to Python
* ![stars](https://img.shields.io/github/stars/airbnb/streamalert) 
https://github.com/airbnb/streamalert StreamAlert is a serverless, realtime data analysis framework which empowers you to ingest, analyze, and alert on data from any environment, using datasources and alerting logic you define
* ![stars](https://img.shields.io/github/stars/nerevu/riko) 
https://github.com/nerevu/riko A Python stream processing engine modeled after Yahoo! Pipes

Web scrapping
* ![stars](https://img.shields.io/github/stars/scrapy/scrapy) 
https://github.com/scrapy/scrapy - create spiders bots that scan website pages and collect structured data

### Data labeling

* ![stars](https://img.shields.io/github/stars/snorkel-team/snorkel) 
https://github.com/snorkel-team/snorkel A system for quickly generating training data with weak supervision
* ![stars](https://img.shields.io/github/stars/Cartucho/yolo-boundingbox-labeler-GUI) 
https://github.com/Cartucho/yolo-boundingbox-labeler-GUI Label images and video for Computer Vision applications (YOLO format)

### Visualization and VA

* Matplotlib http://matplotlib.org/
* Bokeh http://bokeh.pydata.org/en/latest/ The Bokeh library creates interactive and scalable visualizations in a browser using JavaScript widgets
* Seaborn http://stanford.edu/~mwaskom/software/seaborn/# https://seaborn.pydata.org/ - a higher-level API based on the matplotlib library. It contains more suitable default settings for processing charts. Also, there is a rich gallery of visualizations. 
* ![stars](https://img.shields.io/github/stars/lightning-viz/lightning) 
https://github.com/lightning-viz/lightning http://lightning-viz.org/ Data Visualization Server. Lightning provides API-based access to reproducible web visualizations
* Plotly https://plot.ly/
* Pandas built-in plotting http://pandas.pydata.org/pandas-docs/stable/visualization.html
* https://github.com/holoviz/holoviews With Holoviews, your data visualizes itself
* ![stars](https://img.shields.io/github/stars/vispy/vispy) 
https://github.com/vispy/vispy VisPy: interactive scientific visualization in Python. VisPy is a high-performance interactive 2D/3D data visualization library leveraging the computational power of modern Graphics Processing Units (GPUs) through the OpenGL library to display very large datasets
* ![stars](https://img.shields.io/github/stars/sirrice/pygg) 
http://www.github.com/sirrice/pygg ggplot2 syntax in python. Actually wrapper around Wickham's ggplot2 in R
* ![stars](https://img.shields.io/github/stars/altair-viz/altair) 
https://github.com/altair-viz/altair Declarative statistical visualization library for Python
* ![stars](https://img.shields.io/github/stars/ContextLab/hypertools) 
https://github.com/ContextLab/hypertools A Python toolbox for gaining geometric insights into high-dimensional data
* ![stars](https://img.shields.io/github/stars/DistrictDataLabs/yellowbrick) 
https://github.com/DistrictDataLabs/yellowbrick Visual analysis and diagnostic tools to facilitate machine learning model selection
* ![stars](https://img.shields.io/github/stars/AutoViML/AutoViz) 
https://github.com/AutoViML/AutoViz Automatically Visualize any dataset, any size with a single line of code. Created by Ram Seshadri. Collaborators Welcome. Permission Granted upon Request

Dashboards:
* ![stars](https://img.shields.io/github/stars/grafana/grafana) 
https://github.com/grafana/grafana Grafana (time series) https://grafana.com/
  * Integration with InfluxDB
* ![stars](https://img.shields.io/github/stars/apache/superset) 
Superset https://github.com/apache/superset Apache Superset is a Data Visualization and Data Exploration Platform
  * Integration with druid.io database https://github.com/apache/druid
* ![stars](https://img.shields.io/github/stars/metabase/metabase) 
https://github.com/metabase/metabase Metabase https://metabase.com/
* ![stars](https://img.shields.io/github/stars/getredash/redash) 
https://github.com/getredash/redash Redash https://redash.io/
* ![stars](https://img.shields.io/github/stars/apache/zeppelin) 
https://github.com/apache/zeppelin Zeppelin Apache https://zeppelin.apache.org/
* Mixpanel https://mixpanel.com/
* ![stars](https://img.shields.io/github/stars/ankane/blazer) 
Blazer https://github.com/ankane/blazer
* ![stars](https://img.shields.io/github/stars/bdash-app/bdash) 
Bdash https://github.com/bdash-app/bdash
* Datamill, ?
* wagonhq, ?
* ![stars](https://img.shields.io/github/stars/nocodb/nocodb) 
https://github.com/nocodb/nocodb The Open Source Airtable alternative. Turns any MySQL, PostgreSQL, SQL Server, SQLite & MariaDB into a smart-spreadsheet

Publishing notebooks (from github etc.):
* https://mybinder.org/
* http://colab.research.google.com/ a kind of Jupyter notebooks stored in Google Drive

## Asynchronous data processing

### What is asynchronous data processing

TBD

### Reactive programming

Approaches to asynchronous programming:

* Callback model:
  * A callback function is provided as part of an asynchronous call
  * The call is non-blocking and the source program continues execution
  * It is not possible to await for the return (it is essentially done by the callback function)
  * The callback function can be viewed as a one-time listener for a return event, that is, it represents code which consumes the result
  * The source code where the call is made and the consumer of the result are in different functions and cannot share (local) context
  * The callback function may make its own asynchronous calls which leads to a "callback hell"

* Future/promise:
  * An asynchronous call is made as usual but return a special wrapper object
  * A callback function is not specified and is not used
  * The returned result is consumed by the code which follows the call (as opposed to its use in a separate callback function)
  * The future/promise is supposed to be awaited. Awaiting denotes a point where we say that the next instruction needs the result
  * The awaiting point is like a one-time single-value listener for the result where the program execution is suspended until the return event is received

Resources:
* The Reactive Manifesto: https://www.reactivemanifesto.org/
* Reactive Design Patterns: https://www.reactivedesignpatterns.com/
* Reactive programming in Python
* Reactive programming in Java
* Reactive programming in C#

### Reactive streaming

* Listeners:
  * A callback function is registered and then automatically called for each incoming event
  * Callback functions are (normally) called only sequentially, that is, next event can be processed only when the previous event has been processed by the previous callback invocation. Callbacks are not executed concurrently.
  * The result of a callback invocation is frequently needed because it is executed by the event producer

* Reactive streams:
  * It is a graph of producers and consumers
  * Consumers and producers are not supposed to communicate in a free manner by sending messages to each other where a sender knows the address(s) of the receivers
  * An element declares which messages it produces but it is unaware of who will subscribe to and consume its messages, and how they will be used (in contrast to the actor model)
  * An element must know what kind of messages it needs and explicitly subscribe to specific producers - messages will not come automatically just because somebody wants to send them to us
  * For data processing, reative streams provide a number of operators which can be applied to an input stream(s) and produce an output stream
  * Resources:
    * https://www.reactive-streams.org/
    * ![stars](https://img.shields.io/github/stars/reactor/reactor) https://github.com/reactor/reactor
    * http://reactivex.io/ An API for asynchronous programming with observable streams:
      * ![stars](https://img.shields.io/github/stars/ReactiveX/RxJava) https://github.com/ReactiveX/RxJava Reactive Extensions for the JVM – a library for composing asynchronous and event-based programs using observable sequences for the Java VM
      * ![stars](https://img.shields.io/github/stars/ReactiveX/RxPY) https://github.com/ReactiveX/RxPY Reactive Extensions for Python
      * etc. https://github.com/ReactiveX

* Actor model:
  * Each element has an identifier (address, reference) which is used by other actors to send messages
  * A sender (producer) must know what it wants to do when it sends messages and it must know its destination
  * Receiving elements can receive messages from any other element and their task is to respond according to their logic (as expected by the sender)
  * Actors receive messages without subscribing to any producer (as opposed to the actor model where you will not receive anything until you subscribe to some event producer)
  * Each actor has a handler (callback) which is invoked for processing incoming messages
  * Actors are supposed to have a state and frequently it is why we want to define different actors
  * Resources:
    * ![stars](https://img.shields.io/github/stars/akka/akka) https://github.com/akka/akka Build highly concurrent, distributed, and resilient message-driven applications on the JVM
    * ![stars](https://img.shields.io/github/stars/eclipse-vertx/vert.x) https://github.com/eclipse-vertx/vert.x Vert.x is a tool-kit for building reactive applications on the JVM
    * ![stars](https://img.shields.io/github/stars/quantmind/pulsar) [ARCHIVED] https://github.com/quantmind/pulsar Event driven concurrent framework for Python https://quantmind.github.io/pulsar/index.html Pulsar implements two layers of components on top of python asyncio module: the actor layer and the application framework
    * ![stars](https://img.shields.io/github/stars/jodal/pykka) https://github.com/jodal/pykka Python implementation of the actor model, which makes it easier to build concurrent applications
    * ![stars](https://img.shields.io/github/stars/kquick/Thespian) https://github.com/kquick/Thespian Python Actor concurrency library
    * https://gitlab.com/python-actorio/actorio Actorio - a simple actor framework for asyncio

### Event loops vs. threads

* Both a thread task and an event loop task are executed until finished, that is, the code to execute is provided as a procedure
* Thread tasks are dispatched by the system (not application) while dispatching logic of event tasks is part of the application
* At each moment, there is a fixed number of threads concurrently executed by one process. The number of concurrently executed event loop tasks is not limited.
* Thread tasks are (automatically) switched at the instruction level and the dispatcher is unaware of the needs of this thread or the application. Event loop tasks are switched at the level of logical application units depending on what this application needs.
* In a multi-thread application, we need to manage the threads ourselves, e.g., by creating and deleting them. In an event loop application, the tasks (starting, suspending, finishing) is managed by the event loop manager.
* In an event loop application, tasks specify dependencies on other tasks, and these points are used while dispatching the execution of tasks. Threads cannot declare dependencies on the results provided by other tasks. If we need some external result, then the thread has to wait. This logic has to be implemented manually and the system dispatcher is unaware of these dependencies.

Event loops: 
* ![stars](https://img.shields.io/github/stars/libuv/libuv) 
https://github.com/libuv/libuv Cross-platform asynchronous I/O
* ![stars](https://img.shields.io/github/stars/libevent/libevent) 
https://github.com/libevent/libevent Event notification library
* ![stars](https://img.shields.io/github/stars/enki/libev) 
https://github.com/enki/libev Full-featured high-performance event loop loosely modelled after libevent

Resources:
* ![stars](https://img.shields.io/github/stars/timofurrer/awesome-asyncio) 
https://github.com/timofurrer/awesome-asyncio Python asyncio

### Async networking libraries

* ![stars](https://img.shields.io/github/stars/gevent/gevent) 
https://github.com/gevent/gevent coroutine - based Python networking library. "systems like gevent use lightweight threads to offer performance comparable to asynchronous systems, but they do not actually make things asynchronous"
  * greenlet to provide a high-level synchronous API 
    * on top of the libev or libuv event loop (like libevent)

* ![stars](https://img.shields.io/github/stars/eventlet/eventlet) 
https://github.com/eventlet/eventlet concurrent networking library for Python
  * epoll or kqueue or libevent for highly scalable non-blocking I/O

* ![stars](https://img.shields.io/github/stars/aio-libs/aiohttp) 
https://github.com/aio-libs/aiohttp Asynchronous HTTP client/server framework for asyncio and Python

* ![stars](https://img.shields.io/github/stars/twisted/twisted) 
https://github.com/twisted/twisted Event-driven networking engine written in Python. 
  * Twisted projects variously support TCP, UDP, SSL/TLS, IP multicast, Unix domain sockets, many protocols (including HTTP, XMPP, NNTP, IMAP, SSH, IRC, FTP, and others), and much more.
  * Twisted supports all major system event loops:
    * select (all platforms), 
    * poll (most POSIX platforms), 
    * epoll (Linux), 
    * kqueue (FreeBSD, macOS), 
    * IOCP (Windows), 
    * various GUI event loops (GTK+2/3, Qt, wxWidgets)

### Async web frameworks

* ![stars](https://img.shields.io/github/stars/tornadoweb/tornado) 
https://github.com/tornadoweb/tornado Python web framework and asynchronous networking library 
  * "Tornado is integrated with the standard library asyncio module and shares the same event loop (by default since Tornado 5.0). In general, libraries designed for use with asyncio can be mixed freely with Tornado." 
  * Some async client Libraries built on tornado.ioloop:
    * DynamoDB, CouchDB, Hbase, MongoDB, MySQL, PostgresQL, PrestoDB, RethinkDB
    * AMQP, NATS, RabbitMQ, SMTP
    * DNS, Memcached, Reis
    * etc. https://github.com/tornadoweb/tornado/wiki/Links
* ![stars](https://img.shields.io/github/stars/tiangolo/fastapi) 
https://github.com/tiangolo/fastapi FastAPI
* ![stars](https://img.shields.io/github/stars/huge-success/sanic) 
https://github.com/huge-success/sanic Sanic
* ![stars](https://img.shields.io/github/stars/vibora-io/vibora) 
https://github.com/vibora-io/vibora Like Sanic but even faster
* https://gitlab.com/pgjones/quart API compatible with Flask 
* ![stars](https://img.shields.io/github/stars/Pylons/pyramid) 
https://github.com/Pylons/pyramid Python web framework https://trypyramid.com/ (it seems to be a conventional web framework)

### Utilities

Retry libraries:
* ![stars](https://img.shields.io/github/stars/jd/tenacity) https://github.com/jd/tenacity - originates from a fork of retrying
* ![stars](https://img.shields.io/github/stars/rholder/retrying) https://github.com/rholder/retrying - not supported anymore
* ![stars](https://img.shields.io/github/stars/litl/backoff) https://github.com/litl/backoff
* ![stars](https://img.shields.io/github/stars/invl/retry) https://github.com/invl/retry
* ![stars](https://img.shields.io/github/stars/channable/opnieuw) https://github.com/channable/opnieuw

## Libraries, utilities, tools

### Python

Resources:
* Top 10 Python libraries of 2017: https://tryolabs.com/blog/2017/12/19/top-10-python-libraries-of-2017/
* https://github.com/vinta/awesome-python - A curated list of awesome Python frameworks, libraries, software and resources
* Checklist for Python libraries APIs: http://python.apichecklist.com/

### Tools

* Data structures:
  * ![stars](https://img.shields.io/github/stars/mahmoud/boltons) https://github.com/mahmoud/boltons boltons should be builtins
  * ![stars](https://img.shields.io/github/stars/pytoolz/toolz) https://github.com/pytoolz/toolz List processing tools and functional utilities (replaces itertools and functools)
  * Zict: Composable Mutable Mappings
  * HeapDict: a heap with decrease-key and increase-key operations
  * sortedcontainers: Python Sorted Container Types: SortedList, SortedDict, and SortedSet

* Networking:
  * certifi: A carefully curated collection of Root Certificates for validating the trustworthiness of SSL certificates while verifying the identity of TLS hosts
  * urllib3: HTTP library with thread-safe connection pooling, file post support, sanity friendly, and more

Other:
* click: creating beautiful command line interfaces
* chardet

### Formats, persistence and serialization

* ![stars](https://img.shields.io/github/stars/wesm/feather) 
https://github.com/wesm/feather  fast, interoperable binary data frame storage for Python, R, and more powered by Apache Arrow
* ![stars](https://img.shields.io/github/stars/apache/arrow) 
https://github.com/apache/arrow standardized language-independent columnar memory format for flat and hierarchical data
* ![stars](https://img.shields.io/github/stars/Blosc/bcolz) 
https://github.com/Blosc/bcolz A columnar data container that can be compressed
* cloudpickle: serialize Python constructs not supported by the default pickle module from the Python standard library (lambdas etc.)
* ![stars](https://img.shields.io/github/stars/PyTables/PyTables) 
https://github.com/PyTables/PyTables - A Python package to manage extremely large amounts of data http://www.pytables.org
  * based on hdf5 https://www.hdfgroup.org/ HDF supports n-dimensional datasets and each element in the dataset may itself be a complex object.
* ![stars](https://img.shields.io/github/stars/h5py/h5py) 
https://github.com/h5py/h5py HDF5 for Python -- The h5py package is a Pythonic interface to the HDF5 binary data format

### Authentication, Authorization, Security

Identity and Access Management
* ![stars](https://img.shields.io/github/stars/keycloak/keycloak) 
https://github.com/keycloak/keycloak
* ![stars](https://img.shields.io/github/stars/ory/hydra) 
https://github.com/ory/hydra OAuth2 Server and OpenID Certified™ OpenID Connect Provider written in Go - cloud native, security-first, open source API security for your infrastructure. SDKs for any language
* ![stars](https://img.shields.io/github/stars/goauthentik/authentik) 
https://github.com/goauthentik/authentik The authentication glue you need
* ![stars](https://img.shields.io/github/stars/jpadilla/pyjwt) 
https://github.com/jpadilla/pyjwt JSON Web Token implementation in Python
* ![stars](https://img.shields.io/github/stars/apache/shiro) 
https://github.com/apache/shiro Apache Shiro is a powerful and easy-to-use Java security framework that performs authentication, authorization, cryptography, and session management
* ![stars](https://img.shields.io/github/stars/lepture/authlib) 
https://github.com/lepture/authlib The ultimate Python library in building OAuth, OpenID Connect clients and servers. JWS,JWE,JWK,JWA,JWT included
* ![stars](https://img.shields.io/github/stars/RichardKnop/go-oauth2-server) 
https://github.com/RichardKnop/go-oauth2-server A standalone, specification-compliant, OAuth2 server written in Golang. 
* ![stars](https://img.shields.io/github/stars/OpenIDC/pyoidc) 
https://github.com/OpenIDC/pyoidc A Python OpenID Connect implementation
* https://www.gluu.org/

Secrets management, encryption as a service, and privileged access management. A secret is anything that you want to tightly control access to, such as API keys, passwords, certificates, and more.
* ![stars](https://img.shields.io/github/stars/hashicorp/vault) 
https://github.com/hashicorp/vault A tool for secrets management, encryption as a service, and privileged access management

Policy Enforcement Point, Identity And Access Proxy (IAP), Zero-Trust Network Architecture, i.e. a reverse proxy in front of your upstream API or web server that rejects unauthorized requests and forwards authorized ones to your server. 
* ![stars](https://img.shields.io/github/stars/ory/oathkeeper) 
https://github.com/ory/oathkeeper ORY Oathkeeper 

Resources:
* ![stars](https://img.shields.io/github/stars/dwyl/learn-json-web-tokens) 
https://github.com/dwyl/learn-json-web-tokens -  Learn how to use JSON Web Token (JWT) to secure your next Web App! (Tutorial/Example with Tests)

### Linux and OS

Resources:
* https://0xax.gitbooks.io/linux-insides/content/ - Linux inside
* https://john-millikin.com/unix-syscalls - UNIX Syscalls

### Platform and servers

Load balancing and proxy:
* ![stars](https://img.shields.io/github/stars/fatedier/frp) https://github.com/fatedier/frp
* ![stars](https://img.shields.io/github/stars/traefik/traefik) https://github.com/traefik/traefik
* ![stars](https://img.shields.io/github/stars/caddyserver/caddy) https://github.com/caddyserver/caddy
* ![stars](https://img.shields.io/github/stars/Kong/kong) https://github.com/Kong/kong
* ngnix
* haproxy - http://www.haproxy.org/
* ![stars](https://img.shields.io/github/stars/github/glb-director) https://github.com/github/glb-director - Github, Layer 4 load balancer
* https://code.fb.com/open-source/open-sourcing-katran-a-scalable-network-load-balancer/ - Facebook Katran
* https://cloudplatform.googleblog.com/2016/03/Google-shares-software-network-load-balancer-design-powering-GCP-networking.html - Google
* ![stars](https://img.shields.io/github/stars/dariubs/awesome-proxy) https://github.com/dariubs/awesome-proxy

Dockerized automated https reverse proxy:
* ![stars](https://img.shields.io/github/stars/jwilder/nginx-proxy) https://github.com/jwilder/nginx-proxy Automated nginx proxy for Docker containers using docker-gen
  * https://github.com/JrCs/docker-letsencrypt-nginx-proxy-companion LetsEncrypt companion container for nginx-proxy 
* ![stars](https://img.shields.io/github/stars/containous/traefik) https://github.com/containous/traefik https://traefik.io/ cloud native edge router. a modern HTTP reverse proxy and load balancer that makes deploying microservices easy
* ![stars](https://img.shields.io/github/stars/SteveLTN/https-portal) https://github.com/SteveLTN/https-portal A fully automated HTTPS server powered by Nginx, Let's Encrypt and Docker. 
* ![stars](https://img.shields.io/github/stars/sozu-proxy/sozu) https://github.com/sozu-proxy/sozu HTTP reverse proxy, configurable at runtime, fast and safe, built in Rust.
* ![stars](https://img.shields.io/github/stars/mholt/caddy) https://github.com/mholt/caddy https://caddyserver.com/docs/automatic-https (free if you build it yourself) Caddy automatically enables HTTPS for all your sites, given that some reasonable criteria. Fast, cross-platform HTTP/2 web server with automatic HTTPS 
* ![stars](https://img.shields.io/github/stars/Valian/docker-nginx-auto-ssl) https://github.com/Valian/docker-nginx-auto-ssl Docker image for automatic generation of SSL certs using Let's encrypt and Open Resty 

Discussions:
* https://news.ycombinator.com/item?id=17984970

Service registry and orchestrator:
* etcd
* consul

Logging, tracing, monitoring
* ![stars](https://img.shields.io/github/stars/uber-go/zap) https://github.com/uber-go/zap fast logging library for Go Zap
* http://opentracing.io/ - OpenTracing standard
* ![stars](https://img.shields.io/github/stars/jaegertracing/jaeger) https://github.com/jaegertracing/jaeger CNCF Jaeger, a Distributed Tracing System  https://uber.github.io/jaeger/ https://jaegertracing.io/
* ![stars](https://img.shields.io/github/stars/lightstep/lightstep-tracer-go) https://github.com/lightstep/lightstep-tracer-go Lightstep 
* ![stars](https://img.shields.io/github/stars/sourcegraph/appdash) https://github.com/sourcegraph/appdash Application tracing system for Go, based on Google's Dapper. (OpenTracing) https://sourcegraph.com
* ![stars](https://img.shields.io/github/stars/netdata/netdata) https://github.com/netdata/netdata Real-time performance monitoring, done right! https://www.netdata.cloud. Discussion: https://news.ycombinator.com/item?id=26886792

### Computing

* ![stars](https://img.shields.io/github/stars/ray-project/ray) 
https://github.com/ray-project/ray A system for parallel and distributed Python that unifies the ML ecosystem
* ![stars](https://img.shields.io/github/stars/dask/dask) 
https://github.com/dask/dask Parallel computing with task scheduling
* ![stars](https://img.shields.io/github/stars/vaexio/vaex) 
https://github.com/vaexio/vaex Out-of-Core DataFrames for Python, ML, visualize and explore big tabular data at a billion rows per second
* ![stars](https://img.shields.io/github/stars/rapidsai/cudf) 
https://github.com/rapidsai/cudf cuDF - GPU DataFrame Library
* ![stars](https://img.shields.io/github/stars/arrayfire/arrayfire) 
https://github.com/arrayfire/arrayfire a general purpose GPU library
* ![stars](https://img.shields.io/github/stars/pola-rs/polars) 
https://github.com/pola-rs/polars Fast multi-threaded DataFrame library in Rust and Python
* ![stars](https://img.shields.io/github/stars/dask/distributed) 
https://github.com/dask/distributed distributed dask
* ![stars](https://img.shields.io/github/stars/fugue-project/fugue) 
https://github.com/fugue-project/fugue A unified interface for distributed computing. Fugue executes SQL, Python, and Pandas code on Spark, Dask and Ray without any rewrites
* ![stars](https://img.shields.io/github/stars/databricks/spark-sklearn) 
https://github.com/databricks/spark-sklearn [ARCHIVED] Scikit-learn integration package for Spark

Resources:
* https://chryswoods.com/parallel_python/index.html - Parallel Programming with Python

### GPU/ML hosting and clouds

* https://vast.ai/
* https://www.runpod.io/
* https://clore.ai/

## Other resources

### Data sources

* Binance:
  * ![stars](https://img.shields.io/github/stars/binance-exchange/binance-official-api-docs) https://github.com/binance-exchange/binance-official-api-docs
  * https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md
  * ![stars](https://img.shields.io/github/stars/binance-exchange/python-binance) https://github.com/binance-exchange/python-binance

### Books

* Free data science books:
  * https://web.stanford.edu/~hastie/ElemStatLearn/ - The Elements of Statistical Learning: Data Mining, Inference, and Prediction
  * http://www.statlearning.com/ - An Introduction to Statistical Learning
  * https://d2l.ai/ - Dive into Deep Learning
  * https://www.inferentialthinking.com/ - Computational and Inferential Thinking. The Foundations of Data Science
  * http://www.cs.cornell.edu/jeh/book.pdf - Foundations of Data Science
  * https://otexts.org/fpp2/ - Forecasting: Principles and Practice
  * ![stars](https://img.shields.io/github/stars/christophM/interpretable-ml-book) https://github.com/christophM/interpretable-ml-book - Interpretable machine learning
  * ![stars](https://img.shields.io/github/stars/stas00/ml-engineering) https://github.com/stas00/ml-engineering Machine Learning Engineering Open Book

### Python:

* ![stars](https://img.shields.io/github/stars/pamoroso/free-python-books) https://github.com/pamoroso/free-python-books Python books free to read online or download
  * Discussion with more links: https://news.ycombinator.com/item?id=26759677
* How to make an awesome Python package in 2021: https://antonz.org/python-packaging/
  * Discussion with more links: https://news.ycombinator.com/item?id=26733423
* https://github.com/facebook/pyre-check Performant type-checking for python https://pyre-check.org/
