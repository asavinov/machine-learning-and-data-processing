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
* (12.8k) https://github.com/facebook/prophet Prophet https://facebook.github.io/prophet/ 
* (1.3k) https://github.com/ourownstory/neural_prophet Neural Network based Time-Series model
* (4k) https://github.com/alan-turing-institute/sktime A unified toolbox for machine learning with time series
* (893) https://github.com/alkaline-ml/pmdarima A statistical library designed to fill the void in Python's time series analysis capabilities, including the equivalent of R's auto.arima function

Stock price forecasting:
* (3.9k) https://github.com/huseinzol05/Stock-Prediction-Models
* (2.9k) https://github.com/borisbanushev/stockpredictionai

Lists:
* https://github.com/MaxBenChrist/awesome_time_series_in_python

### Spatial data

* (2.6k) https://github.com/geopandas/geopandas Python tools for geographic data
* (530) https://github.com/giswqs/leafmap A Python package for geospatial analysis and interactive mapping with minimal coding in a Jupyter environment

### Audio and sound

General audio processing libraries:
* (5.8k) https://github.com/facebookresearch/wav2letter Automatic Speech Recognition Toolkit
* (4.5k) https://github.com/librosa/librosa Librosa
* (1.8k) https://github.com/MTG/essentia Essentia http://essentia.upf.edu/
* (3.9k) https://github.com/tyiannak/pyAudioAnalysis Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications
* (739) https://github.com/keunwoochoi/kapre keras audio preprocessor that let you calculate STFT/melspectrograms directly in your keras model on the fly without saving them in your storage
* (158) https://github.com/bmcfee/resampy Efficient sample rate conversion in python

Pitch trackers:
* (551) https://github.com/marl/crepe REPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is state-of-the-art (as of 2018), outperforming popular pitch trackers such as pYIN and SWIPE https://arxiv.org/abs/1802.06182
* pYIN: https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf
* SWIPE: https://pdfs.semanticscholar.org/0fd2/6e267cfa9b6d519967ea00db4ffeac272777.pdf

Other
* (575) https://github.com/AddictedCS/soundfingerprinting audio acoustic fingerprinting fingerprinting in C# (advertising etc. - not speech)

Books:
* Think DSP: Digital Signal Processing in Python: http://greenteapress.com/thinkdsp/thinkdsp.pdf
  * Include a Python lib and examples in Python: https://github.com/AllenDowney/ThinkDSP 
* Digital Signal Processing: https://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/index.html
* Sound Processing with Short Time Fourier Transform: https://www.numerical-tours.com/matlab/audio_1_processing/

### Text NLP

* (22.6k) https://github.com/facebookresearch/fastText - Library for fast text representation and classification
* (20.5k) https://github.com/explosion/spaCy (in Cython) -  Industrial-strength Natural Language Processing (NLP) with Python and Cython https://spacy.io
* (12.1k) https://github.com/RaRe-Technologies/gensim Topic Modelling for Humans, robust semantic analysis, topic modeling and vector-space modeling
* (46.6k) https://github.com/huggingface/pytorch-transformers A library of state-of-the-art pretrained models for Natural Language Processing (NLP)
* (9.9k) https://github.com/nltk/nltk - NLTK
* (10.4k) https://github.com/zalandoresearch/flair A very simple framework for state-of-the-art Natural Language Processing (NLP)
* (4.6k) https://github.com/wireservice/csvkit
* (5.2k) https://github.com/deepmipt/DeepPavlov - building end-to-end dialog systems and training chatbots

Lists:
* https://github.com/keon/awesome-nlp
* https://github.com/astorfi/Deep-Learning-NLP
* https://github.com/brianspiering/awesome-dl4nlp

### Video

* (7.5k) https://github.com/Zulko/moviepy - Video editing with Python
* (26.5k) https://github.com/iperov/DeepFaceLab DeepFaceLab is a tool that utilizes machine learning to replace faces in videos.

### Images

* (54.6k) https://github.com/opencv/opencv Open Source Computer Vision Library
* (16.2k) YOLO v4: https://github.com/AlexeyAB/darknet
  * https://pjreddie.com/darknet/yolo/
  * https://medium.com/what-is-artificial-intelligence/the-yolov4-algorithm-introduction-to-you-only-look-once-version-4-real-time-object-detection-5fd8a608b0fa
  * https://medium.com/@whats_ai/what-is-the-yolo-algorithm-introduction-to-you-only-look-once-real-time-object-detection-f26aa81475f2
  * How to train YOLO to detect your own objects: https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
* (13.6k) https://github.com/jantic/DeOldify A Deep Learning based project for colorizing and restoring old images (and video!)
* (8.5k) https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life Bringing Old Photo Back to Life (CVPR 2020 oral)
* (4.4k) https://github.com/scikit-image/scikit-image Image processing in Python
* (8.5k) https://github.com/python-pillow/Pillow PIL is the Python Imaging Library
* (1.4k) https://github.com/symisc/sod An Embedded, Modern Computer Vision & Machine Learning Library
* DeepDetect https://github.com/beniz/deepdetect/tree/master/demo/objsearch http://www.deepdetect.com/
  * Object similarity search: https://github.com/beniz/deepdetect/tree/master/demo/objsearch
    * for suggesting labels (bounding box)

### Graphs, RDFs etc.

Graph stores:
* (1.6k) https://github.com/eBay/beam A distributed knowledge graph store 
* (1.6k) https://github.com/gchq/Gaffer A large-scale entity and relation database supporting aggregation of properties 

Databases:
* (16.1k) https://github.com/dgraph-io/dgraph Dgraph Fast, Distributed Graph DB https://dgraph.io/
* Neo4j
* (4k) https://github.com/janusgraph/janusgraph https://janusgraph.org/ 
  JanusGraph [2] has support for a lot of different backends, built by the old team behind TitanDB
* (11.2k) https://github.com/arangodb/arangodb ArangoDB is a native multi-model database with flexible data models for documents, graphs, and key-values. Best open source graph database is ArrangoDB they have master to master cluster

Visualizations and dashboards:
* graphviz

### Statistics
* https://github.com/lebigot/uncertainties Transparent calculations with uncertainties on the quantities involved (aka "error propagation"); calculation of derivatives
* https://github.com/facebookarchive/bootstrapped Generate bootstrapped confidence intervals for A/B testing in Python


## Data and knowledge engineering

### Feature engineering

Time series:
* (5.7k) https://github.com/blue-yonder/tsfresh
* (2.2k) https://github.com/bukosabino/ta Technical Analysis Library using Pandas (Python)
* (421) https://github.com/benfulcher/hctsa Highly comparative time-series analysis code repository
* (101) https://github.com/chlubba/catch22 catch-22: CAnonical Time-series CHaracteristics

Feature extraction:
* (5.6k) https://github.com/Featuretools/featuretools
* (1.7k) https://github.com/mapbox/robosat - feature extraction from aerial and satellite imagery. Semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds
* (247) https://github.com/tyarkoni/pliers - Automated feature extraction in Python (audio/video)

Feature selection:
* (1.8k) https://github.com/WillKoehrsen/feature-selector Feature selector is a tool for dimensionality reduction of machine learning datasets.
  Find unnecessary (redundant) features using simple methods: missing values, high correlation etc.
  See article: https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
* (955) https://github.com/scikit-learn-contrib/boruta_py - Python implementations of the Boruta all-relevant feature selection method
* (317) https://github.com/EpistasisLab/scikit-rebate - A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning
* (695) https://github.com/abhayspawar/featexp eature exploration for supervised learning

Hyper-parameter optimization:
* (8k) https://github.com/EpistasisLab/tpot Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
* (4.6k) https://github.com/pfnet/optuna - A hyperparameter optimization framework https://optuna.org
* (1.5k) https://github.com/instacart/lore Lore makes machine learning approachable for Software Engineers and maintainable for Machine Learning Researchers
* (1.5k) https://github.com/ClimbsRocks/auto_ml [UNMAINTAINED] Automated machine learning for analytics & production
* (379) https://github.com/machinalis/featureforge - creating and testing machine learning features, with a scikit-learn compatible API

Lists:
* https://github.com/MaxHalford/xam - Personal data science and machine learning toolbox
* https://github.com/xiaoganghan/awesome-feature-engineering

### Feature stores

* (1.9k) https://github.com/feast-dev/feast Feature Store for Machine Learning https://feast.dev
* (492) https://github.com/logicalclocks/hopsworks Hopsworks - Data-Intensive AI platform with a Feature Store

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
* (9.7k) https://github.com/microsoft/nni An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning
* (8k) https://github.com/EpistasisLab/tpot A Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
* (8k) https://github.com/keras-team/autokeras AutoML library for deep learning
* (5.4k) https://github.com/automl/auto-sklearn Automated Machine Learning with scikit-learn
* (4.3k) https://github.com/google/automl Google Brain AutoML
* (3.3k) https://github.com/tensorflow/adanet Fast and flexible AutoML with learning guarantees
* (3.7k) https://github.com/mindsdb/mindsdb Machine Learning in one line of code
* (3.5k) https://github.com/pycaret/pycaret An open source, low-code machine learning library in Python
* (1.2k) https://github.com/AxeldeRomblay/MLBox MLBox is a powerful Automated Machine Learning python library
* (466) https://github.com/automl/HpBandSter a distributed Hyperband implementation on Steroids
* (28) https://github.com/shankarpandala/lazypredict Lazy Predict help build a lot of basic models without much code and helps understand which models works better without any parameter tuning
* (280) https://github.com/automl/autoweka

Systems and companies:
* DataRobot
* DarwinAI
* H2O.ai Dreverless AI
* OneClick.ai

### AI, data mining, machine learning algorithms

Resources:
* https://github.com/astorfi/TensorFlow-World-Resources
* https://github.com/astorfi/Deep-Learning-World

Algorithms:
* XGBoost, CatBoost, LightGBM
* (8.1k) https://github.com/spotify/annoy Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk
  * USP: ability to use static files as indexes, share index across process, that is, in-memory and efficient and multi-process
  * e.g, music recommendation in Spotify, similar images (for labeling etc.)  

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
* (21.6k) https://github.com/apache/airflow https://airflow.apache.org A platform to programmatically author, schedule, and monitor workflows. "Airflow is based on DAG representation and doesn’t have a concept of input or output, just of flow."
  * Tasks (Operators). Pluggable Python classes with many conventional execution types provided like `PythonOperator` or `BashOperator`. Note that these are task types which require custom task code for parameterization and instantiation. For `PythonOperator`, the custom code is provided as a Python function: `python_callable=my_task_fn`. Function arguments are passed in another parameter `op_kwargs`.
    * https://marclamberti.com/blog/airflow-pythonoperator/
  * Data. Small data sharing between tasks is performed via XCOM (cross-communication messages).
    * https://marclamberti.com/blog/airflow-xcom/
  * Dependencies. It is done programmatically and statically, for example: `my_task_1 >> my_task_2 >> [my_task_3, my_task_4]`
  * Scheduling. It is possible to specify `start_date` and `schedule_interval` (using CRON expression or `timedelta` object).
* (17.3k) https://github.com/celery/celery Distributed Task Queue http://celeryproject.org/
* (14.6k) https://github.com/spotify/luigi build complex pipelines of (long-running) batch jobs like Hadoop jobs, Spark jobs, dumping data to/from databases, running machine learning algorithms, Python snippet etc. "Luigi is based on pipelines of tasks that share input and output information and is target-based".
  * Dependencies. "Luigi doesn’t use DAGs. Instead, Luigi refers to "tasks" and "targets." Targets are both the results of a task and the input for the next task." "Luigi has 3 steps to construct a pipeline: `requires()` defines the dependencies between the tasks, `output()` defines the the target of the task, `run()` defines the computation performed by each task"
  * Scheduling. "Luigi ... has a central scheduler *and* custom calendar schedule capabilities, providing users with lots of flexibility" (in contrast to Airflow).
* (6.4k) https://github.com/prefecthq/prefect
* (3.7k) https://github.com/azkaban/azkaban Azkaban workflow manager
* Oozie 
* (911) https://github.com/d6t/d6tflow Python library for building highly effective data science workflows (on top of luigi)

ML workflow, pipelines, training, deployment etc.
* (10.3k) https://github.com/kubeflow/kubeflow Machine Learning Toolkit for Kubernetes
* (7.5k) https://github.com/cortexlabs/cortex Cloud native model serving infrastructure. Cortex is an open source platform for deploying machine learning models as production web services
* (4.4k) https://github.com/Netflix/metaflow Build and manage real-life data science projects with ease

Data science support and tooling
* (4.7k) https://github.com/drivendata/cookiecutter-data-science A logical, reasonably standardized, but flexible project structure for doing and sharing data science work

ETL:
* (1.7k) https://github.com/mara/data-integration A lightweight opinionated ETL framework, halfway between plain scripts and Apache Airflow
* (1.4k) https://github.com/python-bonobo/bonobo Extract Transform Load for Python 3.5+ https://www.bonobo-project.org/

Stream processing:
* (5.6k) https://github.com/robinhood/faust Stream processing library, porting the ideas from Kafka Streams to Python
* (2.5k) https://github.com/airbnb/streamalert StreamAlert is a serverless, realtime data analysis framework which empowers you to ingest, analyze, and alert on data from any environment, using datasources and alerting logic you define
* (1.6k) https://github.com/nerevu/riko A Python stream processing engine modeled after Yahoo! Pipes

Web scrapping
* (40.7k) https://github.com/scrapy/scrapy - create spiders bots that scan website pages and collect structured data

### Data labeling

* (4.6k) https://github.com/snorkel-team/snorkel A system for quickly generating training data with weak supervision
* (727) https://github.com/Cartucho/yolo-boundingbox-labeler-GUI Label images and video for Computer Vision applications (YOLO format)
* (1.6k) https://github.com/Labelbox/Labelbox https://www.labelbox.io/ Labelbox is the fastest way to annotate data to build and ship computer vision applications (non-YOLO format)
* https://github.com/beniz/deepdetect/tree/master/demo/objsearch
* labelmg (free)
* rectlabel (for Pascal VOC format)

### Visualization and VA

* Matplotlib http://matplotlib.org/
* Bokeh http://bokeh.pydata.org/en/latest/ The Bokeh library creates interactive and scalable visualizations in a browser using JavaScript widgets.
* Seaborn http://stanford.edu/~mwaskom/software/seaborn/# https://seaborn.pydata.org/ - a higher-level API based on the matplotlib library. It contains more suitable default settings for processing charts. Also, there is a rich gallery of visualizations. 
* Lightning http://lightning-viz.org/
* Plotly https://plot.ly/
* Pandas built-in plotting http://pandas.pydata.org/pandas-docs/stable/visualization.html
* HoloViews http://holoviews.org/
* VisPy http://vispy.org/
* pygg http://www.github.com/sirrice/pygg
* Altair https://github.com/altair-viz/altair
* hypertools: https://github.com/ContextLab/hypertools

Dashboards:
* Grafana (time series) https://grafana.com/  https://github.com/grafana/grafana 
  * Integration with InfluxDB
* Superset https://github.com/apache/superset Apache Superset is a Data Visualization and Data Exploration Platform
  * Integration with druid.io database https://github.com/apache/druid
* Metabase https://metabase.com/ https://github.com/metabase/metabase
* Redash https://redash.io/  https://github.com/getredash/redash
* Zeppelin Apache https://zeppelin.apache.org/ https://github.com/apache/zeppelin 
* Mixpanel https://mixpanel.com/
* Blazer https://github.com/ankane/blazer
* Bdash https://github.com/bdash-app/bdash
* Datamill, ?
* wagonhq, ?
* https://github.com/nocodb/nocodb The Open Source Airtable alternative. Turns any MySQL, PostgreSQL, SQL Server, SQLite & MariaDB into a smart-spreadsheet

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
    * (3.2k) https://github.com/reactor/reactor
    * http://reactivex.io/ An API for asynchronous programming with observable streams:
      * (44.2k) https://github.com/ReactiveX/RxJava Reactive Extensions for the JVM – a library for composing asynchronous and event-based programs using observable sequences for the Java VM
      * (3.8k) https://github.com/ReactiveX/RxPY Reactive Extensions for Python
      * etc. https://github.com/ReactiveX

* Actor model:
  * Each element has an identifier (address, reference) which is used by other actors to send messages
  * A sender (producer) must know what it wants to do when it sends messages and it must know its destination
  * Receiving elements can receive messages from any other element and their task is to respond according to their logic (as expected by the sender)
  * Actors receive messages without subscribing to any producer (as opposed to the actor model where you will not receive anything until you subscribe to some event producer)
  * Each actor has a handler (callback) which is invoked for processing incoming messages
  * Actors are supposed to have a state and frequently it is why we want to define different actors
  * Resources:
    * (11.4k) https://github.com/akka/akka Build highly concurrent, distributed, and resilient message-driven applications on the JVM
    * (11.8k) https://github.com/eclipse-vertx/vert.x Vert.x is a tool-kit for building reactive applications on the JVM
    * (1.9k) [ARCHIVED] https://github.com/quantmind/pulsar Event driven concurrent framework for Python https://quantmind.github.io/pulsar/index.html Pulsar implements two layers of components on top of python asyncio module: the actor layer and the application framework
    * (931) https://github.com/jodal/pykka Python implementation of the actor model, which makes it easier to build concurrent applications
    * (142) https://github.com/kquick/Thespian Python Actor concurrency library
    * https://gitlab.com/python-actorio/actorio Actorio - a simple actor framework for asyncio

### Event loops vs. threads

* Both a thread task and an event loop task are executed until finished, that is, the code to execute is provided as a procedure
* Thread tasks are dispatched by the system (not application) while dispatching logic of event tasks is part of the application
* At each moment, there is a fixed number of threads concurrently executed by one process. The number of concurrently executed event loop tasks is not limited.
* Thread tasks are (automatically) switched at the instruction level and the dispatcher is unaware of the needs of this thread or the application. Event loop tasks are switched at the level of logical application units depending on what this application needs.
* In a multi-thread application, we need to manage the threads ourselves, e.g., by creating and deleting them. In an event loop application, the tasks (starting, suspending, finishing) is managed by the event loop manager.
* In an event loop application, tasks specify dependencies on other tasks, and these points are used while dispatching the execution of tasks. Threads cannot declare dependencies on the results provided by other tasks. If we need some external result, then the thread has to wait. This logic has to be implemented manually and the system dispatcher is unaware of these dependencies.

Event loops: 
* (16.7k) https://github.com/libuv/libuv Cross-platform asynchronous I/O
* (7.1k) https://github.com/libevent/libevent Event notification library
* (1k) https://github.com/enki/libev Full-featured high-performance event loop loosely modelled after libevent

Resources:
* https://github.com/timofurrer/awesome-asyncio Python asyncio

### Async networking libraries

* (5.5k) https://github.com/gevent/gevent coroutine - based Python networking library. "systems like gevent use lightweight threads to offer performance comparable to asynchronous systems, but they do not actually make things asynchronous"
  * greenlet to provide a high-level synchronous API 
    * on top of the libev or libuv event loop (like libevent)

* (1k) https://github.com/eventlet/eventlet concurrent networking library for Python
  * epoll or kqueue or libevent for highly scalable non-blocking I/O

* (11.2k) https://github.com/aio-libs/aiohttp Asynchronous HTTP client/server framework for asyncio and Python

* (4.3k) https://github.com/twisted/twisted Event-driven networking engine written in Python. 
  * Twisted projects variously support TCP, UDP, SSL/TLS, IP multicast, Unix domain sockets, many protocols (including HTTP, XMPP, NNTP, IMAP, SSH, IRC, FTP, and others), and much more.
  * Twisted supports all major system event loops:
    * select (all platforms), 
    * poll (most POSIX platforms), 
    * epoll (Linux), 
    * kqueue (FreeBSD, macOS), 
    * IOCP (Windows), 
    * various GUI event loops (GTK+2/3, Qt, wxWidgets)

### Async web frameworks

* (20k) https://github.com/tornadoweb/tornado Python web framework and asynchronous networking library 
  * "Tornado is integrated with the standard library asyncio module and shares the same event loop (by default since Tornado 5.0). In general, libraries designed for use with asyncio can be mixed freely with Tornado." 
  * Some async client Libraries built on tornado.ioloop:
    * DynamoDB, CouchDB, Hbase, MongoDB, MySQL, PostgresQL, PrestoDB, RethinkDB
    * AMQP, NATS, RabbitMQ, SMTP
    * DNS, Memcached, Reis
    * etc. https://github.com/tornadoweb/tornado/wiki/Links
* (31.6k) https://github.com/tiangolo/fastapi FastAPI
* (15k) https://github.com/huge-success/sanic Sanic
* (5.7k) https://github.com/vibora-io/vibora Like Sanic but even faster
* https://gitlab.com/pgjones/quart API compatible with Flask 
* (3.6k) https://github.com/Pylons/pyramid Python web framework https://trypyramid.com/ (it seems to be a conventional web framework)

### Utilities

Retry libraries:
* (3k) https://github.com/jd/tenacity - originates from a fork of retrying
* (1.7k) https://github.com/rholder/retrying - not supported anymore
* (1.5k) https://github.com/litl/backoff
* (394) https://github.com/invl/retry
* (252) https://github.com/channable/opnieuw

## Libraries, utilities, tools

### Python

Resources:
* Top 10 Python libraries of 2017: https://tryolabs.com/blog/2017/12/19/top-10-python-libraries-of-2017/
* https://github.com/vinta/awesome-python - A curated list of awesome Python frameworks, libraries, software and resources
* Checklist for Python libraries APIs: http://python.apichecklist.com/

### Tools

* Data structures:
  * https://github.com/mahmoud/boltons boltons should be builtins
  * https://github.com/pytoolz/toolz List processing tools and functional utilities (replaces itertools and functools)
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

* https://github.com/wesm/feather  fast, interoperable binary data frame storage for Python, R, and more powered by Apache Arrow
* https://github.com/apache/arrow standardized language-independent columnar memory format for flat and hierarchical data
* https://github.com/Blosc/bcolz A columnar data container that can be compressed
* cloudpickle: serialize Python constructs not supported by the default pickle module from the Python standard library (lambdas etc.)
* pytables (905): https://github.com/PyTables/PyTables - A Python package to manage extremely large amounts of data http://www.pytables.org
  * based on hdf5 https://www.hdfgroup.org/ HDF supports n-dimensional datasets and each element in the dataset may itself be a complex object.
* h5py (1.5k): https://github.com/h5py/h5py HDF5 for Python -- The h5py package is a Pythonic interface to the HDF5 binary data format

### Authentication, Authorization, Security

Identity and Access Management
* (11k) https://github.com/ory/hydra OAuth2 Server and OpenID Certified™ OpenID Connect Provider written in Go - cloud native, security-first, open source API security for your infrastructure. SDKs for any language
* (9k) https://github.com/keycloak/keycloak
* (3.8k) https://github.com/jpadilla/pyjwt JSON Web Token implementation in Python
* (3.5k) https://github.com/apache/shiro Apache Shiro is a powerful and easy-to-use Java security framework that performs authentication, authorization, cryptography, and session management
* (2.5k) https://github.com/lepture/authlib The ultimate Python library in building OAuth, OpenID Connect clients and servers. JWS,JWE,JWK,JWA,JWT included
* (1.8k) https://github.com/RichardKnop/go-oauth2-server A standalone, specification-compliant, OAuth2 server written in Golang. 
* (558) https://github.com/OpenIDC/pyoidc A Python OpenID Connect implementation
* https://www.gluu.org/

Secrets management, encryption as a service, and privileged access management. A secret is anything that you want to tightly control access to, such as API keys, passwords, certificates, and more.
* (21.2k) https://github.com/hashicorp/vault A tool for secrets management, encryption as a service, and privileged access management

Policy Enforcement Point, Identity And Access Proxy (IAP), Zero-Trust Network Architecture, i.e. a reverse proxy in front of your upstream API or web server that rejects unauthorized requests and forwards authorized ones to your server. 
* (2.2k) https://github.com/ory/oathkeeper ORY Oathkeeper 

Resources:
* (4.1k) https://github.com/dwyl/learn-json-web-tokens -  Learn how to use JSON Web Token (JWT) to secure your next Web App! (Tutorial/Example with Tests)

### Linux and OS

Resources:
* https://0xax.gitbooks.io/linux-insides/content/ - Linux inside
* https://john-millikin.com/unix-syscalls - UNIX Syscalls

### Platform and servers

Load balancing and proxy:
* (45.6k) https://github.com/fatedier/frp
* (33.8k) https://github.com/traefik/traefik
* (33.5k) https://github.com/caddyserver/caddy
* (29.3k) https://github.com/Kong/kong
* ngnix
* haproxy - http://www.haproxy.org/
* GLB Director - https://github.com/github/glb-director - Github, Layer 4 load balancer
* https://code.fb.com/open-source/open-sourcing-katran-a-scalable-network-load-balancer/ - Facebook Katran
* https://cloudplatform.googleblog.com/2016/03/Google-shares-software-network-load-balancer-design-powering-GCP-networking.html - Google
* https://github.com/dariubs/awesome-proxy

Dockerized automated https reverse proxy:
* https://github.com/jwilder/nginx-proxy Automated nginx proxy for Docker containers using docker-gen
  * https://github.com/JrCs/docker-letsencrypt-nginx-proxy-companion LetsEncrypt companion container for nginx-proxy 
* https://traefik.io/ https://github.com/containous/traefik/ - cloud native edge router. a modern HTTP reverse proxy and load balancer that makes deploying microservices easy
* https://github.com/SteveLTN/https-portal A fully automated HTTPS server powered by Nginx, Let's Encrypt and Docker. 
* https://github.com/sozu-proxy/sozu HTTP reverse proxy, configurable at runtime, fast and safe, built in Rust.
* https://caddyserver.com/docs/automatic-https https://github.com/mholt/caddy (free if you build it yourself) Caddy automatically enables HTTPS for all your sites, given that some reasonable criteria. Fast, cross-platform HTTP/2 web server with automatic HTTPS 
* https://github.com/Valian/docker-nginx-auto-ssl Docker image for automatic generation of SSL certs using Let's encrypt and Open Resty 

Discussions:
* https://news.ycombinator.com/item?id=17984970

Service registry and orchestrator:
* etcd
* consul

Logging, tracing, monitoring
* https://github.com/uber-go/zap fast logging library for Go Zap
* http://opentracing.io/ - OpenTracing standard
* CNCF Jaeger, a Distributed Tracing System  https://github.com/jaegertracing/jaeger https://uber.github.io/jaeger/ https://jaegertracing.io/
* https://github.com/lightstep/lightstep-tracer-go Lightstep 
* https://github.com/sourcegraph/appdash Application tracing system for Go, based on Google's Dapper. (OpenTracing) https://sourcegraph.com
* https://github.com/netdata/netdata Real-time performance monitoring, done right! https://www.netdata.cloud. Discussion: https://news.ycombinator.com/item?id=26886792

### Computing

* (16.1k) https://github.com/ray-project/ray A system for parallel and distributed Python that unifies the ML ecosystem (similar to Dask)
* (8.4k) https://github.com/dask/dask Parallel computing with task scheduling
* (6.3k) https://github.com/vaexio/vaex Out-of-Core DataFrames for Python, ML, visualize and explore big tabular data at a billion rows per second
* (3.5k) https://github.com/arrayfire/arrayfire a general purpose GPU library
* ()1.8k) https://github.com/pola-rs/polars Fast multi-threaded DataFrame library in Rust and Python
* (1.2k) https://github.com/dask/distributed distributed dask
* https://github.com/databricks/spark-sklearn [ARCHIVED] Scikit-learn integration package for Spark

Resources:
* https://chryswoods.com/parallel_python/index.html - Parallel Programming with Python


## Other resources

### Data sources

* Binance:
  * https://github.com/binance-exchange/binance-official-api-docs
  * https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md
  * https://github.com/binance-exchange/python-binance

### Books

* Free data science books:
  * https://web.stanford.edu/~hastie/ElemStatLearn/ - The Elements of Statistical Learning: Data Mining, Inference, and Prediction
  * http://www.statlearning.com/ - An Introduction to Statistical Learning
  * https://d2l.ai/ - Dive into Deep Learning
  * https://www.inferentialthinking.com/ - Computational and Inferential Thinking. The Foundations of Data Science
  * http://www.cs.cornell.edu/jeh/book.pdf - Foundations of Data Science
  * https://otexts.org/fpp2/ - Forecasting: Principles and Practice
  * https://github.com/christophM/interpretable-ml-book - Interpretable machine learning

### Python:

* https://github.com/pamoroso/free-python-books Python books free to read online or download
  * Discussion with more links: https://news.ycombinator.com/item?id=26759677
* How to make an awesome Python package in 2021: https://antonz.org/python-packaging/
  * Discussion with more links: https://news.ycombinator.com/item?id=26733423
* https://github.com/facebook/pyre-check Performant type-checking for python https://pyre-check.org/
