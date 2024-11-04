# Homework #2

# Student Information
- First Name: Mohammad Shayan Khan
- Last Name: Khan
- UIN: 667707825
- UIC Email: mkhan405@uic.edu

# Prerequisites
- Ensure Scala 2.12.18 is installed
- Compatible JDK is installed for Scala 2 (e.g. JDK 18)
- Apache Spark 3.5.3 is installed

## Installation
- Clone this Github repository
- In IntelliJ, go to `File > New > Project From Existing Sources` and open the project
- Navigate to project root and to `src/main/resources/application.conf` and replace the project location in `app.fileConfig.baseDir` :
  
  ```
   app {
      ...
  
      sparkModelConfig {
         ...
      }
  
      fileConfig {
          baseDir = "<PATH_TO_PROJECT>/CS441_HW2/src/main/resources"
          ...
      }
  }

  ```
- Navigate to `Build > Build Project` to build the project and install relevant dependencies
- Go to `src/main/scala/com/khan/spark/Driver` and run the `main` function to start the program

### Using CLI
If you are interested in running the programming via the CLI, you can perform the following steps:

- Navigate to the project's root directory
- Run the following command in the terminal: `sbt clean compile run`

## Video Discussion Link
Youtube: https://youtu.be/jtpDMff2-yk


## Project Implementation

This project utilizes Apache Spark to build a program for the parralel processing of a large corpus of text to enable the training of a neural network for text generation. This project
utilizes precomputed embeddings, which are generated from the Word2Vec model trained in the previous homework. To enable this parralel processing, Apache Spark's **Resilient Distributed
Datasets (RDDs)** are utilizes to transform the original large corpus of text into forms necessary for training. At a high-level, these are the following tasks performed by the program:

- Load precomputed embeddings as an RDD and generate mapping from token -> embedding
- Load text corpus as RDD
- Tokenize text corpus RDD using Byte Pair Encoding (BPE)
- Generate Input/Target pairs from tokenized text
- Embed Input Features using precomputed embeddings and generate one-hot encoded labels from targets
- Train Neural Network using DeepLearning4j's `SparkDl4jMultiLayer` Components

### Loading Embeddings and Generating Mappings
The precomputed embeddings are in a CSV file in the following format:

```
word, embeddings, word1, word2, word3, word4
```

- `word` is the textual representation of a word or sub-word (e.g. "new" or "iated")
- `embedding` is a ";"-delimited string representing the 128-dimensional embedding generated using the previously trained Word2Vec Model
- The remaining columns specify similar words determined by the model

For the purposes of this assignment, only the first two columns will be required. To read this document and generate the required RDD, the Spark SQL library is utilized in
the following manner:

```scala
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(embeddingPath)

    val dfRdd = df.rdd

    // Parse Embedding CSV Data
    val embeddingRdd = dfRdd.map(r => new EmbeddingInstance(r.get(0).toString, r.get(1).
      toString.split(";").map(_.toFloat)))
```

This will load the CSV as a Spark DataFrame, from which an RDD can be obtained. Then, each item in the RDD is mapped to an `EmbeddingInstance`, where the token and corresponding vector are
stored. The embedding vector is retrieved by splitting based on the ";" delimiter on the second column of the RDD, and converted to an `Array[Float]`. The `EmbeddingInstance` class has the
following definition:

```scala
class EmbeddingInstance(private val _token: String, private val _vector: Array[Float])  extends Serializable {
  def token: String = _token
  def vector: Array[Float] = _vector
}

```

With an `RDD[EmbeddingInstance]` now available, the following mappings are generated as RDDs:
- Token -> Embedding
- Index -> Token & Token -> Index

#### Generating Embedding & Index Mappings
These mappings are necessary when embedding the input features to prepare for training. To generate this, the `embeddingRDD` is transformed to a tuple of (token, embeddingVec):
```scala
    val tokenToEmbeddingRDD = embeddingRdd.map(i => (i.token, i.vector))
```
This RDD will then be converted to a broadcast and collected a map to allow for seamless lookups:
```scala
    val tokenToEmbeddingBroadcast = sparkContext.broadcast(tokenToEmbeddingRDD.collectAsMap())
```

In addition, we will also need to map a token to it's index in the vocabulary and vice versa since this is essential when generating one-hot encoded labels or retrieiving the corresponding
token predicted by the neural network during inference. This will also be generated as an RDD and broadcasted to executors for lookup since this will be a frequent operation:
```scala
    // Map index to token and token to index
    val indexToIndexRDD = embeddingRdd.zipWithIndex().map(i => (i._2, i._1.token))
    val tokenToIndexRDD = indexToIndexRDD.map((i) => (i._2, i._1))

    // Creating Broadcasts
    val tokenToIndexBroadcast = sc.broadcast(tokenToIndexRDD.collectAsMap())
    val indexToTokenBroadcast = sparkContext.broadcast(indexToIndexRDD.collectAsMap())
```

### Tokenizing Input Data
Initially, the input text is loaded as an `RDD[String]` by Spark such that each line of the input text is an entry in the RDD:

```
  inputRDD = {
    "The materials contained in the...",
    "Roosevelt, while President of the...",
    ...
  }
```

For the purposes of generating sliding window pairs of sufficient lengths, these entries were transformed such that entry in the RDD
contains 10 lines of the original text:

```
  inputRDD = {
    "<lines 1-10>",
    "<lines 11-20>",
    ...
  }
```
Each line in the RDD is then split, with their BPE Token ID computed, and joined back together to obtain the `tokenizedRDD`:
```
"abc def ghi" => ["abc", "def", "ghi"] => [305, 130, 927] => "305 130 927"
```

### Generating Input/Target Pairs

To generate the Input/Target Pairs, each line in the tokenized RDD is first split and sliding windows of length `windowLength + 1` with the configured stride are generated. These can be found in the `application.conf` file. 

```
"305 130 927 ...." => [305, 130, 927, ...] => [[305, 130, 927,...], [130, 927, 425, ...]]
```

Each of these generated windows is then mapped to generate Input/Output Pairs:

```
[305, 130, 927,...] => ([305, 130, 927, ..], 234)
```

In case the length of a window is smaller than the desired window size, they are padded with 0s for the remaining length:

```
[305, 130, 927,...] => ([305, 130, 927, 0, 0, 0, ..., 0], 234)
```

In summary, every line from the original `tokenizedRDD` is now transformed to the following:

```
"305 130 927 ...." => [
                        ([305, 130, 927, ...], 234),
                        ....
                      ]
```

### Embedding Input Features and Generating Output Labels

With the Input/Target Pairs now generated, we need to embed the input features and generate the output labels. The first step in this process is to map over all the sliding windows 
previously computed and extract a list of inputs and outputs:

```
[
    [
      ([305, 130, 927, ...], 234),          => Inputs = [[305, 130, 927, ...], [130, 927, ...]], Outputs = [234, 145, ...]
      ....
    ],
    ...

]
```

#### Embedding Input Features

Each token in the input sequences is mapped to a 128-dimensional embedding vector, with a corresponding vector of zero's in the case the token is not found. To do this, each token ID is first mapped to it's textual representation. Then, each textual representation is mapped to it's embedding vector. These results are then flat mapped and result in a vector of length `windowSize * embeddingDim`

```
[305, 130, 927, ...] => ["abc", "def", "ghi"] => [[132.4, ..., 0], [532.4, ..., 0], ... ] => [1.00, 43.0, ...., 423.3]
```

#### Generating Output Features

Similar to the process of embedding the input features, the token ID is first decoded to it's textual representation, and the index of the token is then retrieved using the mapping 
computed prior to this process:

```
234 => "def" => 1 => [0, 1, 0, ....]
```

The index is then used to generate a one-hot encoded label of size `vocabSize`, where the position with `1` indicates which token has been predicted. In this example, "def" is the second
item in the vocabulary, hence why the label has a 1 at index 1. 

### Training the Neural Network

The neural network comprises of the following layers:
- Input Layer (Input Dimensions: embeddingSize * windowSize, Output Dimensions: 256)
- Hidden Layer (Dimensions: 256)
- Output Layer (Input Dimensions: 256, Output Dimensions: VocabSize)

The hidden layer utilizes the RELU activation function while the output layer uses the SOFTMAX Activation Function. The Multi-Class Cross Entropy Loss Function is utilized on the output layer.

### Model Performance

The `ParameterAveragingTrainingMastger` by DeepLearning4j is utilized to train and fit the model in a distributed fashion. The system was tested locally, and due to processing and memory constraints,
the following parameters were applied:
- Batch Size: 10
- Epochs: 20
- Learning Rate: 0.01
- Hidden Layer Size: 20
- Number of Workers: 32
- Average Frequency: 10
- Vocabulary Size: 50

To train and effectively test the model, an 80/20 split was performed to generate a train and test dataset to evaluate the model performance for every epoch. Here are some of the relevant statistics
from the model training:

- Training Loss: 0
  - This may be attributed to the small dataset the model was fit on, indicating that model perhaps overfit
- Training Accuracy: 0
  - This may further confirm overfitting, since the model is not able to correctly predict for data outside it's training set
- Gradient Norms: Between 0.38575756549835205 and 0.6663835644721985
  - Indicates model is not impeded by vanishing or exploding gradients
- Learning Rate: Consistently stays at configured 0.01 learning rate
- Memory Usage: 100 - 300 MB

  This can be seen from the logs, where it shows how many bytes of the heap memory are used  
  ```
  17:35:46.287 [main] INFO Transformer -- Used Heap Memory: 324449
  17:35:46.287 [main] INFO Transformer -- Total Heap Memory: 2282540236
  17:35:46.287 [main] INFO Transformer -- Used Off-Heap Memory: 0
  17:35:46.287 [main] INFO Transformer -- Total Off-Heap Memory: 0
  ```
- Time Per Epoch: 600ms on Average
- CPU Utilization: 40-58%

