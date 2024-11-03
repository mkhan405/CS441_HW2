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

### Leading Embeddings and Generating Mappings
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


