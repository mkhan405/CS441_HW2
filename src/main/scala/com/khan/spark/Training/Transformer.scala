package com.khan.spark.Training

import com.khan.spark.Embeddings.EmbeddingsCompute._
import com.khan.spark.Training.Listeners.ScoreLogger
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.collection.Map

object Transformer {
  val logger: Logger = LoggerFactory.getLogger("Transformer")

  /**
   * Constructs a multi-layer neural network model based on the provided application configuration.
   *
   * @param config The configuration object containing model parameters such as embedding dimensions,
   *               window size, vocabulary size, and training configurations.
   * @return A fully initialized MultiLayerNetwork model ready for training.
   *
   * This method performs the following tasks:
   * - Retrieves necessary configuration parameters including embedding size, window size, vocabulary size,
   *   and hidden layer size.
   * - Builds the neural network architecture using a sequence of layers:
   *   - A dense layer with ReLU activation for hidden representations.
   *   - An output layer using softmax activation for multi-class classification, configured with cross-entropy loss.
   * - Initializes the model's parameters and returns the constructed model.
   */
  private def buildModel(config: AppConfig) = {
    // Retrieve configuration params
    val embeddingSize = config.embeddingDim
    val windowSize = config.windowSize
    val vocabSize = config.vocabSize
    val hiddenLayerSize = config.sparkTrainingConfig.hiddenLayerSize

    // Initialize model
    val conf = new NeuralNetConfiguration.Builder()
      .seed(config.sparkTrainingConfig.seed)
      .updater(new Adam(config.sparkTrainingConfig.learningRate))  // Set learning rate
      .weightInit(WeightInit.XAVIER)  // Set learning rate init method
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(embeddingSize * windowSize)  // Flattened input feature dimension
        .nOut(hiddenLayerSize)                        // Hidden Layer Size
        .activation(Activation.RELU)
        .build())
      .layer(1, new OutputLayer.Builder()
        .nIn(hiddenLayerSize)                         // Hidden Layer Size
        .nOut(vocabSize)                  // Output for each member of the vocabulary
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT)
        .build())
      .setInputType(InputType.feedForward(embeddingSize * windowSize))
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    model
  }

  /**
   * Trains a multi-layer neural network model using a distributed Spark framework.
   *
   * @param sc The SparkContext for managing the Spark cluster.
   * @param fs The FileSystem interface for saving the trained model.
   * @param config The configuration object containing application parameters such as training settings and file paths.
   * @param dataset The RDD containing the dataset used for training the model.
   * @return The trained MultiLayerNetwork model after completing the training process.
   *
   * This method performs the following steps:
   * - Builds a neural network model based on the provided configuration.
   * - Configures the training process using parameter averaging and sets up a SparkDl4jMultiLayer instance.
   * - Splits the dataset into training and testing subsets.
   * - Trains the model over a specified number of epochs, logging training statistics and performance metrics.
   * - Evaluates the model on the test dataset and logs accuracy and confusion matrix details.
   * - Saves the trained model to the specified output path upon completion.
   */
  def train(sc: SparkContext, fs: FileSystem, config: AppConfig, dataset: RDD[DataSet]): MultiLayerNetwork = {
    val model: MultiLayerNetwork = buildModel(config)
    val modelParams = config.sparkTrainingConfig

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(modelParams.numWorkers)
      .batchSizePerWorker(modelParams.batchSize)
      .averagingFrequency(modelParams.avgFreq)
      .workerPrefetchNumBatches(modelParams.numPrefetchBatches)
      .collectTrainingStats(true)
      .build()

    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)
    sparkModel.setCollectTrainingStats(true)

    // Add ScoreLogger Listener
    sparkModel.setListeners(new ScoreLogger(10, logger))

    // Perform 80/20 train/test split
    val splitDataset = dataset.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val trainDataset = splitDataset.head
    val testDataset = splitDataset.last

    for (i <- 0 to modelParams.numEpochs) {
      val startTime = System.currentTimeMillis()
      sparkModel.fit(trainDataset)
      val stats = sparkModel.getSparkTrainingStats

      if (stats != null) {
        // Log Training Stats
        logger.info(stats.statsAsString())

        // Collect Spark Stats
        val sparkMetrics = sc.statusTracker.getExecutorInfos
        sparkMetrics.foreach(j => {
          logger.info(s"Used Heap Memory: ${j.usedOnHeapStorageMemory()}")
          logger.info(s"Total Heap Memory: ${j.totalOnHeapStorageMemory()}")

          logger.info(s"Used Off-Heap Memory: ${j.usedOffHeapStorageMemory()}")
          logger.info(s"Total Off-Heap Memory: ${j.totalOffHeapStorageMemory()}")
        })
      }

      // Report Spark Statistics
      logger.info(s"Number of Executors: ${sc.getExecutorMemoryStatus.size}")

      // Log learning rate
      logger.info(s"Learning Rate: ${sparkModel.getNetwork.getLearningRate(0)}")

      // Evaluate Training Accuracy
      val eval: Evaluation = sparkModel.evaluate(testDataset)

      logger.info(s"Epoch ${i} Completed In: ${System.currentTimeMillis() - startTime}")
      logger.info(s"Training Accuracy: ${eval.accuracy()}")
      logger.info(s"Confusion Matrix: ${eval.confusionToString()}")
      logger.info(s"True Positive Rate: ${eval.truePositives()}")
      logger.info(s"True Negative Rate: ${eval.trueNegatives()}")
      logger.info(s"False Positive Rate: ${eval.falsePositives()}")
      logger.info(s"False Negative Rate: ${eval.falseNegatives()}")
    }

    // Save Model Once Completed
    val modelPath = new Path(config.fileConfig.baseDir, config.fileConfig.outputFilePath)
    val outputStream = fs.create(modelPath)
    ModelSerializer.writeModel(sparkModel.getNetwork, outputStream, true)

    // Return model
    sparkModel.getNetwork
  }

  /**
   * Tokenizes an input query string and ensures that it meets a fixed length requirement by padding or truncating.
   *
   * @param query The input query string to be tokenized.
   * @param config The configuration object containing parameters such as `windowSize` and `padToken` for padding.
   * @return An array of tokenized strings with a fixed length of `config.windowSize`.
   *         If the number of tokens in the query exceeds `config.windowSize`, the array is truncated.
   *         If the number of tokens is less than `config.windowSize`, the array is padded with the `padToken` to meet the required length.
   */
  def tokenizeQuery(query: String, config: AppConfig): Array[String] = {
    // Split and tokenize query
    val tokenizedQuery = splitAndTokenize(query).split(" ")
    // Consider maximum number of tokens by window, or pad the remaining space with the configured padToken
    if (tokenizedQuery.length > 20)
      tokenizedQuery.slice(0, 20)
    else (tokenizedQuery ++
      Array.fill(config.windowSize - tokenizedQuery.length)(config.padToken))
  }

  /**
   * Generates a prediction from a trained multi-layer neural network model based on a tokenized input query.
   *
   * @param model The trained MultiLayerNetwork model used for making predictions.
   * @param query The input query string to be tokenized and processed for prediction.
   * @param config The configuration object containing parameters such as `embeddingDim` and `windowSize`.
   * @param tokenToEmbeddingBroadcast A broadcast variable mapping tokens to their corresponding embeddings.
   * @param indexToTokenBroadcast A broadcast variable mapping token indices to their original string representation.
   * @return The predicted output token as a string. If the predicted token index is not found in the mapping,
   *         "NULL" is returned.
   */
  def predict(model: MultiLayerNetwork, query: String, config: AppConfig,
      tokenToEmbeddingBroadcast: Broadcast[Map[String, Array[Float]]],
      indexToTokenBroadcast:  Broadcast[Map[Long, String]]): String = {

    val input = tokenizeQuery(query, config)
    // Verify tokenized and truncated/padded input
    logger.debug(s"Text Input: ${input.mkString("Array(", ", ", ")")}")
    val modelInput = generateEmbedding(input, tokenToEmbeddingBroadcast, config.embeddingDim).reshape(1,
      config.embeddingDim * config.windowSize)
    // Verify Model Input and Shape
    logger.debug(s"Model Input: ${modelInput}")
    logger.debug(s"Model Input Shape: ${modelInput.shape().mkString("Array(", ", ", ")")}")
    // Predict Output from model
    val output = model.output(modelInput)
    // Utilize argMax to retrieve token index, and use precomputed mapping to return result
    indexToTokenBroadcast.value.getOrElse(Nd4j.argMax(output, 1).getInt(0).toLong, "NULL")
  }
}
