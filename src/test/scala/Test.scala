import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

import com.khan.spark.Training.AppConfig
import com.khan.spark.Embeddings.EmbeddingsCompute
import com.khan.spark.Training.Transformer.tokenizeQuery

import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.factory.Nd4j

class Test extends AnyFunSuite with BeforeAndAfterAll {
  val conf: SparkConf = new SparkConf().setAppName("CS 411 HW2").setMaster("local[*]")
  val sparkContext: SparkContext = new SparkContext(conf)

  // Stop spark context after tests are done
  override def afterAll(): Unit = {
    sparkContext.stop()
    // Call the parent implementation
    super.afterAll()
  }

  test("splitAndTokenize should return string of integers of size at least of the original array") {
    val input = "hello world"
    val result = EmbeddingsCompute.splitAndTokenize(input)
    assert(result.length >= input.length)
    result.split(" ").forall(t => t.forall(_.isDigit))
  }

  test("generatePositionalEmbedding should return an INDArray of correct shape") {
    val tokens = Array("hello", "world")
    val windowSize = 2
    val embeddingDim = 4
    val result = EmbeddingsCompute.generatePositionalEmbedding(tokens, windowSize, embeddingDim)
    assert(result.shape().sameElements(Array(windowSize * embeddingDim)))
  }

  test("generateEmbedding should return valid shape") {
    val tokens = Array("1", "2")
    val embeddingMap = scala.collection.Map("1" -> Array(0.1f, 0.2f), "2" -> Array(0.3f, 0.4f))
    val broadcastMap = sparkContext.broadcast(embeddingMap)
    val result = EmbeddingsCompute.generateEmbedding(tokens, broadcastMap, 2)
    assert(result.shape().sameElements(Array(2 * 2)))
  }

  test("generateEmbedding should return zero array for missing tokens") {
    val tokens = Array("3") // Token not in the embedding map
    val embeddingMap = scala.collection.Map("1" -> Array(0.1f, 0.2f))
    val broadcastMap = sparkContext.broadcast(embeddingMap)
    val result = EmbeddingsCompute.generateEmbedding(tokens, broadcastMap, 2)
    val expected = Nd4j.create(Array(0.0f, 0.0f))
    assert(result.equalsWithEps(expected, 1e-5))
  }

  test("tokenizeQuery should truncate queries longer than 20 tokens") {
    val testConfig = AppConfig.load()
    // Create a query with more than 20 tokens
    val longQuery = (1 to 25).mkString(" ")  // "1 2 3 ... 25"

    val result = tokenizeQuery(longQuery, testConfig)

    // Verify length is exactly 20
    assert(result.length === 20)
  }

  test("tokenizeQuery should pad shorter queries with zeros up to windowSize") {
    val testConfig = AppConfig.load()
    // Create a query with less than 20 tokens
    val shortQuery = "this is a short query"  // 5 tokens
    val result = tokenizeQuery(shortQuery, testConfig)
    // Verify total length matches windowSize
    assert(result.length === testConfig.windowSize)
    // Verify there are "0"s in the array
    assert(result.exists(_ === "0"), "Array should contain padding zeros")
  }

}
