package com.khan.spark.Embeddings

import com.khan.spark.utils.EncodingData.encoding
import com.knuddels.jtokkit.api.IntArrayList
import org.apache.spark.broadcast.Broadcast
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.Map

object EmbeddingsCompute {

  /**
   * Splits a given string into words, computes their Byte Pair Encoding (BPE) token IDs,
   * and returns a space-delimited string of the corresponding token IDs.
   *
   * @param str The input string to be tokenized, expected to contain words separated by spaces.
   *
   * @return A space-delimited string of BPE token IDs corresponding to the input words.
   *         Each word in the input string is transformed into its respective token IDs using
   *         Byte Pair Encoding, which helps in efficiently representing subword units.
   *
   * The function first splits the input string into an array of words. For each word, it encodes
   * it into BPE token IDs, resulting in a sequence of token IDs for each word. These token IDs
   * are then concatenated into a single string, separated by spaces. This transformation is
   * useful for preparing input for models that require tokenized input data.
   */
  def splitAndTokenize(str: String): String = {
    // Split line, compute BPE Token IDs, join together to form string
    str.split(" ").map(t => encoding.encode(t)).map(bpe => bpe.toArray).map(_.mkString(" ")).mkString(" ")
  }

  /**
   * Generates positional embeddings for a sequence of tokens based on their positions within a window.
   *
   * @param windowSize The size of the sliding window, which determines the number of positions for which embeddings will be generated.
   * @param embeddingDim The dimensionality of the positional embeddings. This should be an even number to allow
   *                     for both sine and cosine components.
   *
   * @return An INDArray containing the positional embeddings for each position in the window, with shape (windowSize * embeddingDim).
   *
   * The positional embeddings are computed using sine and cosine functions of varying frequencies, which allows the model
   * to capture the relative positioning of tokens in the sequence. Specifically, for each position, sine and cosine
   * values are computed for each dimension of the embedding, alternating between the two to ensure that the embeddings
   * are unique for each position while maintaining a consistent pattern across dimensions.
   */
  def generatePositionalEmbedding(windowSize: Int, embeddingDim: Int): INDArray = {
    val result: INDArray = Nd4j.zeros(windowSize * embeddingDim)
    (0 until windowSize).foreach(pos => {
      (0 until embeddingDim by 2).foreach(i => {
        val angle: Double = pos / Math.pow(10000, (2.0 * i) / embeddingDim)
        result.putScalar(Array(pos + i), Math.sin(angle))
        if (i + 1 < embeddingDim)
          result.putScalar(Array(pos + i + 1), Math.cos(angle))
      })
    })
    result
  }

  /**
   * Generates a concatenated INDArray of embeddings for the provided tokens.
   *
   * @param tokens An array of strings representing the tokens for which embeddings are to be generated.
   * @param tokenToEmbeddingBroadcast A Broadcast variable containing a mapping from token strings to their corresponding
   *                                  float array embeddings.
   * @param embeddingDim The dimension of each embedding vector.
   *
   * @return An INDArray containing the concatenated embeddings for the input tokens. If a token does not have a corresponding
   *         embedding, a zero array of the specified embedding dimension is used in its place.
   *
   * This function takes an array of token strings, retrieves their embeddings from the broadcasted map, and constructs a
   * single INDArray by flattening the individual embeddings. The function handles tokens that do not have a predefined
   * embedding by substituting a zero vector of the specified embedding dimension, ensuring that the resulting INDArray
   * maintains a consistent shape.
   */
  def generateEmbedding(tokens: Array[String], tokenToEmbeddingBroadcast:Broadcast[Map[String, Array[Float]]],
                                embeddingDim: Int): INDArray = {

    // Map each token ID to flattened array of 128-dimensional embeddings
    val embeddings: Array[Float] = tokens.flatMap(t => {
      // Add TokenID to List
      val tokenList = new IntArrayList()
      tokenList.add(t.toInt)
      // Retrieve textual representation and fetch embedding for the token, or zero array if not found
      tokenToEmbeddingBroadcast.value.getOrElse(encoding.decode(tokenList), Array.fill(embeddingDim)(0.0f))
    })

    // Create Nd4j INDArray
    Nd4j.create(embeddings)
  }
}
